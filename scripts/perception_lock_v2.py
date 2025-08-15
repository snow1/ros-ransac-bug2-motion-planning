#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, numpy as np, random
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
from std_msgs.msg import Header, Float32
from visualization_msgs.msg import Marker
from math import atan2, sqrt, isfinite, isnan

def line_params(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    n = sqrt(a*a + b*b) + 1e-9
    return a/n, b/n, c/n

class WallPerception:
    def __init__(self):
        # Parameters
        self.right_deg_min = rospy.get_param("~right_deg_min", -90.0)
        self.right_deg_max = rospy.get_param("~right_deg_max", -25.0)
        self.min_points = rospy.get_param("~min_points", 25)
        self.max_range = rospy.get_param("~max_range", 6.0)
        self.min_range = rospy.get_param("~min_range", 0.06)
        self.clear_front_deg = rospy.get_param("~clear_front_deg", 18.0)
        self.ransac_trials = rospy.get_param("~ransac_trials", 120)
        self.inlier_thresh = rospy.get_param("~inlier_thresh", 0.04)
        self.min_inliers = rospy.get_param("~min_inliers", 20)
        self.max_rms = rospy.get_param("~max_rms", 0.06)
        self.frame_id = rospy.get_param("~frame_id", "laser")

        # Publishers
        self.pub = rospy.Publisher("wall/perception", Point32, queue_size=20)
        self.pub_flags = rospy.Publisher("wall/flags", Header, queue_size=10)
        #self.pub_lateral = rospy.Publisher("/wall_lateral", Float32, queue_size=10)
        self.pub_viz = rospy.Publisher("wall/visualization", Marker, queue_size=10)

    def _make_marker(self, id, type, color, scale=0.03):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "wall_perception"
        m.id = id
        m.type = type
        m.action = Marker.ADD
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime = rospy.Duration(0.2)
        return m

    def scan_cb(self, scan: LaserScan):
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=np.float32)

        # Sector boundary lines
        marker_sector = self._make_marker(4, Marker.LINE_LIST, (1, 1, 0, 0.5), 0.01)
        for deg in [self.right_deg_min, self.right_deg_max]:
            rad = np.deg2rad(deg)
            x = np.cos(rad)
            y = np.sin(rad)
            marker_sector.points.append(Point32(0, 0, 0))
            marker_sector.points.append(Point32(x * self.max_range, y * self.max_range, 0))
        self.pub_viz.publish(marker_sector)

        # Front clearance
        front_mask = np.abs(np.rad2deg(angles)) <= self.clear_front_deg
        fr = ranges[front_mask]
        front_clear = np.nanmin(fr) if np.any(np.isfinite(fr)) else np.nan

        # Right sector points
        deg = np.rad2deg(angles)
        mask = (deg >= self.right_deg_min) & (deg <= self.right_deg_max)
        rr, aa = ranges[mask], angles[mask]
        valid = np.isfinite(rr) & (rr >= self.min_range) & (rr <= self.max_range)
        rr, aa = rr[valid], aa[valid]
        xs = rr * np.cos(aa)
        ys = rr * np.sin(aa)

        # Visualize right sector points (blue)
        marker_pts = self._make_marker(0, Marker.POINTS, (0, 0, 1, 1))
        for x, y in zip(xs, ys):
            marker_pts.points.append(Point32(x, y, 0))
        self.pub_viz.publish(marker_pts)

        has_wall = 0
        wall_dist = np.nan
        wall_yaw = np.nan
        fit_rms = np.nan
        inlier_count = 0
        total_count = int(np.sum(valid))

        best = None
        pts = list(zip(xs, ys))
        if total_count >= self.min_points:
            for _ in range(self.ransac_trials):
                p1, p2 = random.sample(pts, 2)
                a, b, c = line_params(p1, p2)
                dists = np.abs(a * xs + b * ys + c)
                inliers = dists <= self.inlier_thresh
                cnt = int(np.sum(inliers))
                if cnt >= self.min_inliers:
                    rms = float(np.sqrt(np.mean(dists[inliers] ** 2)))
                    if (best is None) or (cnt > best["cnt"]) or (cnt == best["cnt"] and rms < best["rms"]):
                        best = {"a": a, "b": b, "c": c, "cnt": cnt, "rms": rms}

        if best is not None and best["rms"] <= self.max_rms:
            has_wall = 1
            inlier_count = best["cnt"]
            fit_rms = best["rms"]
            a, b, c = best["a"], best["b"], best["c"]

            wall_dist = abs(c)
            wall_yaw = atan2(b, a)
            if wall_yaw > np.pi:
                wall_yaw -= 2 * np.pi
            if wall_yaw < -np.pi:
                wall_yaw += 2 * np.pi

            # Inlier points (green)
            marker_inliers = self._make_marker(1, Marker.POINTS, (0, 1, 0, 1))
            inliers_idx = np.where(np.abs(a * xs + b * ys + c) <= self.inlier_thresh)[0]
            for i in inliers_idx:
                marker_inliers.points.append(Point32(xs[i], ys[i], 0))
            self.pub_viz.publish(marker_inliers)

            # Wall line (red)
            marker_line = self._make_marker(2, Marker.LINE_STRIP, (1, 1, 0, 1), 0.01)
            x0, y0 = 0, -c / b if b != 0 else 0
            dx, dy = b, -a
            L = 2
            marker_line.points.append(Point32(x0 - dx * L, y0 - dy * L, 0))
            marker_line.points.append(Point32(x0 + dx * L, y0 + dy * L, 0))
            self.pub_viz.publish(marker_line)

        # Front clearance arrow (cyan)
        if isfinite(front_clear):
            marker_arrow = self._make_marker(3, Marker.ARROW, (0, 1, 1, 1), 0.05)
            marker_arrow.points.append(Point32(0, 0, 0))
            marker_arrow.points.append(Point32(front_clear, 0, 0))
            self.pub_viz.publish(marker_arrow)

        # Publish output
        pt = Point32(x=wall_dist if isfinite(wall_dist) else float("nan"),
                     y=wall_yaw if isfinite(wall_yaw) else float("nan"),
                     z=front_clear if isfinite(front_clear) else float("nan"))
        self.pub.publish(pt)

        h = Header()
        h.seq = has_wall
        h.stamp.secs = 1 if (isfinite(front_clear) and front_clear < 0.4) else 0
        h.frame_id = f"{inlier_count}/{total_count}|{fit_rms if isfinite(fit_rms) else 'nan'}"
        self.pub_flags.publish(h)

    def run(self):
        rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size=1)
        rospy.loginfo("perception_lock_viz started")
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("perception_lock_viz")
    WallPerception().run()
