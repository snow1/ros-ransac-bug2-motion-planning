#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, numpy as np, random
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
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
        # Parameters (right-side wall following)
        self.right_deg_min = rospy.get_param("~right_deg_min", -90.0)
        self.right_deg_max = rospy.get_param("~right_deg_max", -25.0)
        self.min_points     = rospy.get_param("~min_points", 25)
        self.max_range      = rospy.get_param("~max_range", 6.0)
        self.min_range      = rospy.get_param("~min_range", 0.06)
        self.clear_front_deg= rospy.get_param("~clear_front_deg", 18.0)

        self.ransac_trials  = rospy.get_param("~ransac_trials", 120)
        self.inlier_thresh  = rospy.get_param("~inlier_thresh", 0.04)
        self.min_inliers    = rospy.get_param("~min_inliers", 20)
        self.max_rms        = rospy.get_param("~max_rms", 0.06)

        self.frame_id       = rospy.get_param("~frame_id", "laser")

        # Publishers (API expected by bug2_lock_v2.py)
        self.pub       = rospy.Publisher("wall/perception", Point32, queue_size=20)
        self.pub_flags = rospy.Publisher("wall/flags", Header,    queue_size=10)
        self.pub_viz   = rospy.Publisher("wall/visualization", Marker, queue_size=10)

    def _make_marker(self, id_, type_, color, scale=0.03):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "wall_perception"
        m.id = id_
        m.type = type_
        m.action = Marker.ADD
        # For POINTS: scale.x = point width, scale.y = point height
        # For LINE_*:  scale.x = line width
        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime = rospy.Duration(0.2)
        return m

    def scan_cb(self, scan: LaserScan):
        # --- Read scan ---
        n = len(scan.ranges)
        if n == 0:
            return
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        # --- Draw sector boundaries (visual aid) ---
        marker_sector = self._make_marker(4, Marker.LINE_LIST, (1, 1, 0, 0.5), 0.01)
        for deg in (self.right_deg_min, self.right_deg_max):
            rad = np.deg2rad(deg)
            x = np.cos(rad) * self.max_range
            y = np.sin(rad) * self.max_range
            marker_sector.points.append(Point32(0.0, 0.0, 0.0))
            marker_sector.points.append(Point32(x, y, 0.0))
        self.pub_viz.publish(marker_sector)

        # --- Front clearance (±clear_front_deg around 0°) ---
        front_mask = np.abs(np.rad2deg(angles)) <= self.clear_front_deg
        fr = ranges[front_mask]
        fr = fr[np.isfinite(fr)]
        front_clear = float(np.min(fr)) if fr.size > 0 else float('nan')

        # --- Right-sector extraction (NO mixed-length masking) ---
        deg_all = np.rad2deg(angles)
        sector_mask = (deg_all >= self.right_deg_min) & (deg_all <= self.right_deg_max)
        rr_sector = ranges[sector_mask]
        aa_sector = angles[sector_mask]

        if rr_sector.size == 0:
            # Publish “no wall” but still send front clearance / viz arrow
            self._publish_outputs(has_wall=0, wall_dist=float('nan'),
                                  wall_yaw=float('nan'), front_clear=front_clear,
                                  inlier_count=0, total_count=0, fit_rms=float('nan'))
            self._viz_front_arrow(front_clear)
            return

        valid = np.isfinite(rr_sector) & (rr_sector >= self.min_range) & (rr_sector <= self.max_range)
        rr = rr_sector[valid]
        aa = aa_sector[valid]
        total_count = int(valid.sum())

        xs = rr * np.cos(aa)  # ROS laser frame: x forward, y left
        ys = rr * np.sin(aa)

        # Visualize sector points (blue)
        marker_pts = self._make_marker(0, Marker.POINTS, (0, 0, 1, 1), 0.04)
        for x, y in zip(xs, ys):
            marker_pts.points.append(Point32(float(x), float(y), 0.0))
        self.pub_viz.publish(marker_pts)

        has_wall = 0
        wall_dist = float('nan')
        wall_yaw  = float('nan')
        fit_rms   = float('nan')
        inlier_count = 0

        pts = list(zip(xs, ys))
        best = None

        if len(pts) >= self.min_points:
            for _ in range(int(self.ransac_trials)):
                p1, p2 = random.sample(pts, 2)
                a, b, c = line_params(p1, p2)  # normalized (a,b)
                d = np.abs(a * xs + b * ys + c)
                inliers = d <= self.inlier_thresh
                cnt = int(inliers.sum())
                if cnt >= self.min_inliers:
                    rms = float(np.sqrt(np.mean(d[inliers] ** 2)))
                    if (best is None) or (cnt > best["cnt"]) or (cnt == best["cnt"] and rms < best["rms"]):
                        best = {"a": a, "b": b, "c": c, "cnt": cnt, "rms": rms}

        if best is not None and best["rms"] <= self.max_rms:
            has_wall = 1
            inlier_count = best["cnt"]
            fit_rms = best["rms"]
            a, b, c = best["a"], best["b"], best["c"]

            # Distance from origin (robot) to wall; sign not needed for right-follow
            wall_dist = abs(c)

            # Wall orientation via line normal angle; bug2 aligns this to 0
            wall_yaw = atan2(b, a)
            if wall_yaw > np.pi:  wall_yaw -= 2*np.pi
            if wall_yaw < -np.pi: wall_yaw += 2*np.pi

            # Inliers (green)
            marker_in = self._make_marker(1, Marker.POINTS, (0, 1, 0, 1), 0.05)
            d_all = np.abs(a * xs + b * ys + c)
            idx = np.where(d_all <= self.inlier_thresh)[0]
            for i in idx:
                marker_in.points.append(Point32(float(xs[i]), float(ys[i]), 0.0))
            self.pub_viz.publish(marker_in)

            # Wall line visualization (yellow)
            marker_line = self._make_marker(2, Marker.LINE_STRIP, (1, 1, 0, 1), 0.02)
            # Draw a long segment passing near the robot according to ax + by + c = 0
            dx, dy = b, -a  # direction vector along the wall
            # pick a point on the line closest to origin
            if abs(b) > 1e-6:
                x0 = 0.0
                y0 = -c / b
            else:
                y0 = 0.0
                x0 = -c / a if abs(a) > 1e-6 else 0.0
            L = 3.0
            marker_line.points.append(Point32(float(x0 - dx*L), float(y0 - dy*L), 0.0))
            marker_line.points.append(Point32(float(x0 + dx*L), float(y0 + dy*L), 0.0))
            self.pub_viz.publish(marker_line)

        # Front clearance arrow
        self._viz_front_arrow(front_clear)

        # Publish outputs expected by bug2
        self._publish_outputs(has_wall, wall_dist, wall_yaw, front_clear,
                              inlier_count, total_count, fit_rms)

    def _viz_front_arrow(self, front_clear):
        if isfinite(front_clear):
            marker_arrow = self._make_marker(3, Marker.ARROW, (0, 1, 1, 1), 0.05)
            marker_arrow.points.append(Point32(0.0, 0.0, 0.0))
            marker_arrow.points.append(Point32(float(front_clear), 0.0, 0.0))
            self.pub_viz.publish(marker_arrow)

    def _publish_outputs(self, has_wall, wall_dist, wall_yaw, front_clear,
                         inlier_count, total_count, fit_rms):
        pt = Point32(
            x=wall_dist if isfinite(wall_dist) else float('nan'),
            y=wall_yaw  if isfinite(wall_yaw)  else float('nan'),
            z=front_clear if isfinite(front_clear) else float('nan')
        )
        self.pub.publish(pt)

        h = Header()
        h.seq = int(has_wall)
        # encode “front obstacle” for bug2: 1 if too close, else 0
        h.stamp.secs = 1 if (isfinite(front_clear) and front_clear < 0.40) else 0
        h.frame_id = f"{inlier_count}/{total_count}|{fit_rms if isfinite(fit_rms) else 'nan'}"
        self.pub_flags.publish(h)

    def run(self):
        rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size=1)
        rospy.loginfo("perception_lock_v2 started")
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("perception_lock_v2")
    WallPerception().run()
