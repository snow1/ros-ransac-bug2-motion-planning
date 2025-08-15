#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, numpy as np, random
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from math import atan2, sqrt, isfinite, isnan

def line_params(p1, p2):
    # 二维点 → 直线 ax+by+c=0
    (x1,y1),(x2,y2)=p1,p2
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    n = sqrt(a*a + b*b) + 1e-9
    return a/n, b/n, c/n  # 归一化

def point_line_dist(a,b,c,x,y):
    return abs(a*x + b*y + c)

class WallPerception:
    def __init__(self):
        # 右侧扇区（贴右墙）
        self.right_deg_min = rospy.get_param("~right_deg_min", -90.0)
        self.right_deg_max = rospy.get_param("~right_deg_max", -25.0)   # 略放宽
        self.min_points = rospy.get_param("~min_points", 25)
        self.max_range = rospy.get_param("~max_range", 6.0)
        self.min_range = rospy.get_param("~min_range", 0.06)
        self.clear_front_deg = rospy.get_param("~clear_front_deg", 18.0)

        # RANSAC 参数
        self.ransac_trials = rospy.get_param("~ransac_trials", 120)
        self.inlier_thresh = rospy.get_param("~inlier_thresh", 0.04)     # 点到直线距离阈值
        self.min_inliers = rospy.get_param("~min_inliers", 20)
        self.max_rms = rospy.get_param("~max_rms", 0.06)                 # 质量判别

        # 发布
        self.pub = rospy.Publisher("wall/perception", Point32, queue_size=20)  # x=wall_dist, y=wall_yaw(rad), z=front_clearance
        self.pub_flags = rospy.Publisher("wall/flags", Header, queue_size=10)  # seq=has_wall, stamp.secs=obs_front, frame_id放质量

    def scan_cb(self, scan: LaserScan):
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n)*scan.angle_increment
        ranges = np.array(scan.ranges, dtype=np.float32)

        # 前向净空
        front_mask = np.abs(np.rad2deg(angles)) <= self.clear_front_deg
        fr = ranges[front_mask]
        front_clear = np.nanmin(fr) if np.any(np.isfinite(fr)) else np.nan

        # 右侧点集
        deg = np.rad2deg(angles)
        mask = (deg >= self.right_deg_min) & (deg <= self.right_deg_max)
        rr, aa = ranges[mask], angles[mask]
        valid = np.isfinite(rr) & (rr >= self.min_range) & (rr <= self.max_range)
        rr, aa = rr[valid], aa[valid]

        has_wall = 0
        wall_dist = np.nan
        wall_yaw  = np.nan
        fit_rms = np.nan
        inlier_count = 0
        total_count = int(np.sum(valid))

        if total_count >= self.min_points:
            xs = rr * np.cos(aa)
            ys = rr * np.sin(aa)
            pts = list(zip(xs, ys))
            best = None

            # RANSAC
            for _ in range(self.ransac_trials):
                p1, p2 = random.sample(pts, 2)
                a,b,c = line_params(p1, p2)
                dists = np.abs(a*xs + b*ys + c)
                inliers = dists <= self.inlier_thresh
                cnt = int(np.sum(inliers))
                if cnt >= self.min_inliers:
                    rms = float(np.sqrt(np.mean(dists[inliers]**2)))
                    if (best is None) or (cnt > best["cnt"]) or (cnt==best["cnt"] and rms < best["rms"]):
                        best = {"a":a,"b":b,"c":c,"cnt":cnt,"rms":rms}

            if best is not None and best["rms"] <= self.max_rms:
                has_wall = 1
                inlier_count = best["cnt"]
                fit_rms = best["rms"]
                a,b,c = best["a"], best["b"], best["c"]
                # 原点到直线距离
                wall_dist = abs(c)
                # 直线法向 (a,b) 相对 x 轴的角度
                wall_yaw = atan2(b, a)
                # 归一 [-pi, pi]
                if wall_yaw > np.pi:
                    wall_yaw -= 2*np.pi
                if wall_yaw < -np.pi:
                    wall_yaw += 2*np.pi

        # 发布数据
        pt = Point32(x=wall_dist if isfinite(wall_dist) else float("nan"),
                     y=wall_yaw if isfinite(wall_yaw) else float("nan"),
                     z=front_clear if isfinite(front_clear) else float("nan"))
        self.pub.publish(pt)

        # flags: seq=has_wall, stamp.secs=obs_front(阈值0.4m)
        h = Header()
        h.seq = has_wall
        h.stamp.secs = 1 if (isfinite(front_clear) and front_clear < 0.4) else 0
        # 把质量指标放到 frame_id，便于日志： "inliers/total|rms"
        h.frame_id = f"{inlier_count}/{total_count}|{fit_rms if isfinite(fit_rms) else 'nan'}"
        self.pub_flags.publish(h)

    def run(self):
        rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size=1)
        rospy.loginfo("perception_lock_v2 started")
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("perception_lock_v2")
    WallPerception().run()