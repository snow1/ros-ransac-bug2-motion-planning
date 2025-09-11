#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perception_user_opt.py
- Keeps your original logic and RViz visuals (sector outline, wall line, corner sphere)
- Publishes: /wall_angle, /wall_lateral, /wall_quality, /wall_has, /obstacle_ahead,
             /corner_has, /corner_point   (base_link), and NEW: /front_min, /left_min, /right_min
- Always publishes booleans each scan (no silent frames)
- Optional "front wedge" only for the second line (corner) so you can narrow main sector later
"""

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool, Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class WallPerception:
    def __init__(self):
        # ---------------- 参数 ----------------
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.side = rospy.get_param("~side", "right")  # right / left
        self.sector_center_deg = rospy.get_param("~sector_center_deg", -90.0 if self.side=="right" else 90.0)
        self.sector_width_deg  = rospy.get_param("~sector_width_deg", 150.0)
        self.r_min = rospy.get_param("~r_min", 0.3)#最小量测距离（米），默认 0.12 m。距离小于它的点被视为无效（例如近端噪声、底盘反射）。
        self.r_max = rospy.get_param("~r_max", 8.0)
        self.min_points = rospy.get_param("~min_points", 30)

        # 质量门槛（主墙线，仅用于“宣布有墙”）
        self.min_span   = rospy.get_param("~min_span", 0.25)
        self.min_quality_keep = rospy.get_param("~min_quality_keep", 3.0)  # 弱时维持几帧（用于早期版本兼容）

        # 前向避障（布尔）
        self.front_stop = rospy.get_param("~front_stop", 0.55)

        # EMA 平滑
        self.ema_alpha_angle = rospy.get_param("~ema_alpha_angle", 0.3)
        self.ema_alpha_dist  = rospy.get_param("~ema_alpha_dist", 0.3)

        # RViz 可视化（保留）
        self.marker_frame = rospy.get_param("~marker_frame", "laser")  # 也可 base_link/map
        self.draw_length  = rospy.get_param("~draw_length", 1.0)       # 墙线显示长度（不影响算法）
        self.draw_sector = rospy.get_param("~draw_sector", True)        # 计算扇区轮廓可视化
        self.sector_edge_count = rospy.get_param("~sector_edge_count", 48)

        # RANSAC 参数（可在 launch 里调）
        self.ransac_iters  = rospy.get_param("~ransac_iters", 120)
        self.ransac_thresh = rospy.get_param("~ransac_thresh", 0.03)       # 点到线内点阈值（米）
        self.ransac_min_inliers = rospy.get_param("~ransac_min_inliers", 20)

        # 次线（用于角点）质量门槛，防止噪点成线
        self.ransac_iters2  = rospy.get_param("~ransac_iters2", 120)
        self.ransac_thresh2 = rospy.get_param("~ransac_thresh2", 0.03)
        self.ransac_min_inliers2 = rospy.get_param("~ransac_min_inliers2", 15)
        self.corner_min_inlier_ratio2 = rospy.get_param("~corner_min_inlier_ratio2", 0.25)
        self.corner_min_span2 = rospy.get_param("~corner_min_span2", 0.25)
        self.corner_ang_min_deg = rospy.get_param("~corner_ang_min_deg", 70.0)
        self.corner_ang_max_deg = rospy.get_param("~corner_ang_max_deg", 110.0)

        # 角点“位置约束”（右墙时在右前，左墙时在左前），给一点容差
        self.corner_require_forward = rospy.get_param("~corner_require_forward", False)
        self.corner_y_tol = rospy.get_param("~corner_y_tol", 0.10)

        # === 新增：角点前探楔形扇区，仅用于“次线/角点”（可关）===
        self.corner_front_enable = rospy.get_param("~corner_front_enable", False)
        self.corner_front_center_deg = rospy.get_param(
            "~corner_front_center_deg", -30.0 if self.side=="right" else +30.0)
        self.corner_front_width_deg  = rospy.get_param("~corner_front_width_deg", 50.0)

        # === 新增：给 bug2_v4 的净空指标（v3 不依赖） ===
        self.front_width_deg = rospy.get_param("~front_width_deg", 40.0)   # ±20°
        self.side_width_deg  = rospy.get_param("~side_width_deg", 60.0)    # 左右各±30°

        # 可选：为布尔话题加 latch，便于新订阅者立即获得状态
        self.latch_bools = rospy.get_param("~latch_bools", False)

        # TF
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # ---------------- 发布者 ----------------
        self.pub_angle   = rospy.Publisher("/wall_angle",   Float32, queue_size=10)
        self.pub_lateral = rospy.Publisher("/wall_lateral", Float32, queue_size=10)
        self.pub_quality = rospy.Publisher("/wall_quality", Float32, queue_size=10)  # 用“内点数”作质量
        self.pub_has     = rospy.Publisher("/wall_has",     Bool,    queue_size=10, latch=self.latch_bools)
        self.pub_obs     = rospy.Publisher("/obstacle_ahead", Bool,  queue_size=10)

        # 角点话题（保持）
        self.pub_corner_has = rospy.Publisher("/corner_has", Bool, queue_size=10, latch=self.latch_bools)
        self.pub_corner_pt  = rospy.Publisher("/corner_point", PointStamped, queue_size=10)

        # RViz Marker 发布（保持）
        self.pub_marker  = rospy.Publisher("visualization_marker", Marker, queue_size=20)

        # 新增：净空最小距离（供 bug2_v4 使用；v3 忽略）
        self.pub_front   = rospy.Publisher("/front_min",  Float32, queue_size=10)
        self.pub_left    = rospy.Publisher("/left_min",   Float32, queue_size=10)
        self.pub_right   = rospy.Publisher("/right_min",  Float32, queue_size=10)

        # 状态
        self.angle_s = None
        self.dist_s  = None
        self.quality_s = 0.0

        rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan, queue_size=1)

    # ---------------- 工具 ----------------
    def _hdr(self):
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.marker_frame
        return h

    def _laser_to_marker(self, x, y, z=0.0):
        """laser -> marker_frame；若 marker_frame==laser 则原样返回"""
        if self.marker_frame == "laser":
            return (x, y, z)
        try:
            tf = self.tf_buf.lookup_transform(self.marker_frame, "laser", rospy.Time(0), rospy.Duration(0.05))
            ps = PointStamped()
            ps.header.frame_id = "laser"
            ps.point.x, ps.point.y, ps.point.z = x, y, z
            pt = do_transform_point(ps, tf).point
            return (pt.x, pt.y, pt.z)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

    # --- 新增：某扇区（中心/宽度）最小距离 ---
    def _sector_min(self, angles, ranges, valid, center_deg, width_deg):
        c = np.deg2rad(center_deg); half = np.deg2rad(width_deg*0.5)
        m = valid & (np.abs(wrap(angles - c)) <= half)
        if not np.any(m):
            return float('nan')
        return float(np.nanmin(ranges[m]))

    # ---------------- 可视化：扇区轮廓（保留） ----------------
    def _publish_sector_outline(self, center_rad, half_rad):
        if not self.draw_sector:
            return
        angs = np.linspace(center_rad - half_rad, center_rad + half_rad,
                           max(8, int(self.sector_edge_count)))
        pts_outer = np.stack([self.r_max*np.cos(angs), self.r_max*np.sin(angs)], axis=1)
        pts_inner = np.stack([self.r_min*np.cos(angs[::-1]), self.r_min*np.sin(angs[::-1])], axis=1)
        poly = np.vstack([
            [self.r_min*np.cos(center_rad-half_rad), self.r_min*np.sin(center_rad-half_rad)],
            pts_outer,
            [self.r_min*np.cos(center_rad+half_rad), self.r_min*np.sin(center_rad+half_rad)],
            pts_inner
        ])

        mk = Marker()
        mk.header = self._hdr()
        mk.ns = "sector"
        mk.id = 99
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = 0.01
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = (0.1, 0.6, 1.0, 0.8)
        mk.pose.orientation.w = 1.0
        mk.points = []
        for xy in poly:
            pt = self._laser_to_marker(xy[0], xy[1], 0.0)
            if pt is not None:
                mk.points.append(Point(*pt))
        mk.lifetime = rospy.Duration(0.3)
        self.pub_marker.publish(mk)

    # ---------------- RANSAC 直线拟合（返回：模型与内点索引） ----------------
    def fit_line_ransac(self, pts, iters, thresh, min_inliers):
        n = len(pts)
        if n < max(2, min_inliers):
            return None, np.array([], dtype=np.int64)

        best_inliers = np.array([], dtype=np.int64)
        best_model = None
        for _ in range(int(iters)):
            i, j = np.random.choice(n, 2, replace=False)
            p1, p2 = pts[i], pts[j]
            v = p2 - p1
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            v = v / nv
            nrm = np.array([-v[1], v[0]])
            dists = np.abs((pts - p1) @ nrm)
            inliers = np.where(dists < thresh)[0].astype(np.int64)
            if inliers.size > best_inliers.size:
                best_inliers = inliers
                best_model = (p1, v, nrm)

        if best_model is None or best_inliers.size < min_inliers:
            return None, np.array([], dtype=np.int64)

        # 用内点再 PCA 精修方向与中心
        in_pts = pts[best_inliers]
        c = in_pts.mean(axis=0)
        X = in_pts - c
        _, V = np.linalg.eigh((X.T @ X) / max(len(in_pts)-1, 1))
        v = V[:, 1]
        if v[0] < 0:
            v = -v
        nrm = np.array([-v[1], v[0]])
        return (c, v, nrm), best_inliers

    # ---------------- RViz 绘制（保留） ----------------
    def _draw_line(self, c, v, color=(0.0,1.0,0.0,1.0), mid_id=1):
        L = max(0.5, float(self.draw_length))  # 固定长度显示更直观
        p1 = c - v*(L*0.5)
        p2 = c + v*(L*0.5)
        mk = Marker()
        mk.header = self._hdr()
        mk.ns = "wall"
        mk.id = mid_id
        mk.type = Marker.LINE_LIST
        mk.action = Marker.ADD
        mk.scale.x = 0.02
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = color
        mk.pose.orientation.w = 1.0
        p1t = self._laser_to_marker(p1[0], p1[1])
        p2t = self._laser_to_marker(p2[0], p2[1])
        if (p1t is not None) and (p2t is not None):
            mk.points = [Point(*p1t), Point(*p2t)]
            mk.lifetime = rospy.Duration(0.25)
            self.pub_marker.publish(mk)

    def _draw_corner(self, P):
        mk = Marker()
        mk.header = self._hdr()
        mk.ns = "corner"
        mk.id = 2
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.scale.x = mk.scale.y = mk.scale.z = 0.12
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = (1.0, 0.1, 0.1, 0.95)
        mk.pose.orientation.w = 1.0
        pt = self._laser_to_marker(P[0], P[1])
        if pt is not None:
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = pt
            mk.lifetime = rospy.Duration(0.25)
            self.pub_marker.publish(mk)

    # ---------------- 主回调 ----------------
    def cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > self.r_min) & (ranges < self.r_max)

        # 前方障碍（布尔）
        self.pub_obs.publish(self._front_obstacle(angles, ranges, valid))

        # 新增：扇区最小距离（Float32；可能为 NaN）
        fmin = self._sector_min(angles, ranges, valid, 0.0,   self.front_width_deg)
        lmin = self._sector_min(angles, ranges, valid, +90.0, self.side_width_deg)
        rmin = self._sector_min(angles, ranges, valid, -90.0, self.side_width_deg)
        self.pub_front.publish(float(fmin))
        self.pub_left .publish(float(lmin))
        self.pub_right.publish(float(rmin))

        # 扇区筛选 + 扇区可视化（保留）
        center = np.deg2rad(self.sector_center_deg)
        half   = np.deg2rad(self.sector_width_deg*0.5)
        self._publish_sector_outline(center, half)

        m = valid & (np.abs(wrap(angles - center)) <= half)
        if m.sum() < self.min_points:
            # 明确发布“没有墙/没有角点”，避免控制端看到“沉默帧”
            self.pub_has.publish(False)
            self.pub_corner_has.publish(False)
            return

        a = angles[m]; r = ranges[m]
        pts = np.stack([r*np.cos(a), r*np.sin(a)], axis=1)

        # ---------- RANSAC 主墙 ----------
        model1, in1_idx = self.fit_line_ransac(
            pts, self.ransac_iters, self.ransac_thresh, self.ransac_min_inliers
        )
        if model1 is None:
            self.pub_has.publish(False)
            self.pub_corner_has.publish(False)
            return
        c1, v1, n1 = model1

        # 侧向距离 d 与 yaw
        d = float(n1.dot(c1))
        if d < 0:
            n1 = -n1; v1 = -v1; d = -d
        wall_yaw = float(np.arctan2(v1[1], v1[0]))
        if v1[0] < 0:
            v1 = -v1
            wall_yaw = float(np.arctan2(v1[1], v1[0]))

        quality = int(in1_idx.size)  # 用内点数作“质量”指标，简单稳健

        # EMA
        if self.angle_s is None:
            self.angle_s, self.dist_s, self.quality_s = wall_yaw, d, quality
        else:
            self.angle_s = wrap(self.angle_s + self.ema_alpha_angle*wrap(wall_yaw - self.angle_s))
            self.dist_s  = (1-self.ema_alpha_dist)*self.dist_s + self.ema_alpha_dist*d
            self.quality_s = 0.8*self.quality_s + 0.2*quality

        # 发布主墙信息（bug2_v3/v4 都会用到）
        self.pub_has.publish(True)
        self.pub_angle.publish(self.angle_s)
        self.pub_lateral.publish(self.dist_s)
        self.pub_quality.publish(self.quality_s)

        # 主墙线（亮绿色）
        self._draw_line(c1, v1, color=(0.0,1.0,0.0,1.0), mid_id=1)

        # ---------- RANSAC 次墙（角点用） ----------
        # 用布尔掩码剔除主墙内点
        remain_mask = np.ones(len(pts), dtype=bool)
        remain_mask[in1_idx] = False
        remain = pts[remain_mask]

        # 可选：把“前探楔形扇区”的点并入次线候选，提升角点提前量
        if self.corner_front_enable:
            cdeg = self.corner_front_center_deg
            wdeg = self.corner_front_width_deg
            cfr = np.deg2rad(cdeg); hfr = np.deg2rad(wdeg*0.5)
            m_front = valid & (np.abs(wrap(angles - cfr)) <= hfr)
            if np.any(m_front):
                rf = ranges[m_front]; af = angles[m_front]
                pts_front = np.stack([rf*np.cos(af), rf*np.sin(af)], axis=1)
                if pts_front.size:
                    remain = np.vstack([remain, pts_front]) if remain.size else pts_front

        corner_found = False
        corner_xy = None

        if remain.shape[0] >= self.ransac_min_inliers2:
            model2, in2_idx = self.fit_line_ransac(
                remain, self.ransac_iters2, self.ransac_thresh2, self.ransac_min_inliers2
            )

            if model2 is not None:
                c2, v2, n2 = model2

                # 次线长度（span2）+ 内点占比，过滤噪点成线
                in2_pts = remain[in2_idx]
                c2tmp = in2_pts.mean(axis=0)
                X2tmp = in2_pts - c2tmp
                v2tmp = np.linalg.svd(X2tmp, full_matrices=False)[2][0]  # principal dir
                span2 = float((X2tmp @ v2tmp).ptp())
                inlier_ratio2 = in2_idx.size / float(remain.shape[0])
                if (span2 >= self.corner_min_span2) and (inlier_ratio2 >= self.corner_min_inlier_ratio2):
                    # 夹角
                    v1u = v1 / (np.linalg.norm(v1) + 1e-9)
                    v2u = v2 / (np.linalg.norm(v2) + 1e-9)
                    ang = np.degrees(np.arccos(np.clip(abs(np.dot(v1u, v2u)), -1, 1)))
                    if self.corner_ang_min_deg <= ang <= self.corner_ang_max_deg:
                        # 求交点：c1 + t v1 = c2 + s v2
                        A = np.array([v1, -v2]).T
                        b = c2 - c1
                        det = np.linalg.det(A)
                        if abs(det) > 1e-6:
                            t = np.linalg.solve(A, b)[0]
                            P = c1 + t*v1
                            rP = float(np.linalg.norm(P))
                            if (self.r_min <= rP <= self.r_max):
                                # 位置约束（右前/左前）
                                ok_pos = True
                                if self.corner_require_forward:
                                    if P[0] <= 0.0:
                                        ok_pos = False
                                    if self.side == "right" and (P[1] >  self.corner_y_tol):
                                        ok_pos = False
                                    if self.side == "left"  and (P[1] < -self.corner_y_tol):
                                        ok_pos = False
                                if ok_pos:
                                    corner_found = True
                                    corner_xy = P
                                    # 画次墙（淡绿色）
                                    self._draw_line(c2, v2, color=(0.0,0.8,0.0,1.0), mid_id=2)

        # ---------- 角点可视化 + 话题 ----------
        if corner_found and (corner_xy is not None) and np.isfinite(corner_xy).all():
            self._draw_corner(corner_xy)
            self.pub_corner_has.publish(True)

            # 在 base_link 下发布角点
            try:
                tf = self.tf_buf.lookup_transform("base_link", "laser", rospy.Time(0), rospy.Duration(0.05))
                tmp = PointStamped()
                tmp.header.frame_id = "laser"
                tmp.point.x, tmp.point.y, tmp.point.z = corner_xy[0], corner_xy[1], 0.0
                pt = do_transform_point(tmp, tf).point

                ps = PointStamped()
                ps.header.stamp = rospy.Time.now()
                ps.header.frame_id = "base_link"
                ps.point.x, ps.point.y, ps.point.z = pt.x, pt.y, pt.z
                self.pub_corner_pt.publish(ps)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
        else:
            self.pub_corner_has.publish(False)

    # ---------------- 前向障碍检测（保留） ----------------
    def _front_obstacle(self, ang, rng, valid):
        front = (np.abs(ang) <= np.deg2rad(20.0)) & valid
        return bool(np.nanmin(rng[front]) < self.front_stop) if np.any(front) else False

if __name__ == "__main__":
    rospy.init_node("perception", anonymous=False)
    WallPerception()
    rospy.spin()
