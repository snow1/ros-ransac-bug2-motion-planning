#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point

def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

class WallPerception:
    def __init__(self):
        # ---- 参数 ----
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.side = rospy.get_param("~side", "right")  # right/left
        self.sector_center_deg = rospy.get_param("~sector_center_deg", -90.0 if self.side=="right" else 90.0)
        self.sector_width_deg  = rospy.get_param("~sector_width_deg", 85.0)
        self.r_min = rospy.get_param("~r_min", 0.12)
        self.r_max = rospy.get_param("~r_max", 8.0)
        self.min_points = rospy.get_param("~min_points", 30)
        self.min_span   = rospy.get_param("~min_span", 0.25)
        self.quality_min= rospy.get_param("~quality_min", 5.0)
        self.min_quality_keep = rospy.get_param("~min_quality_keep", 3.0)
        self.front_stop = rospy.get_param("~front_stop", 0.55)

        self.ema_alpha_angle = rospy.get_param("~ema_alpha_angle", 0.3)
        self.ema_alpha_dist  = rospy.get_param("~ema_alpha_dist", 0.3)

        self.marker_frame = rospy.get_param("~marker_frame", "laser")  # 或 "map"/"base_link"
        self.tf_buf = tf2_ros.Buffer(); self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        # ---- 发布者 ----
        self.pub_angle   = rospy.Publisher("/wall_angle",   Float32, queue_size=10)
        self.pub_lateral = rospy.Publisher("/wall_lateral", Float32, queue_size=10)
        self.pub_quality = rospy.Publisher("/wall_quality", Float32, queue_size=10)
        self.pub_has     = rospy.Publisher("/wall_has",     Bool,    queue_size=10)
        self.pub_obs     = rospy.Publisher("/obstacle_ahead", Bool,  queue_size=10)

        self.pub_marker  = rospy.Publisher("visualization_marker", Marker, queue_size=1)
        self.mk = Marker(type=Marker.LINE_LIST, action=Marker.ADD)
        self.mk.scale.x = 0.02; self.mk.color.g = 1.0; self.mk.color.a = 1.0

        self.angle_s = None; self.dist_s = None; self.quality_s = 0.0
        rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan, queue_size=1)

    def cb_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > self.r_min) & (ranges < self.r_max)

        # 前方障碍
        self.pub_obs.publish(self._front_obstacle(angles, ranges, valid))

        # 右/左侧扇区
        center = np.deg2rad(self.sector_center_deg); half = np.deg2rad(self.sector_width_deg*0.5)
        m = valid & (np.abs(wrap(angles - center)) <= half)
        if m.sum() < self.min_points:
            if self.quality_s < self.min_quality_keep: self.pub_has.publish(False)
            return

        r = ranges[m]; a = angles[m]
        x = r*np.cos(a); y = r*np.sin(a)
        pts = np.stack([x,y], axis=1)

        # PCA 拟合
        c = pts.mean(axis=0); X = pts - c
        cov = (X.T @ X) / max(len(X)-1, 1)
        w, V = np.linalg.eigh(cov)
        v = V[:,1]  # 切线
        if v[0] < 0: v = -v
        nrm = np.array([-v[1], v[0]]); nrm /= (np.linalg.norm(nrm)+1e-9)

        d = float(nrm.dot(c))        # 原点到直线的有向距离
        if d < 0: nrm=-nrm; v=-v; d=-d

        eig_ratio = float(w[1]/max(w[0],1e-9))
        proj = X @ v; span = float(proj.max()-proj.min())
        quality = eig_ratio * span
        good = (eig_ratio >= self.quality_min) and (span >= self.min_span)
        if not good and self.quality_s < self.min_quality_keep:
            self.pub_has.publish(False); return

        wall_yaw = float(np.arctan2(v[1], v[0]))
        # 保障切向指向“前方”，再收敛到 ±90°
        if v[0] < 0: v = -v; wall_yaw = float(np.arctan2(v[1], v[0]))

        # EMA 平滑
        if self.angle_s is None:
            self.angle_s, self.dist_s, self.quality_s = wall_yaw, d, quality
        else:
            self.angle_s = wrap(self.angle_s + self.ema_alpha_angle*wrap(wall_yaw - self.angle_s))
            self.dist_s  = (1-self.ema_alpha_dist)*self.dist_s + self.ema_alpha_dist*d
            self.quality_s = 0.8*self.quality_s + 0.2*quality

        self.pub_has.publish(True)
        self.pub_angle.publish(self.angle_s)
        self.pub_lateral.publish(self.dist_s)
        self.pub_quality.publish(self.quality_s)

        # RViz 绿色线
        L = 0.8*span if span>0 else 1.0
        p1 = c - v*(L*0.5); p2 = c + v*(L*0.5)
        try:
            if self.marker_frame == "laser":
                self.mk.header.frame_id = "laser"
                self.mk.header.stamp = rospy.Time.now()
                self.mk.points = [Point(p1[0],p1[1],0.0), Point(p2[0],p2[1],0.0)]
            else:
                trans = self.tf_buf.lookup_transform(self.marker_frame, "laser", rospy.Time(0), rospy.Duration(0.05))
                p1s = PointStamped(); p1s.header.frame_id="laser"; p1s.point.x,p1s.point.y= p1[0],p1[1]
                p2s = PointStamped(); p2s.header.frame_id="laser"; p2s.point.x,p2s.point.y= p2[0],p2[1]
                p1t = do_transform_point(p1s, trans).point; p2t = do_transform_point(p2s, trans).point
                self.mk.header.frame_id = self.marker_frame
                self.mk.header.stamp = rospy.Time.now()
                self.mk.points = [p1t, p2t]
            self.pub_marker.publish(self.mk)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            pass

    def _front_obstacle(self, ang, rng, valid):
        front = (np.abs(ang) <= np.deg2rad(20.0)) & valid
        return bool(np.nanmin(rng[front]) < self.front_stop) if np.any(front) else False

if __name__ == "__main__":
    rospy.init_node("perception", anonymous=False)
    WallPerception(); rospy.spin()
