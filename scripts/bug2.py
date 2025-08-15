#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WallFollowerSafe（全向底盘版，内置CSV日志）
- 仅用 vx/vy 贴墙：默认 w=0（不转向）
- 可选：基于 LiDAR 的航向保持闭环（~yaw_hold=true 时生效，w 很小，仅纠偏）
- 绝不朝墙：所有 vy 会被夹断到“只允许远离墙”的方向
- 兼容你实测的 Y 轴方向（~y_positive_is_left 控制；你机器默认 False）
- 自动写 CSV：~/.ros/wall_follower_logs/wf_*.csv

记录字段：
time, state, has_wall, obs_front, wall_dist, wall_yaw,
clearance, e_d, vx_des, vy_des, w_des, cmd_vx, cmd_vy, cmd_w
"""
import os, csv, math, time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def wrap(a): return (a + math.pi) % (2.0*math.pi) - math.pi

class WallFollowerSafe:
    def __init__(self):
        # ====== 侧别与坐标 ======
        self.side = rospy.get_param("~side", "right")    # "right" / "left"
        # 你的机体：实测 +Y 朝右，所以这里默认 False（与 REP-103 相反）
        self.y_positive_is_left = rospy.get_param("~y_positive_is_left", False)

        # ====== 距离/速度参数 ======
        self.d_ref     = rospy.get_param("~d_ref", 0.50)     # 目标【最右侧】与墙的距离（建议 0.45~0.50）
        self.band_d    = rospy.get_param("~band_d", 0.05)    # 距离死区
        self.k_dist    = rospy.get_param("~k_dist", 0.60)    # 距离误差 -> vy 比例
        self.right_overhang = rospy.get_param("~right_overhang", 0.10)  # LiDAR 到最右边缘外伸（请尺量）
        self.d_offset  = rospy.get_param("~d_offset", 0.0)

        self.vx_nom    = rospy.get_param("~vx_nom", 0.09)    # 沿墙巡航速度
        self.vx_min    = rospy.get_param("~vx_min", 0.04)
        self.vx_max    = rospy.get_param("~vx_max", 0.30)
        self.vy_max    = rospy.get_param("~vy_max", 0.30)    # 远离墙最大侧移
        self.dv_max    = rospy.get_param("~dv_max", 0.18)    # m/s^2（x/y）限加速度
        self.dw_max    = rospy.get_param("~dw_max", 1.0)     # rad/s^2 角加速度限幅
        self.rate      = rospy.get_param("~rate", 20.0)      # 控制频率 Hz

        # ====== 航向保持（可选）======
        self.yaw_hold  = rospy.get_param("~yaw_hold", True) # True 开启小 w 纠偏
        self.k_yaw     = rospy.get_param("~k_yaw", 1.2)
        self.yaw_deadband = rospy.get_param("~yaw_deadband", 0.05)  # ≈3°
        self.w_max     = rospy.get_param("~w_max", 0.20)     # 纠偏上限（很小）

        # ====== 安全护栏 ======
        self.brake_margin     = rospy.get_param("~brake_margin", 0.08)   # 近于目标-8cm 降速+外滑
        self.emergency_margin = rospy.get_param("~emergency_margin", 0.16)# 近于目标-16cm 触发急停外滑
        self.emergency_time   = rospy.get_param("~emergency_time", 0.45)  # s
        self.start_out_bias   = rospy.get_param("~start_out_bias", 0.03)  # 起步向外偏置
        self.timeout          = rospy.get_param("~timeout", 0.8)          # 感知超时

        # ====== CSV 日志 ======
        self.log_csv  = rospy.get_param("~log_csv", True)
        self.log_hz   = float(rospy.get_param("~log_hz", 10.0))
        self.log_dir  = os.path.expanduser(rospy.get_param("~log_dir", "~/.ros/wall_follower_logs"))
        self._csv = None
        self._csv_writer = None
        self._last_log_t = 0.0

        # ====== 状态量 ======
        self.state       = "START"
        self.has_wall    = False
        self.obs_front   = False
        self.wall_yaw    = 0.0
        self.wall_dist   = None
        self.t_last      = 0.0
        self.t_state     = time.time()
        self.t_emg_until = 0.0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_w  = 0.0

        # ====== 订阅/发布 ======
        rospy.Subscriber("/wall_angle",     Float32, self.cb_ang,  queue_size=1)
        rospy.Subscriber("/wall_lateral",   Float32, self.cb_dist, queue_size=1)
        rospy.Subscriber("/wall_quality",   Float32, self.cb_q,    queue_size=1)
        rospy.Subscriber("/wall_has",       Bool,    self.cb_has,  queue_size=1)
        rospy.Subscriber("/obstacle_ahead", Bool,    self.cb_obs,  queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # 打印启动信息
        rospy.loginfo("[bug2_safe] side=%s y+is_left=%s d_ref=%.2f band=%.2f vx_nom=%.2f vy_max=%.2f w_max=%.2f overhang=%.2f",
                      self.side, self.y_positive_is_left, self.d_ref, self.band_d,
                      self.vx_nom, self.vy_max, (self.w_max if self.yaw_hold else 0.0), self.right_overhang)

        # CSV 初始化
        if self.log_csv:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.log_dir, f"wf_{ts}.csv")
                self._csv = open(path, "w", newline="")
                self._csv_writer = csv.writer(self._csv)
                self._csv_writer.writerow([
                    "time","state","has_wall","obs_front","wall_dist","wall_yaw",
                    "clearance","e_d","vx_des","vy_des","w_des","cmd_vx","cmd_vy","cmd_w"
                ])
                rospy.loginfo("[bug2_safe] CSV log -> %s (%.1f Hz)", path, self.log_hz)
            except Exception as e:
                rospy.logwarn("CSV init failed: %s", e)
                self.log_csv = False
        rospy.on_shutdown(self._close_csv)

        # 主循环
        r = rospy.Rate(self.rate)
        dt = 1.0 / max(self.rate, 1.0)
        while not rospy.is_shutdown():
            self.step(dt)
            r.sleep()

    # ===== 回调 =====
    def cb_ang(self,  m): self.wall_yaw  = float(m.data); self.t_last=time.time()
    def cb_dist(self, m): self.wall_dist = float(m.data); self.t_last=time.time()
    def cb_q(self,   m): pass
    def cb_has(self, m): self.has_wall   = bool(m.data)
    def cb_obs(self, m): self.obs_front  = bool(m.data)

    def fresh(self): return (time.time()-self.t_last) < self.timeout if self.t_last>0 else False
    @staticmethod
    def deadband(e,b): return 0.0 if abs(e)<=b else (abs(e)-b)*(1 if e>0 else -1)
    def set_state(self,s):
        if s!=self.state:
            self.state=s; self.t_state=time.time()
            rospy.loginfo("[bug2_safe] -> %s", s)

    def step(self, dt):
        right = (self.side=="right")

        # 依据你的机体：+Y 是否为左。左的指令符号
        left_sign  = +1.0 if self.y_positive_is_left else -1.0
        right_sign = -left_sign

        # 远离墙方向对应的 vy 指令符号（右墙=向左，左墙=向右）
        away_sign   = left_sign if right else right_sign
        toward_sign = -away_sign

        fresh = self.fresh()
        vx_des, vy_des, w_des = 0.0, 0.0, 0.0

        if self.state=="START":
            # 起步：慢速前进 + 向外轻偏
            vx_des = self.vx_min
            vy_des = away_sign * self.start_out_bias
            if time.time() - self.t_state > 2.0:
                self.set_state("TRACK")

        elif self.state=="TRACK":
            ok = self.has_wall and fresh and (self.wall_dist is not None)

            d_target = self.d_ref + self.d_offset
            if not ok:
                # 找不到墙：慢速前进 + 向外小偏，绝不朝墙
                vx_des = self.vx_min
                vy_des = away_sign * 0.03
            else:
                # 真实间隙 = 激光测距 - 外伸量（右墙）
                clearance = self.wall_dist - (self.right_overhang if right else 0.0)
                e_d   = (d_target - clearance)            # >0: 太近，需要远离墙
                vx_des = -0.05 if phase > 0.5 else 0.0
                vy_des = 0.0
                w_des  = 0.2
                """e_d_c = self.deadband(e_d, self.band_d)
                vy_des = self.k_dist * e_d_c * away_sign  # 远离墙为正方向
                vx_des = self.vx_nom

                # 非对称限幅（只限制“朝墙”的分量为 0）
                vy_des = clamp(vy_des, -self.vy_max, self.vy_max)
                if vy_des * away_sign < 0.0:   # 任何朝墙尝试都被夹断
                    vy_des = 0.0

                # 护栏：太近（以真实间隙判断）
                margin_close = (d_target - clearance)     # >0 表示太近
                rospy.loginfo("margin_close=%.2f", margin_close)
                if margin_close > self.emergency_margin:
                    self.t_emg_until = time.time() + self.emergency_time
                elif margin_close > self.brake_margin:
                    rospy.loginfo("Magnus")
                    vx_des = min(vx_des, self.vx_min)
                    vy_des = clamp(vy_des + 0.12*away_sign, -self.vy_max, self.vy_max)

                # 前方障碍：只降速（不允许“朝墙贴回去”）
                if self.obs_front:
                    vx_des = min(vx_des, self.vx_min)

                # 航向保持（可选，极小 w，只做纠偏）
                if self.yaw_hold:
                    yaw_err = wrap(self.wall_yaw)          # 目标=0（与墙平行）
                    w_raw  = self.k_yaw * self.deadband(yaw_err, self.yaw_deadband)
                    w_des  = clamp(w_raw, -self.w_max, self.w_max)"""
            # 急停外滑窗口：先轻退，再定住 + 最大外滑
            if time.time() < self.t_emg_until:
                phase = (self.t_emg_until - time.time()) / max(self.emergency_time, 1e-3)
                vx_des = -0.05 if phase > 0.5 else 0.0
                vy_des = away_sign * self.vy_max
                w_des  = 0.0

            vx_des = 0.01   
            vy_des = 0.0
            w_des  = -0.1
            rospy.loginfo("[bug2_safe] -> %s", self.wall_dist)

        else:
            self.set_state("START")

        # 限加速度并发布
        self.cmd_vx = clamp(vx_des, self.cmd_vx - self.dv_max*dt, self.cmd_vx + self.dv_max*dt)
        self.cmd_vy = clamp(vy_des, self.cmd_vy - self.dv_max*dt, self.cmd_vy + self.dv_max*dt)
        # 如果没开航向保持，w 就始终被拉到 0
        w_des = 0.0 if not self.yaw_hold else w_des
        self.cmd_w  = clamp(w_des, self.cmd_w - self.dw_max*dt, self.cmd_w + self.dw_max*dt)

        tw = Twist()
        tw.linear.x  = self.cmd_vx
        tw.linear.y  = self.cmd_vy
        tw.angular.z = self.cmd_w
        self.pub.publish(tw)

        # ===== CSV 日志 =====
        if self.log_csv and self._csv_writer:
            t_now = time.time()
            if (t_now - self._last_log_t) >= (1.0/max(self.log_hz, 1e-3)):
                # 计算当前 clearance / e_d（若可用）
                clearance_log = None
                e_d_log = None
                if self.wall_dist is not None:
                    clearance_log = self.wall_dist - (self.right_overhang if self.side=="right" else 0.0)
                    e_d_log = (self.d_ref + self.d_offset) - clearance_log
                try:
                    self._csv_writer.writerow([
                        f"{t_now:.3f}", self.state, int(self.has_wall), int(self.obs_front),
                        ("%.4f"%self.wall_dist) if self.wall_dist is not None else "",
                        ("%.4f"%self.wall_yaw) if self.wall_dist is not None else "",
                        ("%.4f"%clearance_log) if clearance_log is not None else "",
                        ("%.4f"%e_d_log) if e_d_log is not None else "",
                        ("%.3f"%vx_des), ("%.3f"%vy_des), ("%.3f"%w_des),
                        ("%.3f"%self.cmd_vx), ("%.3f"%self.cmd_vy), ("%.3f"%self.cmd_w)
                    ])
                except Exception as e:
                    rospy.logwarn_throttle(5.0, "CSV write fail: %s", e)
                self._last_log_t = t_now

    def _close_csv(self):
        try:
            if self._csv:
                self._csv.flush(); self._csv.close()
        except Exception:
            pass

def main():
    rospy.init_node("wall_follower", anonymous=False)
    WallFollowerSafe()

if __name__=="__main__":
    main()
