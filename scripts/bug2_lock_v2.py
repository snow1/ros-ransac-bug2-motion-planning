#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, csv, time
from geometry_msgs.msg import Twist, Point32
from std_msgs.msg import Header
from math import isnan
from std_msgs.msg import Float32

class LockerV2:
    def __init__(self):
        # === 行为参数（与之前一致） ===
        self.target_dist = rospy.get_param("~target_dist", 0.50)
        self.align_tol_d = rospy.get_param("~align_tol_d", 0.03)
        self.align_tol_y = rospy.get_param("~align_tol_y", 0.03)

        self.forward_speed = rospy.get_param("~forward_speed", 0.12)
        self.forward_speed_hold = rospy.get_param("~forward_speed_hold", 0.08)
        self.min_forward_speed = rospy.get_param("~min_forward_speed", 0.06)

        self.k_vy = rospy.get_param("~k_vy", 1.2)
        self.k_w  = rospy.get_param("~k_w", 1.5)
        self.max_vy = rospy.get_param("~max_vy", 0.25)
        self.max_w  = rospy.get_param("~max_w", 0.8)

        self.front_stop = rospy.get_param("~front_stop", 0.40)
        self.loss_timeout_stop = rospy.get_param("~loss_timeout_stop", 2.5)
        self.loss_grace = rospy.get_param("~loss_grace", 0.8)
        self.reacq_tol_d = rospy.get_param("~reacq_tol_d", 0.08)
        self.reacq_tol_y = rospy.get_param("~reacq_tol_y", 0.10)
        self.loss_slowdown_window = rospy.get_param("~loss_slowdown_window", 10.0)
        self.loss_slowdown_thresh = rospy.get_param("~loss_slowdown_thresh", 3)

        # === 与 Arduino 对齐的速度限幅（板端 VX_MAX/VY_MAX/WZ_MAX 对齐） ===
        self.limit_vx = rospy.get_param("~limit_vx", 0.20)   # m/s
        self.limit_vy = rospy.get_param("~limit_vy", 0.20)   # m/s（仅 ALIGN 用）
        self.limit_w  = rospy.get_param("~limit_w", 0.60)    # rad/s

        # === 软启动 / 软收 ===
        self.soft_gate = 0.0
        self.soft_up   = rospy.get_param("~soft_up",   0.04)  # 每周期上升幅度（20Hz下≈每秒+0.8上限门）
        self.soft_down = rospy.get_param("~soft_down", 0.06)  # 每周期下降
        self.soft_enable = rospy.get_param("~soft_enable", True)

        # === 话题与日志 ===
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.log_path = rospy.get_param("~log_path", "/tmp/wf_lock_v2_{:d}.csv".format(int(time.time())))

        # === 状态 ===
        self.state = "ALIGN"    # ALIGN → LOCKED_GO → HOLD → (LOCKED_GO / FAIL_STOP)
        self.last_perc_t = None
        self.has_wall = 0
        self.obs_front = 0
        self.quality = "nan"
        self.wall_dist = float('nan')
        self.wall_yaw  = float('nan')
        self.clearance = float('nan')

        # 记忆（最近一次可靠墙）
        self.mem_dist = float('nan')
        self.mem_yaw  = float('nan')
        self.mem_time = None

        # 统计
        self.loss_events = []
        self.reason = ""

        # ROS I/O
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        rospy.Subscriber("wall/perception", Point32, self.cb_perc, queue_size=1)
        rospy.Subscriber("wall/flags", Header, self.cb_flags, queue_size=1)
        #rospy.Subscriber("/wall_lateral",   Float32, self.cb_dist, queue_size=1)

        # CSV（更详细）
        self.csvf = open(self.log_path, "w", newline="")
        self.csv = csv.writer(self.csvf)
        self.csv.writerow([
            "time","state","reason",
            "has_wall","obs_front","quality",
            "wall_dist","wall_yaw","clearance",
            "mem_dist","mem_yaw","mem_age",
            "vx_cmd","vy_cmd","w_cmd",
            "vx_limited","vy_limited","w_limited",
            "soft_gate"
        ])
        rospy.loginfo("bug2_lock_v2 started. log -> %s", self.log_path)

    # === 订阅回调 ===
    def cb_perc(self, pt: Point32):
        self.wall_dist = pt.x
        self.wall_yaw  = pt.y
        self.clearance = pt.z
        self.last_perc_t = rospy.Time.now().to_sec()

    def cb_flags(self, h: Header):
        self.has_wall = int(h.seq)
        self.obs_front = int(h.stamp.secs)
        self.quality = h.frame_id  # "inliers/total|rms"

    # === 统一发布（带限幅与软启动/软收） ===
    def publish(self, vx, vy, wz, reason=""):
        # 限幅（与 Arduino 的 VX_MAX/VY_MAX/WZ_MAX 对齐）
        vx_l = max(min(vx, self.limit_vx), -self.limit_vx)
        vy_l = max(min(vy, self.limit_vy), -self.limit_vy)
        wz_l = max(min(wz, self.limit_w ), -self.limit_w )

        # 软启动门（LOCKED/HOLD/ALIGN 都适用）
        target_mag = max(abs(vx_l), abs(vy_l), abs(wz_l))
        if self.soft_enable:
            if target_mag > 1e-6:
                self.soft_gate = min(1.0, self.soft_gate + self.soft_up)
            else:
                self.soft_gate = max(0.0, self.soft_gate - self.soft_down)
        else:
            self.soft_gate = 1.0 if target_mag > 1e-6 else 0.0

        # 应用软门
        vx_out = vx_l * self.soft_gate
        vy_out = vy_l * self.soft_gate
        wz_out = wz_l * self.soft_gate

        # 发布 cmd_vel
        cmd = Twist()
        cmd.linear.x  = vx_out
        cmd.linear.y  = vy_out
        cmd.angular.z = wz_out
        self.pub_cmd.publish(cmd)

        # 记录 CSV
        now = rospy.Time.now().to_sec()
        mem_age = (now - self.mem_time) if self.mem_time else float('nan')
        self.csv.writerow([
            now, self.state, reason,
            self.has_wall, self.obs_front, self.quality,
            self.wall_dist, self.wall_yaw, self.clearance,
            self.mem_dist, self.mem_yaw, mem_age,
            vx, vy, wz,
            vx_l, vy_l, wz_l,
            self.soft_gate
        ])

    def stop(self, reason="STOP"):
        self.publish(0.0, 0.0, 0.0, reason)

    def losses_in_window(self, now):
        self.loss_events = [t for t in self.loss_events if now - t <= self.loss_slowdown_window]
        return len(self.loss_events)

    # === 主循环 ===
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            vx = vy = wz = 0.0
            reason = ""

            # 安全：前障即停
            if self.obs_front == 1 or (isinstance(self.clearance, float) and self.clearance < self.front_stop):
                self.state = "FAIL_STOP"
                self.stop("OBS_FRONT")
                rate.sleep()
                continue

            # 感知超时：按丢失处理
            perc_age = (now - self.last_perc_t) if self.last_perc_t else 999

            # 状态机
            if self.state == "ALIGN":
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    # 记忆
                    self.mem_dist = self.wall_dist
                    self.mem_yaw  = self.wall_yaw
                    self.mem_time = now

                    # 仅对齐阶段允许 vy & w，x 给微小正向帮助贴墙
                    e_d = self.target_dist - self.wall_dist
                    vy = max(min(self.k_vy * e_d, self.max_vy), -self.max_vy)
                    wz = max(min(self.k_w  * (0.0 - self.wall_yaw), self.max_w), -self.max_w)
                    vx = 0.05
                    reason = "ALIGNING"

                    if abs(e_d) <= self.align_tol_d and abs(self.wall_yaw) <= self.align_tol_y:
                        self.state = "LOCKED_GO"
                        self.stop("ALIGN_DONE")  # 切换瞬间清零（软门也会自动下降）
                        rate.sleep()
                        continue
                else:
                    # 没墙 → HOLD（宽限）
                    self.state = "HOLD"
                    self.loss_events.append(now)
                    vx = self.forward_speed_hold
                    reason = "ALIGN_LOSS→HOLD"

            elif self.state == "LOCKED_GO":
                # 直行：vx 固定；vy=0, wz=0
                vx = self.forward_speed
               # wz = 1.0
                reason = "LOCKED_GO"

                # 维护记忆
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    self.mem_dist = self.wall_dist
                    self.mem_yaw  = self.wall_yaw
                    self.mem_time = now

                # 丢失 → HOLD
                if (not self.has_wall) or perc_age > self.loss_grace:
                    self.state = "HOLD"
                    self.loss_events.append(now)
                    self.stop("LOCKED→HOLD")  # 清零一次，让软门重新按降速直行起来
                    rate.sleep()
                    continue

            elif self.state == "HOLD":
                # 不横移不转向，降速直行
                vx = self.forward_speed_hold
                reason = "HOLD_KEEP_STRAIGHT"

                # 超时则停
                if perc_age > self.loss_timeout_stop:
                    self.state = "FAIL_STOP"
                    self.stop("HOLD_TIMEOUT")
                    rate.sleep()
                    continue

                # 再捕获：与记忆差异不大则回 LOCKED_GO
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw) and self.mem_time:
                    if abs(self.wall_dist - self.mem_dist) <= self.reacq_tol_d and abs(self.wall_yaw - self.mem_yaw) <= self.reacq_tol_y:
                        self.state = "LOCKED_GO"
                        self.stop("REACQ→LOCKED")
                        rate.sleep()
                        continue
                    else:
                        # 更新记忆（质量可能更好）
                        self.mem_dist = self.wall_dist
                        self.mem_yaw  = self.wall_yaw
                        self.mem_time = now
                        reason = "HOLD_MEM_UPDATE"

                # 频繁丢失 → 自适应降速
                nloss = self.losses_in_window(now)
                if nloss >= self.loss_slowdown_thresh and self.forward_speed > self.min_forward_speed:
                    self.forward_speed = max(self.min_forward_speed, self.forward_speed - 0.02)
                    self.forward_speed_hold = max(self.min_forward_speed, self.forward_speed_hold - 0.02)
                    reason = f"HOLD_SLOWDOWN(n={nloss})"

            elif self.state == "FAIL_STOP":
                self.stop("STOP_STATE")
                rate.sleep()
                continue

            # 发布（带限幅与软门）
            print(f"State: {self.state}, Reason: {reason}, Wall Dist: {self.wall_dist:.2f}, Wall Yaw: {self.wall_yaw:.2f}, Clearance: {self.clearance:.2f}")
            print(f"vx: {vx:.2f}, vy: {vy:.2f}, wz: {wz:.2f}, Soft Gate: {self.soft_gate:.2f}")
            self.publish(vx, vy, wz, reason)
            rate.sleep()

    def __del__(self):
        try:
            self.csvf.close()
        except:
            pass

if __name__ == "__main__":
    rospy.init_node("bug2_lock_v2")
    LockerV2().run()
