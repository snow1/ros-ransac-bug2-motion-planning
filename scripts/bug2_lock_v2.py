#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy, csv, time
from geometry_msgs.msg import Twist, Point32, PointStamped
from std_msgs.msg import Header, Bool,Float32
from math import isnan, hypot, pi
class LockerV2:
    def __init__(self):
        # === 行为参数 ===
        # If your base moves the opposite way on Y, set this to -1.0 in the launch.
        self.cmd_vy_sign = rospy.get_param("~cmd_vy_sign", -1.0)  # try -1.0 given your symptom

        self.target_dist = rospy.get_param("~target_dist", 0.60)
        self.align_tol_d = rospy.get_param("~align_tol_d", 0.03)
        self.align_tol_y = rospy.get_param("~align_tol_y", 0.03)

        self.forward_speed = rospy.get_param("~forward_speed", 0.12)
        self.forward_speed_hold = rospy.get_param("~forward_speed_hold", 0.08)
        self.min_forward_speed = rospy.get_param("~min_forward_speed", 0.06)

        # 对齐阶段增益
        self.k_vy = rospy.get_param("~k_vy", 1.2)
        self.k_w  = rospy.get_param("~k_w", 1.5)
        self.max_vy = rospy.get_param("~max_vy", 0.25)
        self.max_w  = rospy.get_param("~max_w", 0.5)

        # LOCKED_GO 阶段的小幅纠偏（麦轮）
        self.k_vy_go = rospy.get_param("~k_vy_go", 0.8)
        self.k_w_go  = rospy.get_param("~k_w_go", 0.8)
        self.max_vy_go = rospy.get_param("~max_vy_go", 0.20)
        self.max_w_go  = rospy.get_param("~max_w_go", 0.35)
        self.deadzone_d   = rospy.get_param("~deadzone_d", 0.01)
        self.deadzone_yaw = rospy.get_param("~deadzone_yaw", 0.01)

        # 安全/丢失
        self.front_stop = rospy.get_param("~front_stop", 0.40)
        self.loss_timeout_stop = rospy.get_param("~loss_timeout_stop", 2.5)
        self.loss_grace = rospy.get_param("~loss_grace", 0.8)
        self.reacq_tol_d = rospy.get_param("~reacq_tol_d", 0.08)
        self.reacq_tol_y = rospy.get_param("~reacq_tol_y", 0.10)
        self.loss_slowdown_window = rospy.get_param("~loss_slowdown_window", 10.0)
        self.loss_slowdown_thresh = rospy.get_param("~loss_slowdown_thresh", 3)

        # 与底盘限幅
        self.limit_vx = rospy.get_param("~limit_vx", 0.20)
        self.limit_vy = rospy.get_param("~limit_vy", 0.20)
        self.limit_w  = rospy.get_param("~limit_w", 0.60)

        # 软启动/收
        self.soft_gate = 0.0
        self.soft_up   = rospy.get_param("~soft_up",   0.04)
        self.soft_down = rospy.get_param("~soft_down", 0.06)
        self.soft_enable = rospy.get_param("~soft_enable", True)

                # __init__ 里加：
        self.turn_back_enable   = rospy.get_param("~turn_back_enable", True)  # 开关
        self.turn_back_vx       = rospy.get_param("~turn_back_vx", -0.05)     # 轻微后退 m/s
        self.turn_back_duration = rospy.get_param("~turn_back_duration", 0.8) # 仅前0.8s
        self.turn_side_push_vy  = rospy.get_param("~turn_side_push_vy", 0.0)  # 默认不侧摆；如需更安全可给 +0.04(右墙向左)


                # ===== 在 __init__ 里新增这些参数（保持原有参数不删） =====
        self.simple_corner_turn_dist = rospy.get_param("~simple_corner_turn_dist", 1.20)  # 触发距离
        self.simple_clear_extra      = rospy.get_param("~simple_clear_extra",     0.15)  # 额外侧向净空
        self.simple_side_speed       = rospy.get_param("~simple_side_speed",      0.18)  # 让位侧移速度上限
        self.simple_side_timeout     = rospy.get_param("~simple_side_timeout",    1.2)   # 让位最长时长(s)
        self.simple_turn_speed       = rospy.get_param("~simple_turn_speed",      0.6)   # 原地转角速度(rad/s)
        self.simple_yaw_tol_deg      = rospy.get_param("~simple_yaw_tol_deg",     5.0)   # 角度容差
        self.simple_yaw_hold_frames  = rospy.get_param("~simple_yaw_hold_frames", 3)     # 连续满足帧数
        self.simple_turn_timeout     = rospy.get_param("~simple_turn_timeout",    5.0)   # 转弯兜底时长(s)
        self.simple_forward_only     = rospy.get_param("~simple_forward_only",    True)  # 只接受前方角点

        # 角点触发（新增）
        self.side = rospy.get_param("~side", "right")  # right/left，用于决定转向
        self.corner_trigger_dist = rospy.get_param("~corner_trigger_dist", 1.20)   # 触发
        self.corner_reset_dist   = rospy.get_param("~corner_reset_dist",   1.30)   # 滞回
        self.corner_turn_speed   = rospy.get_param("~corner_turn_speed",   0.6)    # rad/s
        self.corner_turn_angle   = rospy.get_param("~corner_turn_angle",   pi/2)   # 90°
        self.corner_cooldown     = rospy.get_param("~corner_cooldown",     4.0)    # s
        self.corner_fresh_t      = rospy.get_param("~corner_fresh_t",      0.8)    # 角点消息新鲜度 s
        self.corner_forward_only = rospy.get_param("~corner_forward_only", True)   # 仅前方角点触发

        self.corner_prepare_t = rospy.get_param("~corner_prepare_t", 0.30)  # 预停时长(s)
        self.corner_brake_times = rospy.get_param("~corner_brake_times", 3) # 进入转弯前，多发几帧0速度
        self.corner_state = "IDLE"  # IDLE / APPROACH_STOP / TURNING / COOLDOWN
        self.corner_prepare_end = 0.0

        # --- 角点触发的“真·边沿/武装”参数（用于防止静止反复触发） ---
        self.corner_arm_samples = rospy.get_param("~corner_arm_samples", 3)     # 连续 N 帧 > reset 才武装
        self.corner_trend_eps   = rospy.get_param("~corner_trend_eps", 0.015)   # 下降趋势阈值(米/帧)
        self.corner_min_vx_to_trigger = rospy.get_param("~corner_min_vx_to_trigger", 0.05)  # 仅在前进触发

                # === 参数：安全侧距（可在 launch 里覆盖），以及外摆速度与超时 ===
        # 建议默认 0.85 m；如果雷达不在车体中心，需要 +/− 安装偏置
        self.safe_clearance = rospy.get_param("~safe_clearance", 0.85)  # m
        self.clear_speed    = rospy.get_param("~clear_speed",    0.18)  # m/s，侧向外摆速度上限
        self.clear_timeout  = rospy.get_param("~clear_timeout",  2.0)   # s，外摆最长时长

        # >>> PATCH: 允许在 HOLD 也能触发角点（默认加入 HOLD，仍可通过参数覆盖）
        _default_trig_states = ["LOCKED_GO","HOLD"]
        self.corner_trigger_in_states = rospy.get_param("~corner_trigger_in_states", _default_trig_states)
        # <<< PATCH

        # --- 运行时辅助状态（供 corner_fsm 使用） ---
        self.corner_armed = False
        self.corner_arm_cnt = 0
        self.corner_prev_dist = None
        self.current_vx = 0.0  # 在 publish() 内更新

        # --- 角度闭环参数 ---
        self.turn_yaw_tol_deg   = rospy.get_param("~turn_yaw_tol_deg", 6.0)   # 墙朝向容差 ±6°
        self.turn_hold_frames   = rospy.get_param("~turn_hold_frames", 3)     # 连续K帧满足才算到位
        self.turn_exit_extra_m  = rospy.get_param("~turn_exit_extra_m", 0.30) # 角点距离退出冗余
        self.turn_max_time      = rospy.get_param("~turn_max_time",  5.0)     # 兜底最长转弯时间

        # 进入TURNING时记录
        self.turn_yaw_start = None
        self.turn_yaw_target = None
        self.turn_hold_cnt = 0
        # 用于取墙朝向的“稳态值”
        self._yaw_buf = []
        self._yaw_buf_len = rospy.get_param("~turn_yaw_buf_len", 5)

        # 话题与日志
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.log_path = rospy.get_param("~log_path", "/tmp/wf_lock_v2_{:d}.csv".format(int(time.time())))

        # === 状态 ===
        self.state = "ALIGN"    # ALIGN → LOCKED_GO → HOLD → (LOCKED_GO / FAIL_STOP)
        self.last_perc_t = None
        self.has_wall = 0
        self.obs_front = 0
        self.quality = "nan"
        self.wall_dist = float('nan')   # 与墙的横向距离
        self.wall_yaw  = float('nan')   # 与墙夹角(rad)
        self.clearance = float('nan')

        # 记忆
        self.mem_dist = float('nan')
        self.mem_yaw  = float('nan')
        self.mem_time = None

        # 丢失统计
        self.loss_events = []

        # === 角点子状态机（新增） ===
        self.corner_has = False
        self.corner_pt  = None          # (x,y) in base_link（建议）
        self.corner_dist = float('nan')
        self.corner_last_over = True    # 上一帧是否在“重置距离之外”
        self.corner_state = "IDLE"      # IDLE / TURNING / COOLDOWN
        self.corner_turn_end_time = 0.0
        self.corner_cool_end_time = 0.0
        self.corner_last_stamp = None   # 角点消息时间戳

        # ROS I/O
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        rospy.Subscriber("/wall_lateral",     Float32,     self.cb_wall_dist,   queue_size=1)
        rospy.Subscriber("/wall_angle",       Float32,     self.cb_wall_yaw,    queue_size=1)
        rospy.Subscriber("/wall_has",         Bool,        self.cb_wall_has,    queue_size=1)
        rospy.Subscriber("/obstacle_ahead",   Bool,        self.cb_obs_front,   queue_size=1)
        rospy.Subscriber("/wall_quality",     Float32,     self.cb_wall_quality,queue_size=1)
        rospy.Subscriber("/corner_has",   Bool,         self.cb_corner_has, queue_size=1)
        rospy.Subscriber("/corner_point", PointStamped, self.cb_corner_pt,  queue_size=1)

        # CSV
        self.csvf = open(self.log_path, "w", newline="")
        self.csv = csv.writer(self.csvf)
        self.csv.writerow([
            "time","state","reason",
            "has_wall","obs_front","quality",
            "wall_dist","wall_yaw(rad)","clearance",
            "mem_dist","mem_yaw","mem_age",
            "corner_state","corner_has","corner_dist",
            "vx_cmd","vy_cmd","w_cmd",
            "vx_limited","vy_limited","w_limited",
            "soft_gate"
        ])
        rospy.loginfo("bug2_lock_v2 with CORNER TURN started. log -> %s", self.log_path)
        rospy.loginfo("PARAM: trigger=%.2fm reset=%.2fm turn=%.1fdeg@%.2frad/s cooldown=%.1fs",
                      self.corner_trigger_dist, self.corner_reset_dist,
                      self.corner_turn_angle*180.0/pi, self.corner_turn_speed, self.corner_cooldown)

    # === 订阅回调 ===
    def cb_wall_yaw(self, msg):
        self.wall_yaw = float(msg.data)
        self._yaw_buf.append(self.wall_yaw)
        if len(self._yaw_buf) > self._yaw_buf_len:
            self._yaw_buf.pop(0)
        self.last_perc_t = rospy.Time.now().to_sec()

    def cb_wall_dist(self, msg: Float32):
        self.wall_dist = float(msg.data)
        self.last_perc_t = rospy.Time.now().to_sec()

    def cb_wall_has(self, msg: Bool):
        self.has_wall = 1 if msg.data else 0

    def cb_obs_front(self, msg: Bool):
        self.obs_front = 1 if msg.data else 0

    def cb_wall_quality(self, msg: Float32):
        self.quality = f"{msg.data:.1f}"

    def cb_corner_has(self, msg: Bool):
        self.corner_has = bool(msg.data)

    def cb_corner_pt(self, msg: PointStamped):
        x = msg.point.x; y = msg.point.y
        self.corner_pt = (x, y)
        self.corner_dist = hypot(x, y)
        self.corner_last_stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()

    # === 状态打印 ===
    def set_state(self, new_state, reason=""):
        if new_state != self.state:
            rospy.loginfo("[STATE] %s -> %s | %s | d=%.3f yaw=%.1fdeg corner=%.2fm",
                          self.state, new_state, reason,
                          self.wall_dist if self.wall_dist==self.wall_dist else float('nan'),
                          (self.wall_yaw*180.0/pi) if self.wall_yaw==self.wall_yaw else float('nan'),
                          self.corner_dist if self.corner_dist==self.corner_dist else float('nan'))
            self.state = new_state

    # === 统一发布（限幅+软门+CSV） ===
    def publish(self, vx, vy, wz, reason=""):
        vx_l = max(min(vx, self.limit_vx), -self.limit_vx)
        vy_l = max(min(vy, self.limit_vy), -self.limit_vy)
        wz_l = max(min(wz, self.limit_w ), -self.limit_w )

        target_mag = max(abs(vx_l), abs(vy_l), abs(wz_l))
        if self.soft_enable:
            if target_mag > 1e-6:
                self.soft_gate = min(1.0, self.soft_gate + self.soft_up)
            else:
                self.soft_gate = max(0.0, self.soft_gate - self.soft_down)
        else:
            self.soft_gate = 1.0 if target_mag > 1e-6 else 0.0

        vx_out = vx_l * self.soft_gate
        vy_out = self.cmd_vy_sign * vy_l * self.soft_gate   # ← flip here
        wz_out = wz_l * self.soft_gate

        cmd = Twist()
        cmd.linear.x  = vx_out
        cmd.linear.y  = vy_out
        cmd.angular.z = wz_out
        self.pub_cmd.publish(cmd)
        self.current_vx = vx_out

        now = rospy.Time.now().to_sec()
        mem_age = (now - self.mem_time) if self.mem_time else float('nan')
        self.csv.writerow([
            now, self.state, reason,
            self.has_wall, self.obs_front, self.quality,
            self.wall_dist, self.wall_yaw, self.clearance,
            self.mem_dist, self.mem_yaw, mem_age,
            self.corner_state, int(self.corner_has), self.corner_dist,
            vx, vy, wz,
            vx_l, vy_l, wz_l,
            self.soft_gate
        ])

    def stop(self, reason="STOP"):
        self.publish(0.0, 0.0, 0.0, reason)

    def losses_in_window(self, now):
        self.loss_events = [t for t in self.loss_events if now - t <= self.loss_slowdown_window]
        return len(self.loss_events)
    
    def _wrap(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    # === 角点子状态机 ===
    def corner_fsm(self):
        now = rospy.Time.now().to_sec()

        if self.corner_state == "TURNING":
            wsign = -1.0 if self.side == "right" else +1.0
            self.publish(0.0, 0.0, wsign * self.corner_turn_speed, "CORNER_TURNING")

            done = False

            if (self.turn_yaw_target is not None) and (self.wall_yaw == self.wall_yaw):
                err = abs(self._wrap(self.wall_yaw - self.turn_yaw_target))
                if err <= (self.turn_yaw_tol_deg * np.pi/180.0):
                    self.turn_hold_cnt += 1
                else:
                    self.turn_hold_cnt = 0
                if self.turn_hold_cnt >= self.turn_hold_frames:
                    rospy.loginfo("[CORNER] 墙朝向到位 err=%.1f° 连续%d帧",
                                err*180.0/np.pi, self.turn_hold_frames)
                    done = True

            if (not done) and (self.corner_pt is not None):
                cx, cy = self.corner_pt
                dist_exit = (self.corner_trigger_dist + self.turn_exit_extra_m)
                side_ok = ((self.side=="right" and (cx < 0.0) and (cy < -(self.target_dist+0.10))) or
                        (self.side=="left"  and (cx < 0.0) and (cy >  (self.target_dist+0.10))))
                far_ok  = (self.corner_dist > dist_exit)
                if side_ok or far_ok:
                    rospy.loginfo("[CORNER] 角点退出：%s%s",
                                "身侧/身后 " if side_ok else "",
                                "距离>%.2fm" % dist_exit if far_ok else "")
                    done = True

            if (not done) and (now >= self.corner_turn_end_time or (now - (self.corner_turn_end_time - 9999)) > self.turn_max_time):
                rospy.logwarn("[CORNER] 兜底超时，强制结束转弯")
                done = True

            if done:
                self.corner_state = "COOLDOWN"
                self.corner_cool_end_time = now + self.corner_cooldown
                self.turn_yaw_start = None
                self.turn_yaw_target = None
                self.turn_hold_cnt = 0
                self._yaw_buf[:] = []
                self.set_state("ALIGN", "CORNER_TURN_DONE")
                self.stop("CORNER_TURN_DONE")
            return True

        if self.corner_state == "COOLDOWN":
            if now < self.corner_cool_end_time:
                return False
            else:
                self.corner_state = "IDLE"
                self.corner_armed = False
                self.corner_arm_cnt = 0
                self.corner_prev_dist = None
                rospy.loginfo("[CORNER] 冷却结束，回到 IDLE（解除武装）")
                return False

        if self.corner_state == "APPROACH_STOP":
            if now < self.corner_prepare_end:
                for _ in range(int(self.corner_brake_times)):
                    self.stop("CORNER_PREPARE_STOP")
                return True
            else:
                if len(self._yaw_buf) > 0:
                    yaw_start = sorted(self._yaw_buf)[len(self._yaw_buf)//2]
                else:
                    yaw_start = self.wall_yaw
                self.turn_yaw_start = yaw_start
                turn_sign = -1.0 if self.side == "right" else +1.0
                self.turn_yaw_target = self._wrap(yaw_start + turn_sign * (np.pi/2.0))
                self.turn_hold_cnt = 0

                t_need = abs(self.corner_turn_angle) / max(0.1, abs(self.corner_turn_speed))
                self.corner_turn_end_time = now + max(t_need, 1.5)

                self.corner_state = "TURNING"
                rospy.loginfo("[CORNER] 开始转弯 90.0°(墙角闭环) 目标yaw=%.1f°",
                            self.turn_yaw_target*180.0/np.pi)
                self.stop("CORNER_TURN_BEGIN")
                return True

        if self.corner_state == "IDLE" and self.corner_has and (self.corner_pt is not None):
            dist = self.corner_dist

            if getattr(self, "corner_forward_only", True) and self.corner_pt[0] <= 0.0:
                self.corner_prev_dist = dist
                return False

            if dist > self.corner_reset_dist:
                self.corner_arm_cnt += 1
                if self.corner_arm_cnt >= self.corner_arm_samples:
                    self.corner_armed = True
            else:
                self.corner_arm_cnt = 0

            can_trigger_by_state = (self.state in self.corner_trigger_in_states)
            moving_forward = (abs(self.current_vx) > self.corner_min_vx_to_trigger)

            decreasing = (self.corner_prev_dist is not None) and \
                        ((self.corner_prev_dist - dist) > self.corner_trend_eps)

            if self.corner_armed and can_trigger_by_state and moving_forward and decreasing and (dist < self.corner_trigger_dist):
                self.corner_prepare_end = now + self.corner_prepare_t
                self.corner_state = "APPROACH_STOP"
                self.corner_armed = False
                self.corner_arm_cnt = 0
                rospy.loginfo("[CORNER] 触发@%.2fm → 预停 %.2fs 后转弯", dist, self.corner_prepare_t)
                self.stop("CORNER_PREPARE")
                self.corner_prev_dist = dist
                return True

            # >>> PATCH: 近距直接预停（当场景较小、很难先经历 reset> 的情况）
            if (not self.corner_armed) and can_trigger_by_state and moving_forward and (dist < self.corner_trigger_dist):
                self.corner_prepare_end = now + self.corner_prepare_t
                self.corner_state = "APPROACH_STOP"
                self.corner_armed = False
                self.corner_arm_cnt = 0
                rospy.loginfo("[CORNER] 近距直接预停@%.2fm", dist)
                self.stop("CORNER_PREPARE")
                self.corner_prev_dist = dist
                return True
            # <<< PATCH

            self.corner_prev_dist = dist
            return False

        return False
    def _ensure_side_clearance(self):
        """
        侧向外摆到 safe_clearance（远离当前贴墙一侧），超时则以当下为准。
        返回 True/False 表示是否达到目标（超时也返回 True 继续走后续转弯，以免卡死）。
        """
        want = float(self.safe_clearance)
        deadline = rospy.Time.now().to_sec() + float(self.clear_timeout)
        side_sign = +1.0 if (self.side == "right") else -1.0  # 右墙→+y（向左外摆）

        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            # 没可靠墙距就别硬外摆
            if not self.has_wall or not (self.wall_dist == self.wall_dist):
                self.stop("CLEAR_NO_WALL")
                return False

            e = want - self.wall_dist
            if abs(e) < 0.02:  # 达标
                self.stop("CLEAR_DONE")
                return True

            vy_cmd = side_sign * max(min(self.k_vy * e, self.clear_speed), -self.clear_speed)
            self.publish(0.0, vy_cmd, 0.0, "CLEARING_FOR_TURN")
            rospy.sleep(0.05)

        self.stop("CLEAR_TIMEOUT")
        return True  # 超时也放行，避免卡死

    def _turn_in_place_minus_90(self):
        """
        原地转 -90°（顺时针），只给角速度，vx=vy=0。
        若你之前已经做了“墙角/墙向闭环”，也可以继续用；这里给最简单版的定时兜底。
        """
        turn_sign = 1.0  # changed!!!固定 -90°
        w = turn_sign * self.simple_turn_speed  # 你已有：~simple_turn_speed 缺省 0.6 rad/s
        t_need = (np.pi/2.0) / max(0.1, abs(self.simple_turn_speed))  # 理论时长
        deadline = rospy.Time.now().to_sec() + max(1.5, t_need)       # 兜底

        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            # 可选安全：侧向 watchdog（离墙太近立即停并返回 False）
            if self.has_wall and (self.wall_dist == self.wall_dist) and (self.wall_dist < 0.40):
                self.stop("TURN_ABORT_SIDE_TOO_CLOSE")
                return False

            self.publish(0.0, 0.0, w, "TURN_IN_PLACE_-90")
            rospy.sleep(0.02)

        self.stop("TURN_DONE_SIMPLE")
        return True

    def turn_with_clearance_and_resume(self):
        """
        简单总流程：先外摆到安全侧距 → 原地转 -90° → 回 ALIGN，让原有沿墙接管。
        """
        self.stop("TURN_PREPARE")
        ok_clear = self._ensure_side_clearance()
        ok_turn  = self._turn_in_place_minus_90()
        self.set_state("ALIGN", "AFTER_TURN")
        self.stop("AFTER_TURN_STOP")
        return ok_clear and ok_turn
        # ===== 在类中新增：极简拐角三步 =====
    def _side_offset_until_clear(self):
        """若贴墙太近，先侧向让位到 target_dist + extra，避免转弯扫墙。"""
        want = self.target_dist + self.simple_clear_extra
        deadline = rospy.Time.now().to_sec() + self.simple_side_timeout
        # right → +y (left), left → -y (right)
        sign = +1.0 if (self.side == "right") else -1.0

        while not rospy.is_shutdown() and rospy.Time.now().to_sec() < deadline:
            if self.has_wall and (self.wall_dist == self.wall_dist):
                e = want - self.wall_dist
                if abs(e) < 0.02:  # 足够净空
                    self.stop("SIDE_CLEAR_DONE")
                    return True
                vy_cmd = sign * max(min(self.k_vy * e, self.simple_side_speed), -self.simple_side_speed)
                self.publish(0.0, vy_cmd, 0.0, "SIDE_CLEARING")
            else:
                # 没可靠墙距就别强求
                self.stop("SIDE_CLEAR_NO_WALL")
                return False
            rospy.sleep(0.05)

        self.stop("SIDE_CLEAR_TIMEOUT")
        return True  # 即便超时，也继续转（保守处理）


    def _turn_90_by_wall_yaw(self):
        """原地 90° 转，靠 /wall_angle 做闭环校验，超时兜底。"""
        # 当前墙朝向作为起点
        yaw_start = self.wall_yaw if self.wall_yaw == self.wall_yaw else 0.0
        turn_sign = 1.0 if self.side == "right" else -1.0
        yaw_target = self._wrap(yaw_start + turn_sign * (np.pi/2.0))

        tol = self.simple_yaw_tol_deg * np.pi/180.0
        hold_need = self.simple_yaw_hold_frames
        hold_cnt = 0
        deadline = rospy.Time.now().to_sec() + self.simple_turn_timeout
        turn_start = rospy.Time.now().to_sec()


        while not rospy.is_shutdown():
            # 安全闸：前方障碍则停
            #if self.obs_front:
            #    self.stop("TURN_SAFE_STOP_OBS")
            #    return False

            # 角度闭环
            if self.wall_yaw == self.wall_yaw:
                err = abs(self._wrap(self.wall_yaw - yaw_target))
                if err <= tol: hold_cnt += 1
                else:          hold_cnt  = 0
                if hold_cnt >= hold_need:
                    self.stop("TURN_DONE_BY_YAW")
                    return True

            # 兜底超时
            if rospy.Time.now().to_sec() > deadline:
                self.stop("TURN_TIMEOUT_DONE")
                return True

            # 原地转 -> 改为带轻微后退
            vx_back = 0.0
            if self.turn_back_enable and (rospy.Time.now().to_sec() - turn_start) < self.turn_back_duration:
                vx_back = self.turn_back_vx
            vy_push = self.turn_side_push_vy 
            self.publish(vx_back, vy_push, turn_sign * self.simple_turn_speed, "TURN_90")


    def simple_corner_manager(self):
        """
        极简角点逻辑：
        1) 角点存在 && 距离 <= 1.2m（默认）&&（可选：角点在前方）
        2) 若侧向净空不足，先让位
        3) 原地 90° 转，转完回 ALIGN
        """
        # 角点条件
        if not self.corner_has or (self.corner_dist != self.corner_dist):
            return False
        if self.corner_dist > self.simple_corner_turn_dist:
            return False
        if self.simple_forward_only and self.corner_pt and (self.corner_pt[0] <= 0.0):
            return False  # 只接受前方角点

        # Step 1：停稳（你已有）
        self.stop("CORNER_TRIGGERED")

        # Step 2：外摆到安全侧距 + 原地 -90°
        self.turn_with_clearance_and_resume()

        # 返回 True 让主循环这帧结束
        return True


        # === 主循环 ===
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            vx = vy = wz = 0.0
            reason = ""
            if self.simple_corner_manager():
                rate.sleep()
                continue
                        # 角点子 FSM：若需要转弯则拦截一切
            #if self.corner_fsm():
            #    rate.sleep()
             #   continue

            # >>> PATCH: 全局安全闸——任何状态下遇到前障/近距角点直接刹停
            #if self.obs_front:
            #    self.stop("SAFE_STOP_OBS_FRONT")
             #   rate.sleep(); continue
            #if self.corner_has and (self.corner_dist == self.corner_dist) and (self.corner_dist < 0.55):
            #    self.stop("SAFE_STOP_CORNER_NEAR")
            #    rate.sleep(); continue
            # <<< PATCH

            perc_age = (now - self.last_perc_t) if self.last_perc_t else 999

            if self.state == "ALIGN":
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    self.mem_dist = self.wall_dist
                    self.mem_yaw  = self.wall_yaw
                    self.mem_time = now

                    e_d   = self.target_dist - self.wall_dist
                    e_yaw = -self.wall_yaw

                    side_dir = +1.0 if (self.side == "right") else -1.0  # right→+y(左), left→-y(右)
                    vy = side_dir * max(min(self.k_vy * e_d, self.max_vy), -self.max_vy)
                    wz = max(min(self.k_w  * e_yaw, self.max_w), -self.max_w)
                    vx = 0.05
                    reason = "ALIGNING"

                    if abs(e_d) <= self.align_tol_d and abs(self.wall_yaw) <= self.align_tol_y:
                        self.set_state("LOCKED_GO", "ALIGN_DONE")
                        self.stop("ALIGN_DONE")
                        rate.sleep()
                        continue
                else:
                    self.set_state("HOLD", "ALIGN_LOSS→HOLD")
                    self.loss_events.append(now)
                    vx = self.forward_speed_hold
                    reason = "ALIGN_LOSS→HOLD"

            elif self.state == "LOCKED_GO":
                vx = self.forward_speed

                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    e_d = self.target_dist - self.wall_dist
                    e_d_eff = 0.0 if abs(e_d) < self.deadzone_d else e_d
                    side_dir = +1.0 if (self.side == "right") else -1.0
                    vy_raw = side_dir * self.k_vy_go * e_d_eff
                    vy = max(min(vy_raw, self.max_vy_go), -self.max_vy_go)

                    e_yaw = -self.wall_yaw
                    e_yaw_eff = 0.0 if abs(e_yaw) < self.deadzone_yaw else e_yaw
                    wz_raw = self.k_w_go * e_yaw_eff
                    wz = max(min(wz_raw, self.max_w_go), -self.max_w_go)

                    reason = "LOCKED_GO(vx+vy+wz)"
                    self.mem_dist = self.wall_dist
                    self.mem_yaw  = self.wall_yaw
                    self.mem_time = now
                else:
                    self.set_state("HOLD", "LOCKED→HOLD")
                    self.loss_events.append(now)
                    self.stop("LOCKED→HOLD")
                    rate.sleep()
                    continue

            elif self.state == "HOLD":
                vx = self.forward_speed_hold
                reason = "HOLD_KEEP_STRAIGHT"

                # >>> PATCH: 首次再捕获初始化——一旦有墙信息就建立记忆并回ALIGN
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw) and (not self.mem_time):
                    self.mem_dist = self.wall_dist
                    self.mem_yaw  = self.wall_yaw
                    self.mem_time = now
                    self.set_state("ALIGN", "HOLD→ALIGN_INIT_REACQ")
                    self.stop("HOLD→ALIGN_INIT_REACQ")
                    rate.sleep()
                    continue
                # <<< PATCH

                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw) and self.mem_time:
                    if abs(self.wall_dist - self.mem_dist) <= self.reacq_tol_d and abs(self.wall_yaw - self.mem_yaw) <= self.reacq_tol_y:
                        self.set_state("LOCKED_GO", "REACQ→LOCKED")
                        self.stop("REACQ→LOCKED")
                        rate.sleep()
                        continue
                    else:
                        self.mem_dist = self.wall_dist
                        self.mem_yaw  = self.wall_yaw
                        self.mem_time = now
                        reason = "HOLD_MEM_UPDATE"

                nloss = self.losses_in_window(now)
                if nloss >= self.loss_slowdown_thresh and self.forward_speed > self.min_forward_speed:
                    self.forward_speed = max(self.min_forward_speed, self.forward_speed - 0.02)
                    self.forward_speed_hold = max(self.min_forward_speed, self.forward_speed_hold - 0.02)
                    reason = f"HOLD_SLOWDOWN(n={nloss})"

            elif self.state == "FAIL_STOP":
                self.stop("STOP_STATE")
                rate.sleep()
                continue

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
