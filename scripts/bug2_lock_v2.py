#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bug2_lock_v4_openloop.py
- TURN: time + optional wall-angle snap；snap目标=本次“平行零点”（_yaw_trim_epi或yaw_trim）
- ALIGN: 距离优先的“软冻结” yaw；首次ALIGN冻结wz取中位零点；后续ALIGN短窗自零点
- STEP_OUT: 单阶段（vx+vy 同时），先保侧向余量，再小幅前爬把cx压到目标并hold
- Post-TURN: 强制进入ALIGN，用更严角度容差；LOCKED_GO发现不平行立即降级ALIGN
"""

import os, csv, time, math, numpy as np
import rospy
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist, PointStamped

# ---------- helpers ----------
def clamp(x, lo, hi): return max(lo, min(hi, x))
def ang_norm(a): return (a + math.pi) % (2*math.pi) - math.pi

# ---------- lightweight visited grid (optional) ----------
class VisitedGrid:
    def __init__(self, res=0.25, mark_step=0.25, ahead_cells=8, decay=0.0):
        self.res, self.mark_step, self.ahead_cells, self.decay = float(res), float(mark_step), int(ahead_cells), float(decay)
        self.grid, self.last_mark_pose = {}, None
    def _key(self, x, y):
        r = self.res
        return (int(round(x/r)), int(round(y/r)))
    def mark(self, x, y):
        if any(map(lambda v: v!=v, (x,y))):  # NaN check
            return False
        if self.last_mark_pose is None or math.hypot(x-self.last_mark_pose[0], y-self.last_mark_pose[1]) >= self.mark_step:
            self.last_mark_pose = (x, y)
            k = self._key(x,y)
            self.grid[k] = self.grid.get(k, 0) + 1
            if self.decay > 0:
                for kk in list(self.grid.keys()):
                    self.grid[kk] *= (1.0 - self.decay)
                    if self.grid[kk] < 0.25: self.grid.pop(kk, None)
            return True
        return False
    def score_ahead(self, x, y, yaw):
        if any(map(lambda v: v!=v, (x,y,yaw))): return float('nan')
        s, c, sY = 0.0, math.cos(yaw), math.sin(yaw)
        for i in range(1, self.ahead_cells+1):
            px, py = x + i*self.res*c, y + i*self.res*sY
            s += self.grid.get(self._key(px,py), 0)
        return s

# ---------- main controller ----------
class WallFollower:
    def __init__(self):
        # ========== BASIC ==========
        self.side = rospy.get_param("~side", "right")
        self.cmd_vx_sign = rospy.get_param("~cmd_vx_sign", 1.0)
        self.cmd_vy_sign = rospy.get_param("~cmd_vy_sign", 1.0)

        # Align/lock gains and targets
        self.target_dist  = rospy.get_param("~target_dist", 0.60)
        self.align_tol_d  = rospy.get_param("~align_tol_d", 0.05)
        self.k_vy         = rospy.get_param("~k_vy", 1.2)
        self.k_w          = rospy.get_param("~k_w", 1.5)
        self.max_vy       = rospy.get_param("~max_vy", 0.25)
        self.max_w        = rospy.get_param("~max_w",  0.65)
        self.forward_speed      = rospy.get_param("~forward_speed", 0.05)#0.18
        self.forward_speed_hold = rospy.get_param("~forward_speed_hold", 0.05)
        self.deadzone_d   = rospy.get_param("~deadzone_d", 0.02)
        self.deadzone_y   = rospy.get_param("~deadzone_y", 0.12)
        self.yaw_trim     = rospy.get_param("~yaw_trim", 0.106)  # 全局零点；首次/后续ALIGN会给会话零点

        # Loss slow-down policy
        self.loss_window  = rospy.get_param("~loss_window", 20)
        self.loss_slowdown_thresh = rospy.get_param("~loss_slowdown_thresh", 3)

        # Output limits + soft start
        self.limit_vx   = rospy.get_param("~limit_vx", 0.22)
        self.limit_vy   = rospy.get_param("~limit_vy", 0.22)
        self.limit_w    = rospy.get_param("~limit_w",  1.00)
        self.soft_enable= rospy.get_param("~soft_enable", True)
        self.soft_up    = rospy.get_param("~soft_up",   0.04)
        self.soft_down  = rospy.get_param("~soft_down", 0.06)
        self._soft_gate = 0.0

        # ========== ROBOT GEOMETRY & TURN NEED ==========
        self.robot_L = rospy.get_param("~robot_length", 1.00)
        self.robot_W = rospy.get_param("~robot_width",  0.66)
        self.turn_padding = rospy.get_param("~turn_padding", 0.25)
        self.R_turn_need = 0.5*math.hypot(self.robot_L, self.robot_W) + self.turn_padding

        # ========== CORNER FSM ==========
        self.corner_forward_need_extra = rospy.get_param("~corner_forward_need_extra", 0.01)
        self.corner_side_need_extra    = rospy.get_param("~corner_side_need_extra",    0.15)
        self.corner_trigger_x          = rospy.get_param("~corner_trigger_x", 0.01)
        self.corner_arm_samples        = rospy.get_param("~corner_arm_samples", 1)
        self.corner_step_vy            = rospy.get_param("~corner_step_vy", 0.82)
        self.corner_turn_angle_deg     = rospy.get_param("~corner_turn_angle_deg", 90.0)
        self.corner_w_max              = rospy.get_param("~corner_w_max", 1.0)
        self.corner_w_accel            = rospy.get_param("~corner_w_accel", 1.2)
        self.corner_step_timeout       = rospy.get_param("~corner_step_timeout", 3.0)
        self.corner_turn_timeout       = rospy.get_param("~corner_turn_timeout", 15.0)
        self.corner_cooldown           = rospy.get_param("~corner_cooldown", 1.0)
        self.corner_yaw_kp             = rospy.get_param("~corner_yaw_kp", 2.0)
        self.corner_yaw_dead_deg       = rospy.get_param("~corner_yaw_dead_deg", 3.0)

        # ========== TURN BY TIME / WALL-SNAP ==========
        self.turn_mode        = rospy.get_param("~turn_mode", "time")  # "time" | "wall" | "yaw"
        self.turn_w_set       = rospy.get_param("~turn_w_set", 0.80)
        self.turn_time_gain   = rospy.get_param("~turn_time_gain", 15.00)
        self.turn_time_margin = rospy.get_param("~turn_time_margin", 1.15)
        self.turn_time_min    = rospy.get_param("~turn_time_min", 0.5)
        self.turn_time_max    = rospy.get_param("~turn_time_max", 17.5)
        self.wall_snap_eps_deg= rospy.get_param("~wall_snap_eps_deg", 90.0) #6.0
        self.wall_snap_hold   = rospy.get_param("~wall_snap_hold", 0.30)
        self._turn_T = None; self._turn_t0 = None; self._snap_t0 = None

        # ========== OPEN-LOOP YAW EST ==========
        self.yaw_openloop_gain = rospy.get_param("~yaw_openloop_gain", 1.0)
        self.yaw_est = 0.0
        self.yaw_est_stamp = rospy.Time.now().to_sec()
        self.pub_turn_deg  = rospy.Publisher("/turn_angle_deg", Float32, queue_size=1)
        self.pub_turn_done = rospy.Publisher("/turn_done",     Bool,    queue_size=1)
        self.turn_dead_deg = rospy.get_param("~turn_dead_deg", 3.0)
        self.turn_stable_N = rospy.get_param("~turn_stable_N", 3)
        self._turn_stable_cnt = 0

        # ========== VISITED GRID (optional) ==========
        self.visited_enable   = rospy.get_param("~visited_enable", True)
        self.vis_res          = rospy.get_param("~visited_res", 0.25)
        self.vis_mark_step    = rospy.get_param("~visited_mark_step", 0.25)
        self.vis_ahead_cells  = rospy.get_param("~visited_ahead_cells", 8)
        self.vis_decay        = rospy.get_param("~visited_decay", 0.0)
        self.vis = VisitedGrid(self.vis_res, self.vis_mark_step, self.vis_ahead_cells, self.vis_decay)

        # ========== IO & LOG ==========
        self.pub_cmd = rospy.Publisher(rospy.get_param("~cmd_vel", "/cmd_vel"), Twist, queue_size=1)
        logdir = rospy.get_param("~log_dir", "/tmp")
        os.makedirs(logdir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(logdir, f"wf_lock_v4_{ts}.csv")
        self.csvf = open(self.log_path, "w", newline="")
        self.csv = csv.writer(self.csvf)
        self.csv.writerow([
            "time","state","reason","has_wall","obs_front","quality",
            "wall_dist","wall_yaw(rad)","front_min","left_min","mem_dist","mem_yaw","mem_age",
            "corner_state","corner_has","corner_dist","odom_x","odom_y","odom_yaw","vis_score_ahead",
            "vx_cmd","vy_cmd","w_cmd","vx_limited","vy_limited","w_limited","soft_gate"
        ])
        rospy.loginfo("bug2_lock_v4_openloop started. log -> %s", self.log_path)

        # SUBS (from perception)
        rospy.Subscriber("/wall_lateral", Float32, self.cb_wall_dist, queue_size=1)
        rospy.Subscriber("/wall_angle",   Float32, self.cb_wall_yaw,  queue_size=1)
        rospy.Subscriber("/wall_has",     Bool,    self.cb_wall_has,  queue_size=1)
        rospy.Subscriber("/obstacle_ahead", Bool,  self.cb_obs_front, queue_size=1)
        rospy.Subscriber("/wall_quality", Float32, self.cb_wall_quality, queue_size=1)
        rospy.Subscriber("/corner_has",   Bool,    self.cb_corner_has, queue_size=1)
        rospy.Subscriber("/corner_point", PointStamped, self.cb_corner_pt, queue_size=1)
        rospy.Subscriber("/front_min",    Float32, self.cb_front_min, queue_size=1)
        rospy.Subscriber("/left_min",     Float32, self.cb_left_min,  queue_size=1)
        self.right_min = float('inf')
        rospy.Subscriber("/right_min",    Float32, lambda m: setattr(self,"right_min", float(m.data)), queue_size=1)

        # RUNTIME mem_dist
        self.state = "ALIGN"
        self.has_wall = 0; self.obs_front = 0; self.quality = "nan"
        self.wall_dist = float('nan')
        self.wall_yaw  = float('nan')
        self.front_min = float('inf')
        self.left_min  = float('inf')
        self.mem_dist  = float('nan')
        self.mem_yaw   = float('nan')
        self.mem_time  = None

        # Corner runtime
        self.corner_has = False; self.corner_pt = None; self.corner_dist = float('nan')
        self.corner_state = "C_IDLE"
        self._c_t_enter   = 0.0
        self._turn_yaw0   = None
        self._wz_prev     = 0.0
        self._last_time   = rospy.Time.now().to_sec()
        self._corner_arm_cnt = 0

        # Loss slow-down
        self.loss_hist = [0]*self.loss_window
        self.loss_ptr  = 0

        # ---------- ALIGN tuning ----------
        self.align_w_max         = rospy.get_param("~align_w_max", 0.6)   # ALIGN阶段角速度上限
        self.dist_freeze_on      = rospy.get_param("~dist_freeze_on", 0.10)
        self.dist_freeze_scale   = rospy.get_param("~dist_freeze_scale", 0.30)
        self.align_tol_y         = rospy.get_param("~align_tol_y", 0.12)  # ≈6.9°

        # ---------- POST-TURN ALIGN POLICY ----------
        self.post_turn_force_align      = rospy.get_param("~post_turn_force_align", True)
        self.post_turn_align_lock_time  = rospy.get_param("~post_turn_align_lock_time", 1.0)
        self.post_turn_align_tol_y      = rospy.get_param("~post_turn_align_tol_y", min(self.align_tol_y, 0.12))
        self.locked_to_align_y          = rospy.get_param("~locked_to_align_y", max(0.12, self.align_tol_y))
        self._post_turn_until           = 0.0

        # ---------- Auto-zero for later ALIGNs ----------
        self.auto_yaw_zero          = rospy.get_param("~auto_yaw_zero", True)
        self.auto_yaw_window        = rospy.get_param("~auto_yaw_window", 0.40)  # s
        self.auto_yaw_samples_min   = rospy.get_param("~auto_yaw_samples_min", 6)
        self.auto_yaw_dist_gate     = rospy.get_param("~auto_yaw_dist_gate", 0.06)   # m
        self.auto_yaw_var_gate_deg  = rospy.get_param("~auto_yaw_var_gate_deg", 2.0) # deg
        self._az_t0        = None
        self._az_buf       = []
        self._yaw_trim_epi = None
        self._prev_state   = None

        # ---------- FIRST ALIGN freeze & calibration ----------
        self.first_align_freeze       = rospy.get_param("~first_align_freeze", True)
        self.first_align_zero_window  = rospy.get_param("~first_align_zero_window", 0.50)  # s
        self.first_align_samples_min  = rospy.get_param("~first_align_samples_min", 6)
        self.first_align_var_gate_deg = rospy.get_param("~first_align_var_gate_deg", 2.5)  # deg
        self.first_align_dist_gate    = rospy.get_param("~first_align_dist_gate", 0.10)    # m
        self.first_align_require_dist = rospy.get_param("~first_align_require_dist", False)
        self.first_align_freeze_max   = rospy.get_param("~first_align_freeze_max", 2.0)    # s
        self._first_align_active = False
        self._first_align_done   = False
        self._first_align_t0     = None
        self._first_align_buf    = []

        # ----- STEP_OUT single-phase approach-X（转前贴角）-----
        self.corner_cx_offset   = rospy.get_param("~corner_cx_offset", 0.0)   # 雷达到车头/转向参考的x偏置
        self.corner_cx_target   = rospy.get_param("~corner_cx_target", 0.06)  # 期望转弯时的cx
        self.corner_cx_min      = rospy.get_param("~corner_cx_min",    0.03)  # 不小于此，留保險
        self.corner_cx_hold     = rospy.get_param("~corner_cx_hold",   0.25)  # 到位保持时长
        self.corner_approach_vx = rospy.get_param("~corner_approach_vx", 0.08)
        self.corner_approach_timeout = rospy.get_param("~corner_approach_timeout", 1.5)
        self.corner_cx_beta     = rospy.get_param("~corner_cx_beta",   0.35)
        self.front_stop_min     = rospy.get_param("~front_stop_min",   0.12)

        self._cx_filt = None
        self._cx_hold_t0 = None
        self._approach_t0 = None

    # ---------- callbacks ----------
    def cb_wall_dist(self, msg):
        self.wall_dist = float(msg.data)
        if self.has_wall and self.wall_dist == self.wall_dist:
            self.mem_dist = self.wall_dist
            self.mem_time = rospy.Time.now().to_sec()

    def cb_wall_yaw(self, msg):
        self.wall_yaw = float(msg.data)  # 墙切向角（≈0 表示与车身平行）
        if self.has_wall and self.wall_yaw == self.wall_yaw:
            self.mem_yaw = self.wall_yaw

    def cb_wall_has(self, msg): self.has_wall = int(bool(msg.data))
    def cb_obs_front(self, msg): self.obs_front = int(bool(msg.data))
    def cb_wall_quality(self, msg): self.quality = f"{float(msg.data):.1f}"
    def cb_front_min(self, msg): self.front_min = float(msg.data)
    def cb_left_min(self, msg):  self.left_min  = float(msg.data)

    def cb_corner_pt(self, msg: PointStamped):
        x, y = msg.point.x, msg.point.y
        self.corner_pt = (float(x), float(y))
        self.corner_dist = math.hypot(x,y)
        if self.corner_has and (x==x) and (x>0):
            self._corner_arm_cnt = min(self.corner_arm_samples, self._corner_arm_cnt+1)
        else:
            self._corner_arm_cnt = 0

    def cb_corner_has(self, msg: Bool):
        self.corner_has = bool(msg.data)
        if not self.corner_has:
            self.corner_dist = float('inf')
            self.corner_pt   = None
            self._corner_arm_cnt = 0

    # ---------- helpers ----------
    def _is_wall_following_simple(self):
        if self.has_wall and (self.wall_dist == self.wall_dist) and (self.wall_yaw == self.wall_yaw):
            dist_ok  = abs(self.wall_dist - self.target_dist) <= (self.align_tol_d + 0.02)
            yaw_meas = self.wall_yaw - self.yaw_trim
            yaw_ok   = abs(yaw_meas) <= (self.align_tol_y + 1e-6)
            return dist_ok and yaw_ok
        return False

    def _limit_soft(self, vx, vy, wz):
        vx = clamp(vx, -self.limit_vx, self.limit_vx)
        vy = clamp(vy, -self.limit_vy, self.limit_vy)
        wz = clamp(wz, -self.limit_w,  self.limit_w)
        gate_for_w = 1.0 if self.corner_state == "C_TURN" else self._soft_gate
        if self.soft_enable:
            mag = max(abs(vx), abs(vy), abs(wz))
            if mag > 1e-3: self._soft_gate = min(1.0, self._soft_gate + self.soft_up)
            else:          self._soft_gate = max(0.0, self._soft_gate - self.soft_down)
            vx *= self._soft_gate
            vy *= self._soft_gate
            wz *= gate_for_w
        return vx, vy, wz

    def _get_pose_odom(self):
        """No IMU/odom: return NaN x,y and open-loop yaw_est"""
        return float('nan'), float('nan'), self.yaw_est

    def _publish(self, vx, vy, wz, reason=""):
        vx_l, vy_l, wz_l = self._limit_soft(vx, vy, wz)
        now = rospy.Time.now().to_sec()
        dt  = max(1e-3, now - self.yaw_est_stamp)
        self.yaw_est += (self.yaw_openloop_gain * wz_l) * dt
        self.yaw_est = ang_norm(self.yaw_est)
        self.yaw_est_stamp = now

        msg = Twist()
        msg.linear.x  = vx_l * self.cmd_vx_sign
        msg.linear.y  = vy_l * self.cmd_vy_sign
        msg.angular.z = wz_l
        self.pub_cmd.publish(msg)

        ox, oy, oyaw = self._get_pose_odom()
        vis_score = self.vis.score_ahead(ox, oy, oyaw) if self.visited_enable else float('nan')

        tnow = rospy.Time.now().to_sec()
        mem_age = (tnow - self.mem_time) if self.mem_time else float('nan')
        self.csv.writerow([tnow, self.state, reason, int(self.has_wall), int(self.obs_front), self.quality,
                           self.wall_dist, self.wall_yaw, self.front_min, self.left_min,
                           self.mem_dist, self.mem_yaw, mem_age,
                           self.corner_state, int(self.corner_has), self.corner_dist,
                           ox, oy, self.yaw_est, vis_score,
                           vx, vy, wz, vx_l, vy_l, wz_l, self._soft_gate])
        try: self.csvf.flush()
        except Exception: pass

    def stop(self, reason="STOP"):
        self._publish(0.0, 0.0, 0.0, reason)

    # ---------- Corner FSM ----------
    def corner_fsm(self):
        """trigger by corner_x (+radius fallback) -> STEP_OUT (single-phase) -> TURN(by time/optional snap) -> COOL"""
        now = rospy.Time.now().to_sec()
        dt  = max(0.02, now - self._last_time)
        self._last_time = now

        R_turn_need = self.R_turn_need
        need_front  = R_turn_need + self.corner_forward_need_extra
        need_side   = R_turn_need + self.corner_side_need_extra

        # corner_x (forward component)
        cx = float('inf')
        if self.corner_has and (self.corner_pt is not None):
            px, py = self.corner_pt
            if (px==px) and (px>0) and (self._corner_arm_cnt >= self.corner_arm_samples):
                cx = px

        # opposite-side clearance
        opp_min = self.left_min if self.side == "right" else self.right_min
        if not np.isfinite(opp_min): opp_min = float('inf')

        # 半径兜底（可作为额外触发条件）
        r_ok = self.corner_has and (self.corner_dist == self.corner_dist) and (self.corner_dist < (R_turn_need + 0.15))
        x_ok = (cx < self.corner_trigger_x)

        s = self.corner_state

        # 1) trigger
        if s == "C_IDLE":
            yaw_ok = (self.wall_yaw==self.wall_yaw) and (abs(self.wall_yaw - self.yaw_trim) <= self.align_tol_y)
            if (self._is_wall_following_simple() and yaw_ok and (x_ok or r_ok)):
                self.corner_state = "C_STEP_OUT"
                self._c_t_enter   = now
                self._wz_prev     = 0.0
                self._turn_yaw0   = None
                self._turn_T = None; self._turn_t0 = None; self._snap_t0 = None
                self._cx_filt = None; self._cx_hold_t0 = None; self._approach_t0 = None
                self.stop("C_IDLE→C_STEP_OUT (cx=%.2f r=%.2f)" % (cx, self.corner_dist))
                return True
            return False

        # 2) STEP_OUT —— 单阶段：同时 vx + vy（侧向优先，前爬贴角）
        if s == "C_STEP_OUT":
            ok_side  = (opp_min >= need_side)
            if self._approach_t0 is None: self._approach_t0 = now
            timeoutA = ((now - self._c_t_enter) > self.corner_step_timeout)
            timeoutB = ((now - self._approach_t0) > self.corner_approach_timeout)
            timeout  = timeoutA or timeoutB

            # 角点x：补偿 + 轻滤波
            cx_corr = cx - self.corner_cx_offset
            if not np.isfinite(cx_corr):
                cx_f = float('inf')
            else:
                self._cx_filt = cx_corr if (self._cx_filt is None) else ((1.0 - self.corner_cx_beta)*self._cx_filt + self.corner_cx_beta*cx_corr)
                cx_f = float(self._cx_filt)

            # 目标到位 & 保持
            target_ok = ok_side and (cx_f <= self.corner_cx_target) and (cx_f >= self.corner_cx_min)
            if target_ok:
                if self._cx_hold_t0 is None: self._cx_hold_t0 = now
                hold_ok = ((now - self._cx_hold_t0) >= self.corner_cx_hold)
            else:
                self._cx_hold_t0 = None
                hold_ok = False

            # vy：侧向未达标 -> 给足；已达标 -> 40% 维持
            vy_mag = (self.corner_step_vy if not ok_side else 0.4*self.corner_step_vy)
            vy = vy_mag * ( +1.0 if self.side=="right" else -1.0 ) * self.cmd_vy_sign

            # vx：仅在侧向达标 + 前方安全 + cx偏大时前爬
            front_safe = (not np.isfinite(self.front_min)) or (self.front_min >= self.front_stop_min)
            vx = self.corner_approach_vx if (ok_side and front_safe and (cx_f > self.corner_cx_target)) else 0.0

            # 进入转弯：目标已保持 or 超时兜底
            if hold_ok or timeout:
                self.corner_state = "C_TURN"
                self._c_t_enter   = now
                self._turn_yaw0   = self._get_pose_odom()[2]
                self._wz_prev     = 0.0
                self._soft_gate   = 1.0

                # snap目标 = 本次会话平行零点（优先 _yaw_trim_epi）
                base = self._yaw_trim_epi if (self._yaw_trim_epi is not None) else self.yaw_trim
                self.wall_yaw_enter = self.wall_yaw if (self.has_wall and self.wall_yaw == self.wall_yaw) else None
                self.yaw_trim_turn  = float(base)

                # 定时兜底
                ang   = math.radians(self.corner_turn_angle_deg)
                T_nom = self.turn_time_gain * ang / max(1e-3, self.turn_w_set) + self.turn_time_margin
                self._turn_T  = min(self.turn_time_max, max(self.turn_time_min, T_nom))
                self._turn_t0 = now
                self._snap_t0 = None

                self.stop("C_STEP_OUT→C_TURN (cx=%.2f tgt=%.2f side=%.2f/%.2f front=%.2f T=%.2fs%s)" %
                          (cx_f, self.corner_cx_target, opp_min, need_side, self.front_min, self._turn_T,
                           " timeout" if timeout else ""))
                return True

            # 继续单阶段贴角
            self._publish(vx, vy, 0.0, "C_STEP_OUT combo vx=%.2f vy=%.2f cx=%.2f side=%.2f/%.2f front=%.2f"
                          % (vx, vy, cx_f, opp_min, need_side, self.front_min))
            return True

        # 3) TURN (by time, optional wall snap)
        if s == "C_TURN":
            sign = -1.0 if self.side == "right" else +1.0

            # 误差闭环 + 加速度限幅
            use_err = (self.turn_mode == "wall") and self.has_wall and (self.wall_yaw == self.wall_yaw)
            if use_err:
                err  = ang_norm(self.yaw_trim_turn - self.wall_yaw)
                k_wt = getattr(self, "k_w_turn", 1.5)
                w_des = clamp(k_wt * err, -self.turn_w_set, self.turn_w_set)
            else:
                w_des = sign * self.turn_w_set

            dv = self.corner_w_accel * dt
            w_cmd = self._wz_prev
            if w_des > w_cmd + dv: w_cmd = w_cmd + dv
            elif w_des < w_cmd - dv: w_cmd = w_cmd - dv
            else: w_cmd = w_des
            self._wz_prev = w_cmd

            elapsed   = now - (self._turn_t0 if self._turn_t0 is not None else now)
            done_time = (self._turn_T is not None) and (elapsed >= self._turn_T)

            # snap：最小时间门槛后，直接对“平行零点”判稳并hold
            done_snap = False
            allow_snap = (elapsed >= self.turn_time_min)
            if self.turn_mode in ("wall", "time") and self.has_wall and (self.wall_yaw == self.wall_yaw):
                if allow_snap:
                    err_snap = ang_norm(self.wall_yaw - self.yaw_trim_turn)
                    if abs(err_snap) <= math.radians(self.wall_snap_eps_deg):
                        if self._snap_t0 is None: self._snap_t0 = now
                        if (now - self._snap_t0) >= self.wall_snap_hold:
                            done_snap = True
                    else:
                        self._snap_t0 = None
            else:
                self._snap_t0 = None

            finish = False
            if self.turn_mode == "time":
                finish = done_time
            elif self.turn_mode == "wall":
                finish = done_snap or done_time
            else:
                finish = done_time

            if finish:
                self.corner_state = "C_COOL"
                self._c_t_enter   = now
                self._wz_prev     = 0.0

                # 转后强制进入 ALIGN，开启更严容差，并重置later-align自零点窗口
                if self.post_turn_force_align:
                    self.state = "ALIGN"
                    self._post_turn_until = now + self.post_turn_align_lock_time
                    if self.auto_yaw_zero:
                        self._az_t0 = now
                        self._az_buf = []
                        self._yaw_trim_epi = None

                self.stop("C_TURN→C_COOL (t=%.2fs/T=%.2fs snap=%s allow=%s)" %
                          (elapsed, self._turn_T or -1, done_snap, allow_snap))
                return True

            self._publish(0.0, 0.0, w_cmd, "C_TURN t=%.2f/%.2f wz=%.2f" % (elapsed, self._turn_T or -1, w_cmd))
            return True

        # 4) COOL
        if s == "C_COOL":
            if (now - self._c_t_enter) >= self.corner_cooldown:
                self.corner_state = "C_IDLE"
                if self.post_turn_force_align:
                    self.state = "ALIGN"
                self.stop("C_COOL→C_IDLE")
                return False
            self.stop("C_COOL")
            return True

        # fallback
        self.corner_state = "C_IDLE"
        return False

    # ---------- main FSM ----------
    def run(self):
        rate = rospy.Rate(30)
        hold_loss_cnt = 0
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()

            # Corner FSM 优先
            if self.corner_fsm():
                rate.sleep(); continue

            # 进入ALIGN的瞬间：首次ALIGN/后续ALIGN的零点窗口
            entered_align = (self.state == "ALIGN" and self._prev_state != "ALIGN")
            if entered_align:
                if (not self._first_align_done) and self.first_align_freeze:
                    self._first_align_active = True
                    self._first_align_t0 = now
                    self._first_align_buf = []
                if self.auto_yaw_zero:
                    self._az_t0 = now
                    self._az_buf = []
                    self._yaw_trim_epi = None
            self._prev_state = self.state

            vx = vy = wz = 0.0
            reason = ""

            if self.state == "ALIGN":
                # FIRST ALIGN：冻结wz，收集中位零点
                if self._first_align_active and (not self._first_align_done):
                    if self.has_wall and (self.wall_yaw == self.wall_yaw):
                        dist_ok = (abs(self.target_dist - self.wall_dist) <= self.first_align_dist_gate) if self.first_align_require_dist else True
                        if dist_ok:
                            self._first_align_buf.append(self.wall_yaw)

                    elapsed = now - self._first_align_t0
                    got_enough = (len(self._first_align_buf) >= self.first_align_samples_min)
                    stable = False
                    if got_enough:
                        spread = (max(self._first_align_buf) - min(self._first_align_buf))
                        stable = (spread <= math.radians(self.first_align_var_gate_deg))

                    lock_now = (got_enough and stable) or (elapsed >= self.first_align_zero_window and got_enough)
                    timeout_force = (elapsed >= self.first_align_freeze_max and len(self._first_align_buf) > 0)

                    if lock_now or timeout_force:
                        median_yaw = float(np.median(self._first_align_buf))
                        self._yaw_trim_epi = median_yaw
                        self.yaw_trim = median_yaw
                        self._first_align_done = True
                        self._first_align_active = False
                        reason = f"ALIGN_FIRST_ZERO_LOCK({median_yaw:.3f}rad)"

                    e_d = (self.target_dist - self.wall_dist) if (self.wall_dist==self.wall_dist) else 0.0
                    e_d_eff = 0.0 if abs(e_d) <= self.deadzone_d else e_d
                    vy = clamp(self.k_vy * e_d_eff, -self.max_vy, self.max_vy) * (+1.0 if self.side=="right" else -1.0)
                    vx = +0.03
                    wz = 0.0
                    self._publish(vx, vy, wz, "ALIGN_FIRST_ZEROING" if reason=="" else reason)
                    rate.sleep()
                    continue

                # Later ALIGN：短窗自零点（不冻结）
                if self.auto_yaw_zero and (self._az_t0 is not None) and ((now - self._az_t0) <= self.auto_yaw_window):
                    if self.has_wall and self.wall_dist==self.wall_dist and self.wall_yaw==self.wall_yaw:
                        if abs(self.target_dist - self.wall_dist) <= self.auto_yaw_dist_gate:
                            self._az_buf.append(self.wall_yaw)
                    if len(self._az_buf) >= self.auto_yaw_samples_min:
                        med = float(np.median(self._az_buf))
                        spread = (max(self._az_buf) - min(self._az_buf)) if self._az_buf else 1e9
                        if spread <= math.radians(self.auto_yaw_var_gate_deg):
                            self._yaw_trim_epi = med
                if self.auto_yaw_zero and (self._az_t0 is not None) and ((now - self._az_t0) > self.auto_yaw_window) and (self._yaw_trim_epi is None) and len(self._az_buf) > 0:
                    self._yaw_trim_epi = float(np.median(self._az_buf))

                if self.has_wall and (self.wall_dist==self.wall_dist) and (self.wall_yaw==self.wall_yaw):
                    e_d      = (self.target_dist - self.wall_dist)
                    yaw_bias = (self._yaw_trim_epi if (self._yaw_trim_epi is not None) else self.yaw_trim)
                    yaw_meas = self.wall_yaw - yaw_bias

                    e_d_eff = 0.0 if abs(e_d) <= self.deadzone_d else e_d
                    vy = clamp(self.k_vy * e_d_eff, -self.max_vy, self.max_vy) * (+1.0 if self.side=="right" else -1.0)

                    yaw_err = 0.0 if abs(yaw_meas) <= self.deadzone_y else -yaw_meas
                    if abs(e_d) > self.dist_freeze_on:
                        yaw_err *= self.dist_freeze_scale

                    wz = clamp(self.k_w * yaw_err, -self.align_w_max, self.align_w_max)
                    vx = +0.05

                    post_turn = (self._post_turn_until > now)
                    tol_y = (self.post_turn_align_tol_y if post_turn else self.align_tol_y)

                    if (abs(e_d) <= self.align_tol_d) and (abs(yaw_meas) <= tol_y):
                        self.state = "LOCKED_GO"
                        reason = "ALIGN->LOCKED_GO"
                        if self._yaw_trim_epi is not None:
                            alpha = 0.1
                            self.yaw_trim = (1.0 - alpha) * self.yaw_trim + alpha * self._yaw_trim_epi
                    else:
                        reason = "ALIGNING"
                else:
                    vx, vy, wz = +0.02, 0.0, 0.0
                    reason = "ALIGN_WAIT_WALL"

            elif self.state == "LOCKED_GO":
                if self.has_wall and (self.wall_dist==self.wall_dist) and (self.wall_yaw==self.wall_yaw):
                    e_d      = (self.target_dist - self.wall_dist)
                    yaw_bias = (self._yaw_trim_epi if (self._yaw_trim_epi is not None) else self.yaw_trim)
                    yaw_meas = self.wall_yaw - yaw_bias

                    e_yaw    = 0.0 if abs(yaw_meas) <= self.deadzone_y else -yaw_meas
                    e_d_eff  = 0.0 if abs(e_d) <= self.deadzone_d else e_d

                    if self.post_turn_force_align and (
                        (abs(yaw_meas) > self.locked_to_align_y) or (abs(e_d) > self.align_tol_d)
                    ):
                        self.state = "ALIGN"
                        reason = "LOCKED->ALIGN_POST_TURN"
                        vx, vy, wz = +0.02, 0.0, 0.0
                    else:
                        vy = clamp(0.9*self.k_vy * e_d_eff, -self.max_vy, self.max_vy) * ( +1.0 if self.side=="right" else -1.0 )
                        wz = clamp(0.8*self.k_w  * e_yaw,   -self.max_w,  self.max_w)
                        vx = self.forward_speed
                        reason = "LOCKED_GO"
                else:
                    self.state = "HOLD"; hold_loss_cnt = 0; reason = "LOCKED->HOLD"

            else:  # HOLD
                vx = self.forward_speed_hold; vy = 0.0; wz = 0.0
                reason = "HOLD_KEEP_STRAIGHT"
                got = (self.has_wall and self.wall_dist==self.wall_dist and self.wall_yaw==self.wall_yaw)
                if got:
                    if (self.mem_dist==self.mem_dist) and (self.mem_yaw==self.mem_yaw):
                        if abs(self.wall_dist-self.mem_dist) <= (self.align_tol_d+0.02) and abs(self.wall_yaw-self.mem_yaw) <= (self.align_tol_y+0.05):
                            self.state = "LOCKED_GO"; reason = "HOLD->LOCKED(mem match)"
                        else:
                            self.state = "ALIGN";     reason = "HOLD->ALIGN"
                    else:
                        self.state = "ALIGN";         reason = "HOLD->ALIGN"
                else:
                    hold_loss_cnt = min(self.loss_window, hold_loss_cnt+1)
                    if hold_loss_cnt >= self.loss_slowdown_thresh:
                        vx = 0.06; reason += "|slow"

            self._publish(vx, vy, wz, reason)
            rate.sleep()

    def __del__(self):
        try: self.csvf.close()
        except Exception: pass


if __name__ == "__main__":
    rospy.init_node("bug2_lock_v4_openloop")
    WallFollower().run()
