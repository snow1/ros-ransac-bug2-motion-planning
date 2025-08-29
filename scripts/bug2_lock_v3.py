#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =======================
# File: bug2_lock_v4.py
# Node: bug2_lock_v4
# Description:
#主控节点，喷到边 → 前探找空地 → 侧移让位 → 预停 → 原地 90° → 冷却）
#   ✅ 贴墙喷涂 + 角点“喷到边→找空地→让位→再转弯”流程（大车转弯不剐蹭）
#   ✅ 仅保留 1 套角点 FSM（互斥控制），彻底避免“对齐 vs 转弯”打架
#   ✅ 终端打印清晰的状态迁移与角点阶段日志
#   ✅ CSV 详单日志（含前/左最小距离、位姿、访问记忆分数、角点阶段）
#   ✅ 轻量“访问记忆”visited grid（软约束，不回头）
#   ✅ 代码内中文解释，所有关键参数可在 launch 覆盖
#
# 关键思路：
#   - 你的车体：长 L=1.2 m，宽 W=1.0 m。原地 90° 转弯需要的“包络圆半径” R_turn ≈ 0.5*sqrt(L^2+W^2)。
#     对你是 R_turn≈0.781 m。再加安全余量（建议 ~0.15 m）→ R_need≈0.93 m。
#     这意味着：在“贴右墙=0.6 m 侧距”处直接原地转，会剐蹭！
#   - 因此角点流程改为：
#       SPRAY_EDGE（喷到边）→ STAGE_OPEN（前探找空地）→ STAGE_STEP_OUT（向左让位至 ≥R_need）
#       → APPROACH_STOP（预停多帧0速）→ TURNING（原地 90°）→ COOLDOWN → ALIGN/LOCKED_GO
#   - 过程中用 /front_min 与 /left_min 校验可用空间；让位采用里程计 y 位移闭环，不依赖墙测距。
#
# 依赖：rospy, tf2_ros, tf_conversions
# 输入话题：
#   /wall_angle (Float32 rad)  /wall_lateral (Float32 m) /wall_has (Bool) /wall_quality (Float32)
#   /corner_has (Bool) /corner_point (geometry_msgs/PointStamped)
#   /front_min (Float32) /left_min (Float32)   # 来自 perception.py 的扇区最小距离
#   /obstacle_ahead (Bool)  # 仍保留
# 输出：/cmd_vel (geometry_msgs/Twist)
# 坐标系：REP-103 标准  base_link：+X前  +Y左  +Z上
# =======================

import time, csv, math
import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from std_msgs.msg import Bool, Float32
from math import pi, isnan, hypot

# ---------- 小工具 ----------
def ang_wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def sat(x, lo, hi):
    return max(lo, min(hi, x))

class VisitedGrid:
    """轻量“走过即记”网格，用于日志与小幅引导（HOLD 期减少回头）。"""
    def __init__(self, res=0.25, mark_step=0.25, ahead_cells=8, decay=0.0):
        self.res = float(res)
        self.mark_step = float(mark_step)
        self.ahead_cells = int(ahead_cells)
        self.decay = float(decay)
        self.grid = {}
        self.last_mark_pose = None
    def _key(self, x, y):
        r = self.res
        return (int(np.floor(x/r)), int(np.floor(y/r)))
    def mark_if_far(self, x, y):
        if self.last_mark_pose is None:
            self.last_mark_pose = (x, y)
            self.grid[self._key(x, y)] = self.grid.get(self._key(x, y), 0) + 1
            return True
        dx = x - self.last_mark_pose[0]
        dy = y - self.last_mark_pose[1]
        if (dx*dx + dy*dy) >= (self.mark_step*self.mark_step):
            self.last_mark_pose = (x, y)
            self.grid[self._key(x, y)] = self.grid.get(self._key(x, y), 0) + 1
            if self.decay > 0:
                for k in list(self.grid.keys()):
                    self.grid[k] *= (1.0 - self.decay)
                    if self.grid[k] < 0.25:
                        self.grid.pop(k, None)
            return True
        return False
    def score_ahead(self, x, y, yaw):
        s = 0.0
        c, sY = np.cos(yaw), np.sin(yaw)
        for i in range(1, self.ahead_cells+1):
            px = x + i*self.res*c
            py = y + i*self.res*sY
            s += self.grid.get(self._key(px, py), 0)
        return float(s)

class WallFollower:
    def __init__(self):
        # ====== 基础参数（可在 launch 覆盖）======
        self.side = rospy.get_param("~side", "right")  # right/left  → 决定转向与侧向正方向
        self.cmd_vy_sign = rospy.get_param("~cmd_vy_sign", 1.0)  # +1 标准：+Y向左；若物理接线反，设 -1
        self.cmd_vx_sign = rospy.get_param("~cmd_vx_sign", 1.0)  # 若底盘接线/固件把X反了，就设为 -1.0


        # 目标与对齐
        self.target_dist = rospy.get_param("~target_dist", 0.60)
        self.align_tol_d = rospy.get_param("~align_tol_d", 0.03)
        self.align_tol_y = rospy.get_param("~align_tol_y", 0.03)
        self.k_vy        = rospy.get_param("~k_vy", 1.2)
        self.k_w         = rospy.get_param("~k_w", 1.5)
        self.max_vy      = rospy.get_param("~max_vy", 0.25)
        self.max_w       = rospy.get_param("~max_w", 0.50)

        # 前进速度与保持
        self.forward_speed      = rospy.get_param("~forward_speed", 0.12)
        self.forward_speed_hold = rospy.get_param("~forward_speed_hold", 0.08)
        self.min_forward_speed  = rospy.get_param("~min_forward_speed", 0.06)

        # GO 相位的小幅纠偏
        self.k_vy_go     = rospy.get_param("~k_vy_go", 0.8)
        self.k_w_go      = rospy.get_param("~k_w_go", 0.8)
        self.max_vy_go   = rospy.get_param("~max_vy_go", 0.20)
        self.max_w_go    = rospy.get_param("~max_w_go", 0.35)
        self.deadzone_d   = rospy.get_param("~deadzone_d", 0.01)
        self.deadzone_yaw = rospy.get_param("~deadzone_yaw", 0.01)

        # 安全 / 丢失
        self.front_stop = rospy.get_param("~front_stop", 0.40)
        self.loss_timeout_stop = rospy.get_param("~loss_timeout_stop", 2.5)
        self.loss_grace  = rospy.get_param("~loss_grace", 0.8)
        self.reacq_tol_d = rospy.get_param("~reacq_tol_d", 0.08)
        self.reacq_tol_y = rospy.get_param("~reacq_tol_y", 0.10)
        self.loss_slowdown_window = rospy.get_param("~loss_slowdown_window", 10.0)
        self.loss_slowdown_thresh = rospy.get_param("~loss_slowdown_thresh", 3)

        # 底盘限幅 + 软启动
        self.limit_vx = rospy.get_param("~limit_vx", 0.22)
        self.limit_vy = rospy.get_param("~limit_vy", 0.22)
        self.limit_w  = rospy.get_param("~limit_w",  0.65)
        self.soft_enable = rospy.get_param("~soft_enable", True)
        self.soft_up   = rospy.get_param("~soft_up",   0.04)
        self.soft_down = rospy.get_param("~soft_down", 0.06)
        self._soft_gate = 0.0

        # === 车体尺寸 & 转弯需求 ===
        self.robot_L = rospy.get_param("~robot_length", 0.20)
        self.robot_W = rospy.get_param("~robot_width",  0.20)
        self.turn_padding = rospy.get_param("~turn_padding", 0.15)  # 包络圆安全余量
        self.R_turn_need = 0.5*math.hypot(self.robot_L, self.robot_W) + self.turn_padding
        rospy.loginfo("[TURN] R_turn_need=%.3fm (L=%.2f W=%.2f pad=%.2f)",
                      self.R_turn_need, self.robot_L, self.robot_W, self.turn_padding)

        # === 喷到边 & 转弯前“找空地→让位”策略 ===
        self.spray_front_reach = rospy.get_param("~spray_front_reach", 0.35)  # 车体中心前方可喷到的极限（需按实测改）
        self.corner_pass_x     = rospy.get_param("~corner_pass_x", 0.30)     # 角点相对基座 x< -这个值，视为“已通过拐角”
        self.stage_vx = rospy.get_param("~stage_vx", 0.15)   # 前探速度
        self.stage_vy = rospy.get_param("~stage_vy", 0.15)   # 让位侧移速度
        self.stage_timeout = rospy.get_param("~stage_timeout", 6.0)
        self.side_extra = rospy.get_param("~turn_side_extra", 0.05)  # 侧向再多预留一点
        self.left_safe_extra = rospy.get_param("~left_safe_extra", 0.30)  # 左侧可用最小空间冗余（避免侧移撞左）

        # ==== 极简角点 FSM 额外参数（新增） ====
        self.c_forward_need_extra = rospy.get_param("~corner_forward_need_extra", 0.05)  # 前向比 R_need 多预留
        self.c_side_need_extra    = rospy.get_param("~corner_side_need_extra",    0.05)  # 对侧比 R_need 多预留
        self.c_front_safe_floor   = rospy.get_param("~front_safe_floor",          0.40)  # 前向硬安全地板
        self.c_step_vy            = rospy.get_param("~corner_step_vy",            0.18)  # 侧移速度 m/s
        self.c_back_vx            = rospy.get_param("~corner_back_vx",            0.12)  # 后退速度 m/s
        self.c_max_step_out       = rospy.get_param("~corner_max_step_out",       0.30)  # 最大侧移距离（可选）
        self.c_use_odom_y         = rospy.get_param("~corner_use_odom_y",         True) # 若要用里程计y限幅

        # 把角点状态重置为极简版本的初值
        self.corner_state   = "C_IDLE"     # C_IDLE / C_STEP_OUT / C_TURN / C_COOL
        self._c_t_enter     = 0.0
        self._c_yaw_start   = None
        self._c_step_y0     = None


        # === 角点 FSM 参数 ===
        self.corner_trigger_dist = rospy.get_param("~corner_trigger_dist", 1.20)
        self.corner_reset_dist   = rospy.get_param("~corner_reset_dist",   1.30)
        self.corner_turn_speed   = rospy.get_param("~corner_turn_speed",   1.0)  # rad/s 0.6
        self.corner_turn_angle   = rospy.get_param("~corner_turn_angle",   pi/2)
        self.corner_cooldown     = rospy.get_param("~corner_cooldown",     4.0)
        self.corner_forward_only = rospy.get_param("~corner_forward_only", True)
        self.corner_prepare_t    = rospy.get_param("~corner_prepare_t",    0.30)
        self.corner_brake_times  = rospy.get_param("~corner_brake_times",  3)
        self.corner_trigger_in_states = rospy.get_param("~corner_trigger_in_states", ["LOCKED_GO","HOLD"])  # 允许在哪些主态触发
        self.corner_arm_samples = rospy.get_param("~corner_arm_samples", 3)
        self.corner_trend_eps   = rospy.get_param("~corner_trend_eps", 0.015)
        self.corner_min_vx_to_trigger = rospy.get_param("~corner_min_vx_to_trigger", 0.05)
        self.turn_yaw_tol_deg = rospy.get_param("~turn_yaw_tol_deg", 6.0)
        self.turn_hold_frames = rospy.get_param("~turn_hold_frames", 3)
        self.turn_exit_extra_m= rospy.get_param("~turn_exit_extra_m", 0.30)
        self.turn_max_time    = rospy.get_param("~turn_max_time", 13.0) # turns timing, it is 6 in the begining

        # === 访问记忆（可关闭） ===
        self.visited_enable   = rospy.get_param("~visited_enable", True)
        self.vis_res          = rospy.get_param("~visited_res", 0.25)
        self.vis_mark_step    = rospy.get_param("~visited_mark_step", 0.25)
        self.vis_ahead_cells  = rospy.get_param("~visited_ahead_cells", 8)
        self.vis_decay        = rospy.get_param("~visited_decay", 0.0)
        self.vis_repulse_gain = rospy.get_param("~visited_repulse_gain", 0.05)
        self.vis = VisitedGrid(self.vis_res, self.vis_mark_step, self.vis_ahead_cells, self.vis_decay)

        # === IO & 日志 ===
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")
        self.log_path = rospy.get_param("~log_path", f"/tmp/wf_lock_v4_{int(time.time())}.csv")
        self.pub_cmd = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        rospy.Subscriber("/wall_lateral",   Float32, self.cb_wall_dist,   queue_size=1)
        rospy.Subscriber("/wall_angle",     Float32, self.cb_wall_yaw,    queue_size=1)
        rospy.Subscriber("/wall_has",       Bool,    self.cb_wall_has,    queue_size=1)
        rospy.Subscriber("/obstacle_ahead", Bool,    self.cb_obs_front,   queue_size=1)
        rospy.Subscriber("/wall_quality",   Float32, self.cb_wall_quality,queue_size=1)
        rospy.Subscriber("/corner_has",     Bool,    self.cb_corner_has,  queue_size=1)
        rospy.Subscriber("/corner_point",   PointStamped, self.cb_corner_pt, queue_size=1)
        rospy.Subscriber("/front_min",      Float32, self.cb_front_min,   queue_size=1)
        rospy.Subscriber("/left_min",       Float32, self.cb_left_min,    queue_size=1)
        self.right_min = float('nan')
        rospy.Subscriber("/right_min", Float32, lambda m: setattr(self, "right_min", float(m.data)), queue_size=1)


        self.csvf = open(self.log_path, "w", newline="")
        self.csv = csv.writer(self.csvf)
        self.csv.writerow([
            "time","state","reason",
            "has_wall","obs_front","quality",
            "wall_dist","wall_yaw(rad)",
            "front_min","left_min",
            "mem_dist","mem_yaw","mem_age",
            "corner_state","corner_has","corner_dist",
            "odom_x","odom_y","odom_yaw","vis_score_ahead",
            "vx_cmd","vy_cmd","w_cmd",
            "vx_limited","vy_limited","w_limited",
            "soft_gate"
        ])
        rospy.loginfo("bug2_lock_v4 started. log -> %s", self.log_path)

        # === 运行态 ===
        self.state = "ALIGN"
        self.has_wall = 0; self.obs_front = 0; self.quality = "nan"
        self.wall_dist = float('nan'); self.wall_yaw = float('nan')
        self.front_min = float('nan'); self.left_min = float('nan')
        self.mem_dist  = float('nan'); self.mem_yaw  = float('nan'); self.mem_time = None

        # 角点相关运行态
        self.corner_has = False; self.corner_pt = None; self.corner_dist = float('nan')
        #self.corner_state = "IDLE"  # IDLE / SPRAY_EDGE / STAGE_OPEN / STAGE_STEP_OUT / APPROACH_STOP / TURNING / COOLDOWN
        self.corner_prepare_end = 0.0
        self.corner_turn_end_time = 0.0
        self.corner_cool_end_time = 0.0
        self.corner_armed = False
        self.corner_arm_cnt = 0
        self.corner_prev_dist = None
        self.turn_yaw_start = None
        self.turn_yaw_target= None
        self.turn_hold_cnt  = 0
        self._yaw_buf = []
        self._yaw_buf_len = rospy.get_param("~turn_yaw_buf_len", 5)
        self.current_vx = 0.0
        self.stage_start_t = 0.0
        self.stage_goal_dy = 0.0
        self.stage_start_y = None

        # 统计
        self.loss_events = []

        # TF（用于访问记忆与侧移里程计闭环）
        self.tf_buf = tf2_ros.Buffer(); self.tf_lst = tf2_ros.TransformListener(self.tf_buf)
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")



    # ---------- 订阅回调 ----------
    def cb_wall_yaw(self, msg):
        self.wall_yaw = float(msg.data)
        self._yaw_buf.append(self.wall_yaw)
        if len(self._yaw_buf) > self._yaw_buf_len:
            self._yaw_buf.pop(0)
        self.last_perc_t = rospy.Time.now().to_sec()
    def cb_wall_dist(self, msg):
        self.wall_dist = float(msg.data)
        self.last_perc_t = rospy.Time.now().to_sec()
    def cb_wall_has(self, msg):
        self.has_wall = 1 if msg.data else 0
    def cb_obs_front(self, msg):
        self.obs_front = 1 if msg.data else 0
    def cb_wall_quality(self, msg):
        self.quality = f"{msg.data:.1f}"
    def cb_corner_has(self, msg):
        self.corner_has = bool(msg.data)
    def cb_corner_pt(self, msg: PointStamped):
        x = msg.point.x; y = msg.point.y
        self.corner_pt = (x, y); self.corner_dist = hypot(x, y)
        self.corner_last_stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
    def cb_front_min(self, msg):
        self.front_min = float(msg.data)
    def cb_left_min(self, msg):
        self.left_min = float(msg.data)
    def _is_wall_following_simple(self):
        """
        简单判断当前确实在“贴墙走”（用于避免误触发拐角）：
        有墙、侧距可用，且 |侧距-目标| <= 容差
        """
        if (self.has_wall) and (self.wall_dist == self.wall_dist):
            return abs(self.wall_dist - self.target_dist) <= (self.align_tol_d + 0.02)
        return False

    # ---------- 状态管理 ----------
    def set_state(self, new_state, reason=""):
        if new_state != self.state:
            rospy.loginfo("[STATE] %s -> %s | %s | d=%.3f yaw=%.1fdeg corner=%.2fm",
                          self.state, new_state, reason,
                          self.wall_dist if self.wall_dist==self.wall_dist else float('nan'),
                          (self.wall_yaw*180.0/pi) if self.wall_yaw==self.wall_yaw else float('nan'),
                          self.corner_dist if self.corner_dist==self.corner_dist else float('nan'))
            self.state = new_state

    # ---------- 发布统一出口（限幅+软门+CSV） ----------
    def _publish(self, vx, vy, wz, reason=""):
        vx_l = sat(vx, -self.limit_vx, self.limit_vx)
        vy_l = sat(vy, -self.limit_vy, self.limit_vy)
        wz_l = sat(wz, -self.limit_w,  self.limit_w)
        mag = max(abs(vx_l), abs(vy_l), abs(wz_l))
        if self.soft_enable:
            self._soft_gate = min(1.0, self._soft_gate + self.soft_up) if mag > 1e-6 \
                              else max(0.0, self._soft_gate - self.soft_down)
        else:
            self._soft_gate = 1.0 if mag > 1e-6 else 0.0
        vx_out = self.cmd_vx_sign * vx_l * self._soft_gate
        vy_out = self.cmd_vy_sign * vy_l * self._soft_gate
        wz_out = wz_l * self._soft_gate
        cmd = Twist(); cmd.linear.x=vx_out; cmd.linear.y=vy_out; cmd.angular.z=wz_out
        self.pub_cmd.publish(cmd)
        self.current_vx = vx_out
        ox, oy, oyaw = self._get_pose_odom()
        vis_score = float('nan')
        if self.visited_enable and (not math.isnan(ox)):
            self.vis.mark_if_far(ox, oy)
            vis_score = self.vis.score_ahead(ox, oy, oyaw)
        now = rospy.Time.now().to_sec()
        mem_age = (now - self.mem_time) if self.mem_time else float('nan')
        self.csv.writerow([
            now, self.state, reason,
            self.has_wall, self.obs_front, self.quality,
            self.wall_dist, self.wall_yaw,
            self.front_min, self.left_min,
            self.mem_dist, self.mem_yaw, mem_age,
            self.corner_state, int(self.corner_has), self.corner_dist,
            ox, oy, oyaw, vis_score,
            vx, vy, wz,
            vx_l, vy_l, wz_l,
            self._soft_gate
        ])

    def stop(self, reason="STOP"):
        self._publish(0.0, 0.0, 0.0, reason)

    def losses_in_window(self, now):
        self.loss_events = [t for t in self.loss_events if now - t <= self.loss_slowdown_window]
        return len(self.loss_events)

    # ---------- TF: 取 odom→base_link 位姿 ----------
    def _get_pose_odom(self):
        try:
            tr: TransformStamped = self.tf_buf.lookup_transform(self.odom_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.05))
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            q = tr.transform.rotation
            yaw = tf_conversions.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
            return x, y, yaw
        except Exception:
            return float('nan'), float('nan'), float('nan')

    # ---------- 角点子状态机（唯一拐角逻辑） ----------
    def corner_fsm(self):
        """
        极简角点 FSM：只用 front_min + 对侧最小距离（右墙时=left_min）+ 车体包络 R_turn_need
        四阶段：C_IDLE → C_STEP_OUT → C_TURN → C_COOL
        逻辑只保证“贴右墙”完备（side='right'）；贴左墙需要 /right_min 话题或自行对称实现。
        返回值：True=由拐角流程接管控制（本周期跳过主状态机）；False=不接管。
        """
        now = rospy.Time.now().to_sec()
        side = self.side  # 'right' / 'left'

        # ---- 取需要的传感量 ----
        front = self.front_min
        # 右墙场景下，对侧距离=left_min；左墙场景需要 right_min（本版本未订阅）
        opp_min = self.left_min if self.side == "right" else self.right_min


        # ---- 需求空间 ----
        need_side  = self.R_turn_need + self.c_side_need_extra
        need_front = self.R_turn_need + self.c_forward_need_extra

        # ---- 取当前 yaw ----
        _, _, yaw = self._get_pose_odom()

        # ========== C_IDLE：检测是否接近角点 ==========
        if self.corner_state == "C_IDLE":
            if (front == front) and (front < self.corner_trigger_dist) and self._is_wall_following_simple():
                self.corner_state = "C_STEP_OUT"
                self._c_t_enter   = now
                self._c_yaw_start = yaw
                self._c_step_y0   = None
                self.stop("C_IDLE->C_STEP_OUT")  # 预停一帧
                return True
            return False

        # ========== C_STEP_OUT：向对侧侧移 + 必要时小幅后退 ==========
        if self.corner_state == "C_STEP_OUT":
            # 满足条件：对侧≥need_side 且 前方≥need_front → 进入转弯
            ok_side  = (opp_min == opp_min) and (opp_min >= need_side)
            ok_front = (front   == front)   and (front   >= need_front)
            if ok_side and ok_front:
                self.corner_state = "C_TURN"
                self._c_t_enter   = now
                self._c_yaw_start = yaw
                self.stop("STEP_OUT->TURN")
                return True

            # 前方过近 → 小幅后退（硬地板 or 比R_turn_need略小）
            vx = 0.0
            if (front == front) and (front < max(self.c_front_safe_floor, self.R_turn_need - 0.10)):
                vx = -self.c_back_vx

            # 侧移到对侧：右墙→+Y（opp_sign=+1），左墙→-Y（本版未对称实现 right_min）
            opp_sign = +1.0 if side == "right" else -1.0
            vy = opp_sign * self.c_step_vy
            wz = 0.0

            if (now - self._c_t_enter) > 3.0:
                # Try turning anyway, or fall back to HOLD
                self.corner_state = "C_TURN"
                self._c_t_enter   = now
                self._c_yaw_start = yaw
                self.stop("STEP_OUT_TIMEOUT->TURN")
                return True

            # 可选：用里程计y限幅，防止侧移过头
            if self.c_use_odom_y:
                ox, oy, _ = self._get_pose_odom()
                if (oy == oy):
                    if self._c_step_y0 is None:
                        self._c_step_y0 = oy
                    if abs(oy - self._c_step_y0) >= self.c_max_step_out:
                        # 侧移到上限仍未满足 → 直接尝试进入转弯（也可改成放弃）
                        self.corner_state = "C_TURN"
                        self._c_t_enter   = now
                        self._c_yaw_start = yaw
                        self.stop("STEP_OUT_LIMIT->TURN")
                        return True

            self._publish(vx, vy, wz, f"STEP_OUT(opp {opp_min:.2f}/{need_side:.2f}, front {front:.2f}/{need_front:.2f})")
            return True

        # ========== C_TURN：原地 90° ==========
        if self.corner_state == "C_TURN":
            # 右墙→右转（顺时针，负角速度），左墙→左转（逆时针，正角速度）
            target = -np.pi/2 if side == "right" else +np.pi/2
            # 以“相对进入时的Δyaw”为控制量
            def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi
            d_yaw = wrap(yaw - (self._c_yaw_start if self._c_yaw_start is not None else yaw))
            err   = wrap(d_yaw - target)

            # 结束条件：角度到位 or 兜底超时
            if abs(err) <= (self.turn_yaw_tol_deg*np.pi/180.0) or (now - self._c_t_enter) >= self.turn_max_time:
                self.corner_state = "C_COOL"
                self._c_t_enter   = now
                self.stop("TURN->COOL")
                return True

            wz = (-1.0 if side == "right" else +1.0) * self.corner_turn_speed
            self._publish(0.0, 0.0, wz, f"TURN({np.rad2deg(err):.1f}deg)")
            return True

        # ========== C_COOL：冷却 ==========
        if self.corner_state == "C_COOL":
            if (now - self._c_t_enter) >= self.corner_cooldown:
                self.corner_state = "C_IDLE"
                self.stop("COOL->IDLE")
                return False  # 冷却完毕，释放控制
            else:
                self.stop("COOL")
                return True

        # 兜底复位
        self.corner_state = "C_IDLE"
        return False

    # ---------- 主循环 ----------
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            vx = vy = wz = 0.0; reason = ""

            # 角点 FSM：若需要转弯/喷边/前探/让位，则拦截一切（避免与 ALIGN/GO 打架）
            if self.corner_fsm():
                rate.sleep(); continue

            # 可选安全：前障直接减速/停车
            #if self.obs_front:
            #    self.stop("SAFE_STOP_OBS_FRONT"); rate.sleep(); continue

            # 主状态机
            if self.state == "ALIGN":
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    self.mem_dist = self.wall_dist; self.mem_yaw = self.wall_yaw; self.mem_time = now
                    e_d   = self.target_dist - self.wall_dist
                    e_yaw = -self.wall_yaw
                    side_dir = +1.0 if (self.side=="right") else -1.0
                    vy = side_dir * sat(self.k_vy * e_d, -self.max_vy, self.max_vy)
                    wz = sat(self.k_w * e_yaw, -self.max_w, self.max_w)
                    vx = 0.05
                    reason = "ALIGNING"
                    if abs(e_d) <= self.align_tol_d and abs(self.wall_yaw) <= self.align_tol_y:
                        self.set_state("LOCKED_GO", "ALIGN_DONE"); self.stop("ALIGN_DONE"); rate.sleep(); continue
                else:
                    self.set_state("HOLD", "ALIGN_LOSS→HOLD"); self.loss_events.append(now)
                    vx = self.forward_speed_hold; reason = "ALIGN_LOSS→HOLD"

            elif self.state == "LOCKED_GO":
                vx = self.forward_speed
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw):
                    e_d = self.target_dist - self.wall_dist
                    e_d_eff = 0.0 if abs(e_d) < self.deadzone_d else e_d
                    side_dir = +1.0 if (self.side=="right") else -1.0
                    vy = sat(side_dir * self.k_vy_go * e_d_eff, -self.max_vy_go, self.max_vy_go)
                    e_yaw = -self.wall_yaw
                    e_yaw_eff = 0.0 if abs(e_yaw) < self.deadzone_yaw else e_yaw
                    wz = sat(self.k_w_go * e_yaw_eff, -self.max_w_go, self.max_w_go)
                    reason = "LOCKED_GO(vx+vy+wz)"
                    self.mem_dist = self.wall_dist; self.mem_yaw = self.wall_yaw; self.mem_time = now
                else:
                    self.set_state("HOLD", "LOCKED→HOLD"); self.loss_events.append(now)
                    self.stop("LOCKED→HOLD"); rate.sleep(); continue

            elif self.state == "HOLD":
                vx = self.forward_speed_hold; reason = "HOLD_KEEP_STRAIGHT"
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw) and (not self.mem_time):
                    self.mem_dist = self.wall_dist; self.mem_yaw = self.wall_yaw; self.mem_time = now
                    self.set_state("ALIGN", "HOLD→ALIGN_INIT_REACQ"); self.stop("HOLD→ALIGN_INIT_REACQ"); rate.sleep(); continue
                if self.has_wall and not isnan(self.wall_dist) and not isnan(self.wall_yaw) and self.mem_time:
                    if abs(self.wall_dist - self.mem_dist) <= self.reacq_tol_d and abs(self.wall_yaw - self.mem_yaw) <= self.reacq_tol_y:
                        self.set_state("LOCKED_GO", "REACQ→LOCKED"); self.stop("REACQ→LOCKED"); rate.sleep(); continue
                    else:
                        self.mem_dist = self.wall_dist; self.mem_yaw = self.wall_yaw; self.mem_time = now
                        reason = "HOLD_MEM_UPDATE"
                nloss = self.losses_in_window(now)
                if nloss >= self.loss_slowdown_thresh and self.forward_speed > self.min_forward_speed:
                    self.forward_speed      = max(self.min_forward_speed, self.forward_speed      - 0.02)
                    self.forward_speed_hold = max(self.min_forward_speed, self.forward_speed_hold - 0.02)
                    reason = f"HOLD_SLOWDOWN(n={nloss})"
                if self.visited_enable:
                    ox, oy, oyaw = self._get_pose_odom()
                    if not math.isnan(ox):
                        s_vis = self.vis.score_ahead(ox, oy, oyaw)
                        if s_vis > 0.0:
                            vy += self.vis_repulse_gain
                            reason += f"|VIS_AVOID({s_vis:.1f})"

            elif self.state == "FAIL_STOP":
                self.stop("STOP_STATE"); rate.sleep(); continue

            self._publish(vx, vy, wz, reason)
            rate.sleep()

    def __del__(self):
        try:
            self.csvf.close()
        except Exception:
            pass

if __name__ == "__main__":
    rospy.init_node("bug2_lock_v4")
    WallFollower().run()


