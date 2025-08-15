#!/usr/bin/env python3
# Auto-calibrate base_link -> laser: x, y, yaw
# Usage: rosrun ransacbug2 calibrate_laser_extrinsics.py _scan_topic:=/scan _buffer_secs:=2.0 _max_points:=8000
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import math
from collections import deque
import sys

def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def scan_to_xy(scan):
    # Convert LaserScan to Nx2 XY points in lidar frame
    angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    r = np.asarray(scan.ranges, dtype=np.float64)
    mask = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.float64)
    angles = angles[mask]
    r = r[mask]
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return np.stack((x, y), axis=1)  # (N,2)

def fit_line_tls(pts):
    """
    Robust 2D total least squares line fit without large allocations.
    Returns centroid c (2,), unit direction u (2,), unit normal n (2,)
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 20:
        raise ValueError("Not enough valid 2D points for line fit")
    c = pts.mean(axis=0)
    centered = (pts - c).T  # (2, N)
    # SVD on 2xN is safe and tiny
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)  # U: 2x2
    direction = U[:, 0]
    direction /= np.linalg.norm(direction) + 1e-12
    normal = np.array([-direction[1], direction[0]])
    return c, direction, normal

def capture(buf, prompt, max_points):
    input(prompt)
    scans = list(buf)
    if not scans:
        raise RuntimeError("No scans buffered. Hold still ~2s before pressing ENTER.")
    # Concatenate points from buffered scans
    all_pts = []
    for s in scans:
        pts = scan_to_xy(s)
        if pts.size:
            all_pts.append(pts)
    if not all_pts:
        raise RuntimeError("No valid points in scans (all NaN/inf or out of range).")
    pts = np.vstack(all_pts)

    # Downsample to keep memory small
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    c, dirvec, normal = fit_line_tls(pts)
    # Signed distance of centroid to the line normal (for reference)
    d = float(normal.dot(c))
    normal_angle = math.atan2(normal[1], normal[0])
    return {
        "centroid": c, "dir": dirvec, "normal": normal,
        "d_lidar": d, "normal_angle_lidar": normal_angle
    }

def main():
    rospy.init_node("calibrate_laser_extrinsics", anonymous=True)
    scan_topic  = rospy.get_param("~scan_topic", "/scan")
    buffer_secs = float(rospy.get_param("~buffer_secs", 2.0))
    max_points  = int(rospy.get_param("~max_points", 8000))
    z_guess     = float(rospy.get_param("~z_guess", 0.85))  # edit or param

    rate_hz = 20
    buf = deque(maxlen=int(rate_hz * buffer_secs))
    last_scan = [None]

    def cb(msg):
        last_scan[0] = msg
        buf.append(msg)

    sub = rospy.Subscriber(scan_topic, LaserScan, cb, queue_size=1)

    r = rospy.Rate(rate_hz)
    print("\n=== Laser Extrinsics Calibration (x, y, yaw) ===")
    print("Setup:")
    print("  - Axes: x forward, y left, base_link at robot body origin.")
    print("  - Do 3 captures while holding the robot still ~2 seconds each.")
    print("Captures:")
    print("  1) Parallel to a LEFT wall (wall on robot's LEFT side).")
    print("  2) Parallel to a RIGHT wall (wall on robot's RIGHT side).")
    print("  3) Facing a wall straight AHEAD.")
    print("\nTip: Keep ~0.5–2.0 m from the wall for clean lines.\n")

    while not rospy.is_shutdown() and last_scan[0] is None:
        r.sleep()

    try:
        cap_left  = capture(buf,  "[ENTER] when ready for LEFT-wall capture: ", max_points)
        cap_right = capture(buf,  "[ENTER] when ready for RIGHT-wall capture: ", max_points)
        cap_front = capture(buf,  "[ENTER] when ready for FRONT-wall capture: ", max_points)
    except Exception as e:
        print("Capture error:", e)
        sys.exit(1)

    # Build lidar-frame normals
    nL_lidar = np.array([math.cos(cap_left["normal_angle_lidar"]),
                         math.sin(cap_left["normal_angle_lidar"])])
    nR_lidar = np.array([math.cos(cap_right["normal_angle_lidar"]),
                         math.sin(cap_right["normal_angle_lidar"])])
    nF_lidar = np.array([math.cos(cap_front["normal_angle_lidar"]),
                         math.sin(cap_front["normal_angle_lidar"])])

    # Target normals in base frame
    nL_base = np.array([0.0, -1.0])  # left wall normal points -y_base
    nR_base = np.array([0.0,  1.0])  # right wall normal points +y_base
    nF_base = np.array([-1.0, 0.0])  # front wall normal points -x_base

    def ang(v): return math.atan2(v[1], v[0])

    psi_L = wrap(ang(nL_base) - ang(nL_lidar))
    psi_R = wrap(ang(nR_base) - ang(nR_lidar))
    psi_F = wrap(ang(nF_base) - ang(nF_lidar))

    # Robust circular mean of yaw
    psi = math.atan2(
        np.mean(np.sin([psi_L, psi_R, psi_F])),
        np.mean(np.cos([psi_L, psi_R, psi_F]))
    )

    # Translation least-squares
    R = np.array([[math.cos(psi), -math.sin(psi)],
                  [math.sin(psi),  math.cos(psi)]])
    normals_base = [nL_base, nR_base, nF_base]
    cents_lidar  = [cap_left["centroid"], cap_right["centroid"], cap_front["centroid"]]

    A = []
    b = []
    for nk, ck in zip(normals_base, cents_lidar):
        proj = float(nk.dot(R.dot(ck)))
        A.append(nk)
        b.append(-proj)
    A = np.vstack(A)
    b = np.array(b).reshape(-1,)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    tx, ty = float(t[0]), float(t[1])

    yaw_deg = math.degrees(psi)
    qz = math.sin(psi/2.0)
    qw = math.cos(psi/2.0)

    print("\n=== RESULT (base_link → laser) ===")
    print("x    = {:.4f} m".format(tx))
    print("y    = {:.4f} m".format(ty))
    print("z    = {:.4f} m  (set via param ~z_guess or measure)".format(z_guess))
    print("yaw  = {:.4f} rad  ({:.2f}°)".format(psi, yaw_deg))
    print("roll = 0.0000 rad   pitch = 0.0000 rad   (2D LiDAR assumption)")
    print("\nROS1 tf static_transform_publisher (yaw pitch roll in radians):")
    print('  <node pkg="tf" type="static_transform_publisher" name="base_to_laser"')
    print('        args="{:.4f} {:.4f} {:.4f}  {:.4f} 0 0  base_link laser 50"/>'.format(tx, ty, z_guess, psi))
    print("\nIf you use tf2_ros (quaternion qx qy qz qw): qx=0 qy=0 qz={:.5f} qw={:.5f}".format(qz, qw))
    print("  rosrun tf2_ros static_transform_publisher {:.4f} {:.4f} {:.4f}  0 0 {:.5f} base_link laser".format(tx, ty, z_guess, psi))

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
