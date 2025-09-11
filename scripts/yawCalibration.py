#!/usr/bin/env python3
import rospy, math
from sensor_msgs.msg import LaserScan
import numpy as np

def cb(msg):
    ranges = np.array(msg.ranges, dtype=float)
    # 只看有效距离
    valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
    if not np.any(valid): 
        return
    i = np.argmin(ranges[valid])
    idxs = np.nonzero(valid)[0]
    k = idxs[i]
    angle = msg.angle_min + k * msg.angle_increment
    print("min range: %.2fm @ %.1f°" % (ranges[k], math.degrees(angle)))

if __name__ == "__main__":
    rospy.init_node("scan_heading_test")
    rospy.Subscriber("/scan", LaserScan, cb, queue_size=1)
    rospy.spin()
