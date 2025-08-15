#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import numpy as np
from std_msgs.msg import Float32, Bool

# Global variables
bot_odom = Odometry()
obstacle_ahead = False
wall_yaw = 0

def odom_data(data: Odometry):
    global bot_odom
    bot_odom = data

def obstacle_callback(data: Bool):
    global obstacle_ahead
    obstacle_ahead = data.data

def wall_angle_callback(data: Float32):
    global wall_yaw
    wall_yaw = data.data

def init():
    rospy.Subscriber('/odom', Odometry, odom_data)
    rospy.Subscriber("/wall_angle", Float32, wall_angle_callback)
    rospy.Subscriber("/obstacle_ahead", Bool, obstacle_callback)

    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    yaw_thresh = 0.2

    rospy.loginfo("🚀 Starting in WALL FOLLOW mode only")
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        bot_pos = bot_odom.pose.pose.position
        bot_yaw = euler_from_quaternion([
            bot_odom.pose.pose.orientation.x,
            bot_odom.pose.pose.orientation.y,
            bot_odom.pose.pose.orientation.z,
            bot_odom.pose.pose.orientation.w
        ])[2]

        twist_msg = Twist()

        # If no wall detected, rotate to search
        if wall_yaw == 0.0:
            twist_msg.angular.z = 0.5
            if not obstacle_ahead:
                twist_msg.linear.x = 0.2
        # Wall is to the left — turn right
        elif wall_yaw > 0 + yaw_thresh:
            twist_msg.angular.z = + 3 * np.abs(bot_yaw - wall_yaw)
            twist_msg.linear.x = 0.2
        # Wall is to the right — turn left
        elif wall_yaw < 0 - yaw_thresh:
            twist_msg.angular.z = - 3 * np.abs(bot_yaw - wall_yaw)
            twist_msg.linear.x = 0.2
        # Wall straight ahead — move forward
        if -yaw_thresh < wall_yaw < yaw_thresh:
            twist_msg.linear.x = 0.3
            if obstacle_ahead:
                twist_msg.linear.x = 0.15  # slow down if close

        twist_pub.publish(twist_msg)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('wall_follower', anonymous=True)
    init()
