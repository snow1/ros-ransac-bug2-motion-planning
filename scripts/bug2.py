#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from enum import Enum
from std_msgs.msg import Float32, Bool


class BotState(Enum):
    GOAL_SEEK = 0
    WALL_FOLLOW = 1
    DONE = 2


# Odometry callback
def odom_data(data: Odometry):
    global bot_odom
    bot_odom = data
    global start_pos
    if not start_pos:
        start_pos = [data.pose.pose.position.x, data.pose.pose.position.y]
        rospy.loginfo(f"Start position initialized as: {start_pos}")


def obstacle_callback(data: Bool):
    global obstacle_ahead
    obstacle_ahead = data.data


# RANSAC yaw angle
def wall_angle_callback(data: Float32):
    global wall_yaw
    wall_yaw = data.data


def init():
    rospy.Subscriber('/odom', Odometry, odom_data)
    rospy.Subscriber("/wall_angle", Float32, wall_angle_callback)
    rospy.Subscriber("/obstacle_ahead", Bool, obstacle_callback)
    twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    yaw_thresh = 0.2
    follow_point = (0, 0)
    seek_point = (0, 0)
    state = BotState.GOAL_SEEK
    rospy.loginfo("Starting in Goal Seek Mode")
    global obstacle_ahead, wall_yaw, start_pos
    rate = rospy.Rate(10)
    # Main Loop
    while not rospy.is_shutdown():
        if not start_pos:
            continue

        bot_pos = bot_odom.pose.pose.position
        bot_yaw = euler_from_quaternion([bot_odom.pose.pose.orientation.x, bot_odom.pose.pose.orientation.y, bot_odom.pose.pose.orientation.z, bot_odom.pose.pose.orientation.w])[2]
        goal_dist = np.sqrt(np.square(bot_pos.x - goal_pos[0]) + np.square(bot_pos.y - goal_pos[1]))
        x1, y1 = start_pos
        x2, y2 = goal_pos
        x0, y0 = bot_pos.x, bot_pos.y
        line_dist = abs(((x2 - x1) * (y1 - y0)) - ((x1 - x0) * (y2 - y1))) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if state == BotState.WALL_FOLLOW:
            twist_msg = Twist()
            twist_msg.angular.z = 0
            twist_msg.linear.x = 0

            if wall_yaw == 0.0:
                twist_msg.angular.z = 0.5
                if not obstacle_ahead:
                    twist_msg.linear.x = 2
            elif wall_yaw > 0 + yaw_thresh:
                twist_msg.angular.z = + 3 * np.abs(bot_yaw - wall_yaw)
                twist_msg.linear.x = 1
            elif wall_yaw < 0 - yaw_thresh:
                twist_msg.angular.z = - 3 * np.abs(bot_yaw - wall_yaw)
                twist_msg.linear.x = 0.2

            if -yaw_thresh < wall_yaw < yaw_thresh and goal_dist >= 0.5:
                twist_msg.linear.x = 0.2
                if obstacle_ahead:
                    twist_msg.linear.x = 1  # Approach slowly

            twist_pub.publish(twist_msg)
            leave_point_dist = np.sqrt((follow_point[0] - x0) ** 2 + (follow_point[1] - y0) ** 2)
            if leave_point_dist >= 1 and line_dist <= 0.5:
                seek_point = (x0, y0)
                state = BotState.GOAL_SEEK
                rospy.loginfo("Initiating Goal Seek Mode")

        elif state == BotState.GOAL_SEEK:
            goal_yaw = np.arctan2(goal_pos[1] - bot_pos.y, goal_pos[0] - bot_pos.x)
            relative_yaw = np.rad2deg(np.arctan2(np.sin(goal_yaw - bot_yaw), np.cos(goal_yaw - bot_yaw)))
            twist_msg = Twist()
            twist_msg.angular.z = 0
            twist_msg.linear.x = 0

            if relative_yaw > 0 + yaw_thresh:
                twist_msg.angular.z = + 3 * np.abs(bot_yaw-goal_yaw)
            elif relative_yaw < 0 - yaw_thresh:
                twist_msg.angular.z = - 3 * np.abs(bot_yaw-goal_yaw)

            if -yaw_thresh < relative_yaw < yaw_thresh and goal_dist >= 0.5:
                twist_msg.linear.x = 2

            twist_pub.publish(twist_msg)

            seek_point_dist = np.sqrt((seek_point[0] - x0) ** 2 + (seek_point[1] - y0) ** 2)
            if obstacle_ahead and seek_point_dist >= 0.5:
                follow_point = (x0, y0)
                state = BotState.WALL_FOLLOW
                rospy.loginfo("Initiating Wall Follow Mode")

        if state != BotState.DONE and goal_dist <= 0.5:
            state = BotState.DONE
            rospy.loginfo("Goal Achieved!")
        rate.sleep()


if __name__ == '__main__':
    bot_odom = Odometry()
    start_pos = []
    obstacle_ahead = False
    wall_yaw = 0
    rospy.init_node('bug2', anonymous=True)
    goal_pos = rospy.get_param('~goal_pos')
    rospy.loginfo(f"Goal position initialized as: {goal_pos}")
    init()
