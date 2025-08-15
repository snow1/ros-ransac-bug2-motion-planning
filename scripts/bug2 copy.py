#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from enum import Enum
from std_msgs.msg import Float32, Bool

#GOAL_SEEK: drives directly toward a goal.
#WALL_FOLLOW: follows walls when an obstacle is encountered.
#DONE: stops when close to goal.
goal_marker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=1)
wall_marker_pub = rospy.Publisher('/wall_marker', Marker, queue_size=1)

def publish_goal_marker(x, y):
    marker = Marker()
    marker.header.frame_id = "odom"  # 或 "map"，要和你的全局frame对应
    marker.header.stamp = rospy.Time.now()
    marker.ns = "goal"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.25
    marker.scale.y = 0.25
    marker.scale.z = 0.25
    marker.color.r = 1.0
    marker.color.g = 0.2
    marker.color.b = 0.2
    marker.color.a = 1.0
    goal_marker_pub.publish(marker)

def publish_wall_line(x1, y1, x2, y2):
    marker = Marker()
    marker.header.frame_id = "odom"  # 要和实际frame一致
    marker.header.stamp = rospy.Time.now()
    marker.ns = "wall"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 0.2
    marker.color.g = 1.0
    marker.color.b = 0.2
    marker.color.a = 1.0
    marker.points = [Point(x1, y1, 0), Point(x2, y2, 0)]
    wall_marker_pub.publish(marker)
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
    publish_goal_marker(4.36, 9)

    rospy.loginfo(f"Goal position initialized as: {goal_pos}")
    init()
