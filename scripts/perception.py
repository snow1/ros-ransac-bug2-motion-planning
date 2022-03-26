#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np


def polar2cart(r, theta):
    return r * np.cos(np.deg2rad(theta)), r * np.sin(np.deg2rad(theta))


def odom_data(data: Odometry, args):
    args[1].pose.orientation = data.pose.pose.orientation
    args[2].pose.orientation = data.pose.pose.orientation


def laser_data(data, args):
    k = 20
    min_dist = 0.15
    min_points = 20

    ranges = np.zeros((len(data.ranges), 2))
    for index in range(ranges.shape[0]):
        ranges[index] = [data.ranges[index], index / 2]
    strip_indices = np.where(ranges[:, 0] < 3.0)
    ranges = ranges[strip_indices]

    args[1].points = []
    while(ranges.shape[0] > min_points):
        iterations = 0
        best_inliers_count = 0
        best_inliers_indices = []
        args[2].points = []
        best_line = [(-1, -1), (-1, -1)]
        while (iterations < k):
            iterations += 1
            point1, point2 = np.random.choice(range(ranges.shape[0]), size=2, replace=False)
            x1, y1 = polar2cart(ranges[point1][0], ranges[point1][1])
            x2, y2 = polar2cart(ranges[point2][0], ranges[point2][1])
            inliers_indices = []
            start_point = []
            end_point = []
            best_inlier_dist = 0
            denum = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            denum = np.clip(denum, 0.001, np.inf)
            for index in range(ranges.shape[0]):
                x0, y0 = polar2cart(ranges[index][0], ranges[index][1])
                distance = abs(((x2 - x1) * (y1 - y0)) - ((x1 - x0) * (y2 - y1))) / denum
                if (distance < min_dist):
                    inliers_indices.append(index)
                    if(not start_point):
                        start_point = [x0, y0]

                    inlier_dist = np.sqrt((x0 - start_point[0]) ** 2 + (y0 - start_point[1]) ** 2)
                    if (inlier_dist > best_inlier_dist):
                        best_inlier_dist = inlier_dist
                        end_point = [x0, y0]

            if (len(inliers_indices) >= best_inliers_count):
                best_inliers_indices = inliers_indices
                best_inliers_count = len(inliers_indices)
                best_line = [start_point, end_point]

        (p00, p01), (p10, p11) = best_line
        args[1].points.append(Point(x=p00, y=p01, z=0))
        args[1].points.append(Point(x=p10, y=p11, z=0))
        new_ranges = np.delete(ranges, best_inliers_indices, axis=0)
        ranges = new_ranges

    if (not rospy.is_shutdown()):
        args[0].publish(args[1])

    # Debug scanner
    args[2].points = []
    for index in range(len(data.ranges)):
        xm, ym = polar2cart(data.ranges[index], index / 2)
        args[2].points.append(Point(x=xm, y=ym, z=0))
    if(not rospy.is_shutdown()):
        args[0].publish(args[2])


def init():

    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
    line_marker = Marker()
    point_marker = Marker()

    line_marker.header.frame_id = "ransac"
    line_marker.header.stamp = rospy.Time.now()
    line_marker.type = line_marker.LINE_LIST
    line_marker.id = 0
    line_marker.pose.orientation.w = 1.0
    line_marker.scale.x = 0.02
    line_marker.color.b = 1.0
    line_marker.color.a = 1.0

    point_marker.header.frame_id = "ransac"
    point_marker.header.stamp = rospy.Time.now()
    point_marker.type = line_marker.POINTS
    point_marker.id = 1
    point_marker.scale.x = 0.02
    point_marker.color.r = 1.0
    point_marker.color.a = 1.0

    rospy.Subscriber('/base_scan', LaserScan, laser_data, (marker_pub, line_marker, point_marker))
    # rospy.Subscriber('/odom', Odometry, odom_data, (marker_pub, line_marker, point_marker))

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('perception', anonymous=True)
    init()
