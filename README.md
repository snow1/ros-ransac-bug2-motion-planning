# Bug2 Motion Planning using RANSAC, ROS and Python

Uses RANSAC algorithm to detect walls from laser scan data and pass it to Bug2 node.

Bug2 is a motion planning algorithm uses that RANSAC data during wall following mode and reach the goal defined in launch file.

### System Info

* Ubuntu 20.04 LTS
* ROS Noetic
* Python 3.8.10
* Simulator: stage_ros


### To run the project use

Source your ros workspace first, and build package if not done

```
cd Desktop/ros1_ws/
source devel/setup.bash
catkin_make

roslaunch ransacbug2 perception.launch
roslaunch ransacbug2 bug2.launch
```

### Motion Planning Demo
![](https://github.com/JayParikh20/ransac_bug2/blob/main/demo/ransac_bug2.gif)

