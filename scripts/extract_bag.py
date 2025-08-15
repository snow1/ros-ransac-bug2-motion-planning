import rosbag
import pandas as pd

# 读取 bag 文件
bag_path = "/home/chunxue/wf_debug.bag"
bag = rosbag.Bag(bag_path)

# 存储数据
data = {
    "time": [],
    "cmd_vel_linear_y": [],
    "cmd_vel_angular_z": [],
    "wall_lateral": [],
    "wall_has": [],
    "obstacle_ahead": []
}

start_time = None

for topic, msg, t in bag.read_messages(topics=["/cmd_vel", "/wall_lateral", "/wall_has", "/obstacle_ahead"]):
    if start_time is None:
        start_time = t.to_sec()
    time_sec = t.to_sec() - start_time

    if topic == "/cmd_vel":
        data["time"].append(time_sec)
        data["cmd_vel_linear_y"].append(msg.linear.y)
        data["cmd_vel_angular_z"].append(msg.angular.z)
        # 填充其他字段为 None
        data["wall_lateral"].append(None)
        data["wall_has"].append(None)
        data["obstacle_ahead"].append(None)

    elif topic == "/wall_lateral":
        data["time"].append(time_sec)
        data["cmd_vel_linear_y"].append(None)
        data["cmd_vel_angular_z"].append(None)
        data["wall_lateral"].append(msg.data)
        data["wall_has"].append(None)
        data["obstacle_ahead"].append(None)

    elif topic == "/wall_has":
        data["time"].append(time_sec)
        data["cmd_vel_linear_y"].append(None)
        data["cmd_vel_angular_z"].append(None)
        data["wall_lateral"].append(None)
        data["wall_has"].append(msg.data)
        data["obstacle_ahead"].append(None)

    elif topic == "/obstacle_ahead":
        data["time"].append(time_sec)
        data["cmd_vel_linear_y"].append(None)
        data["cmd_vel_angular_z"].append(None)
        data["wall_lateral"].append(None)
        data["wall_has"].append(None)
        data["obstacle_ahead"].append(msg.data)

bag.close()

# 转换成 DataFrame
df = pd.DataFrame(data)

# 合并相同时间的不同字段（按最近一次填充）
df = df.sort_values("time").reset_index(drop=True)
df = df.ffill()

# 只保留主要列
df_main = df[["time", "cmd_vel_linear_y", "cmd_vel_angular_z", "wall_lateral", "wall_has", "obstacle_ahead"]]

# 保存为 CSV
df_main.to_csv("wf_debug.csv", index=False)
print("已保存到 wf_debug.csv")
