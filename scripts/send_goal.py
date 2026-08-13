#!/usr/bin/env python3
"""发一个 Nav2 NavigateToPose 目标，等结果后退出。用于"走到终点"演示。

用法（远程已放 ~/simtrack/send_goal.py + run_goal.sh）:
    /usr/bin/python3 ~/simtrack/send_goal.py 13.5 13.5
本地不直接用（需 ROS2 环境），仅入库存档。
"""
import sys
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped


class Sender(Node):
    def __init__(self, x, y):
        super().__init__("send_goal")
        self.cli = ActionClient(self, NavigateToPose, "navigate_to_pose")
        if not self.cli.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("navigate_to_pose 服务未就绪"); return
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = "map"
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x = float(x)
        g.pose.pose.position.y = float(y)
        g.pose.pose.orientation.w = 1.0
        fut = self.cli.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10)
        gh = fut.result()
        print("accepted", bool(gh and gh.accepted))
        if gh and gh.accepted:
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(self, rf, timeout_sec=200)
            print("RESULT", rf.result().status if rf.result() else None)


def main():
    if len(sys.argv) < 3:
        print("用法: send_goal.py X Y"); sys.exit(1)
    rclpy.init()
    Sender(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
