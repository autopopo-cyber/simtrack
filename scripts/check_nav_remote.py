"""一次性检查节点：打印 /odom 位姿 + /map 尺寸/已知率。远程诊断用。"""
import rclpy, time
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class C(Node):
    def __init__(self):
        super().__init__("check_nav")
        self.odom = None
        self.map = None
        sq = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 10)
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "map", m), sq)


rclpy.init()
n = C()
t0 = time.time()
while time.time() - t0 < 8 and n.odom is None:
    rclpy.spin_once(n, timeout_sec=0.2)
t1 = time.time()
while time.time() - t1 < 6 and n.map is None:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.odom:
    p = n.odom.pose.pose.position
    print("ODOM x=%.2f y=%.2f" % (p.x, p.y))
else:
    print("ODOM_NONE")
if n.map:
    i = n.map.info
    known = sum(1 for v in n.map.data if v >= 0)
    free = sum(1 for v in n.map.data if v == 0)
    print("MAP %dx%d res=%.3f origin=(%.1f,%.1f) cells=%d known=%d free=%d" % (
        i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y,
        len(n.map.data), known, free))
else:
    print("MAP_NONE")
n.destroy_node()
rclpy.shutdown()
