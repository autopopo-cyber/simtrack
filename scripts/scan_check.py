"""一次性：检查 /scan 的射程分布 + /map 当前已知率。远程诊断用。"""
import rclpy, time, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class C(Node):
    def __init__(self):
        super().__init__("scan_check")
        self.scan = None
        sq = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, "/scan", lambda m: setattr(self, "scan", m), sq)
        mq = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        self.map = None
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "map", m), mq)


rclpy.init()
n = C()
t0 = time.time()
while time.time() - t0 < 6 and n.scan is None:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.scan:
    r = np.array(n.scan.ranges)
    fin = r[np.isfinite(r)]
    print("SCAN: %d rays, range_max=%.1f, hits=%d, min=%.2f median=%.2f max=%.2f"
          % (len(r), n.scan.range_max, len(fin), fin.min(), np.median(fin), fin.max()))
    print("  rays>5m: %d, rays>2m: %d" % (int((fin > 5).sum()), int((fin > 2).sum())))
else:
    print("SCAN_NONE")
t1 = time.time()
while time.time() - t1 < 4 and n.map is None:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.map:
    g = np.array(n.map.data)
    print("MAP: %dx%d known=%d free=%d wall=%d"
          % (n.map.info.width, n.map.info.height, int((g >= 0).sum()),
             int((g == 0).sum()), int((g > 50).sum())))
n.destroy_node()
rclpy.shutdown()
