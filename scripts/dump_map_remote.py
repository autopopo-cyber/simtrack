"""一次性：把 /map 存成 npz 供本地渲染。远程诊断用。"""
import rclpy, time, numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class C(Node):
    def __init__(self):
        super().__init__("dump_map")
        self.map = None
        sq = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, "/map", lambda m: setattr(self, "map", m), sq)


rclpy.init()
n = C()
t0 = time.time()
while time.time() - t0 < 8 and n.map is None:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.map:
    i = n.map.info
    arr = np.array(n.map.data, dtype=np.int8).reshape(i.height, i.width)
    np.savez("/home/qin/simtrack/_mapdump.npz",
             data=arr, res=i.resolution,
             ox=i.origin.position.x, oy=i.origin.position.y,
             w=i.width, h=i.height)
    known = int((arr >= 0).sum())
    free = int((arr == 0).sum())
    print("DUMPED %dx%d res=%.3f origin=(%.1f,%.1f) known=%d free=%d" % (
        i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y,
        known, free))
else:
    print("MAP_NONE")
n.destroy_node()
rclpy.shutdown()
