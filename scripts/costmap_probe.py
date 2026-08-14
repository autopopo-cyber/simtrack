"""探 global_costmap 指定点的代价值。诊断 Nav2 'failed to create plan' 用。
用法：python3 costmap_probe.py [gx gy] [gx gy] ...  （默认探狗当前位姿 + (12.6,46.4)）
"""
import sys, math, rclpy, time
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros


def cost_label(v):
    if v == 0:
        return "FREE(0)"
    if v >= 254 or v == 100:
        return "LETHAL/OCC(%d)" % v
    if v == 253:
        return "INSCRIBED(253)"
    if v == 255:
        return "NO_INFO(255)"
    return "inflated(%d)" % v


class P(Node):
    def __init__(self, pts):
        super().__init__("cost_probe")
        self.pts = pts
        self.cm = None
        q = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE,
                       history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 lambda m: setattr(self, "cm", m), q)
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf, self)


rclpy.init()
args = [float(x) for x in sys.argv[1:]]
n = P([(args[i], args[i + 1]) for i in range(0, len(args), 2)] if args else None)
t0 = time.time()
while time.time() - t0 < 8 and n.cm is None:
    rclpy.spin_once(n, timeout_sec=0.2)
if n.cm is None:
    print("NO_COSTMAP"); raise SystemExit
# add robot pose
try:
    tf = n.tf.lookup_transform("map", "base_footprint", rclpy.time.Time())
    rp = (tf.transform.translation.x, tf.transform.translation.y)
except Exception:
    rp = None
i = n.cm.info
def cost(wx, wy):
    col = int((wx - i.origin.position.x) / i.resolution)
    row = int((wy - i.origin.position.y) / i.resolution)
    if not (0 <= col < i.width and 0 <= row < i.height):
        return None
    return n.cm.data[row * i.width + col]
print("costmap %dx%d res=%.3f origin=(%.1f,%.1f)" % (i.width, i.height, i.resolution,
      i.origin.position.x, i.origin.position.y))
if rp:
    print("robot (%.2f,%.2f) cost=%s" % (rp[0], rp[1], cost_label(cost(rp[0], rp[1]))))
for (gx, gy) in (n.pts or []):
    print("goal  (%.2f,%.2f) cost=%s" % (gx, gy, cost_label(cost(gx, gy))))
n.destroy_node(); rclpy.shutdown()
