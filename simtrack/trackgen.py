"""
TrackGen — 蛇形赛道 hfield 生成器

生成 50×50m 的蛇形赛道高度图 (hfield PNG)，
包含路面、3m 高护栏（3px 宽保证雷达可检测）。

用法:
    from simtrack.trackgen import TrackGenerator
    tg = TrackGenerator(hf_res=2000, guard_height=3.0)
    tg.generate()
    tg.save("/tmp/track_hd.png")

可调参数:
    hf_res       — hfield 分辨率 (默认 2000)
    guard_height — 护栏高度 (默认 3m, >雷达安装高度以屏蔽相邻赛道)
    road_width   — 路宽 (默认 5m)
    n_segments   — 蛇形段数 (默认 10)
    turn_radius  — U 型弯半径 (默认 5m)
"""

import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    import warnings
    warnings.warn("opencv-python-headless 未安装，PNG 保存功能不可用")


class TrackGenerator:
    """蛇形赛道 hfield 生成器。

    生成 50×50m 地图上的蛇形赛道，包含:
    - 路面 (像素 128, 高度 ≈0m)
    - 连续护栏 (像素 255, 高度 = guard_height)
    - 弧形帽 (起终点封闭)

    hfield 编码公式:
        height_m = pixel / 255 * scale - negative
        其中 scale = guard_height / 1.0 + negative

    护栏使用 3px 笔刷保证 MuJoCo 射线检测可靠性。
    """

    def __init__(
        self,
        map_size: float = 50.0,
        hf_res: int = 2000,
        road_width: float = 5.0,
        guard_height: float = 3.0,
        n_segments: int = 10,
        turn_radius: float = 5.0,
        start_clear: float = 5.0,
        end_clear: float = 5.0,
        guard_brush: int = 3,
        seed: int = 42,
    ):
        """初始化 TrackGenerator。

        Args:
            map_size: 地图大小 (米), 正方形
            hf_res: hfield 分辨率 (像素)
            road_width: 路面宽度 (米)
            guard_height: 护栏高度 (米), 应 > 雷达高度以屏蔽相邻赛道
            n_segments: 蛇形段数
            turn_radius: U 型弯半径 (米)
            start_clear: 起点无障碍区 (米)
            end_clear: 终点无障碍区 (米)
            guard_brush: 护栏笔刷半径 (像素), ≥3 保证雷达可检测
            seed: 随机种子
        """
        self.map_size = map_size
        self.hf_res = hf_res
        self.road_width = road_width
        self.guard_height = guard_height
        self.n_segments = n_segments
        self.turn_radius = turn_radius
        self.start_clear = start_clear
        self.end_clear = end_clear
        self.guard_brush = max(guard_brush, 3)  # 最少 3px
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # hfield 编码: height = pixel/255 * scale - negative
        # 路面=0m → pixel=128; 护栏=guard_height → pixel=255
        self.negative = guard_height  # 满足 255/255*scale - negative = guard_height
        self.scale = guard_height * 2  # 满足 128/255*scale - negative ≈ 0

        # 像素值
        self.ROAD_PIX = 128
        self.GUARD_PIX = 255

    # ── 中心线生成 ──

    def _gen_centerline(self):
        """生成蛇形赛道中心线。

        Returns:
            list[tuple]: 中心线点列表 [(x,y), ...]
            float: 赛道总长 (米)
            list[tuple]: waypoints [(x,y), ...]
        """
        y_start, y_end = 45.0, 5.0
        dy = (y_start - y_end) / (self.n_segments - 1)
        pts = []
        waypoints = []

        # 起点直道
        y0 = y_start - dy / 2
        waypoints.append((5.0, y0))
        waypoints.append((45.0, y0))
        for x in np.arange(5.0, 45.01, 0.25):
            pts.append((float(x), float(y0)))

        for i in range(self.n_segments):
            y = y_start - i * dy
            left_to_right = (i % 2 == 0)
            if left_to_right:
                waypoints.append((5.0, float(y)))
                waypoints.append((45.0, float(y)))
            else:
                waypoints.append((45.0, float(y)))
                waypoints.append((5.0, float(y)))
            xs = np.arange(5.0, 45.01, 0.25) if left_to_right else np.arange(45.0, 4.99, -0.25)
            for x in xs:
                pts.append((float(x), float(y)))

            # U 型弯头
            if i < self.n_segments - 1:
                ny = y_start - (i + 1) * dy
                cx = 45.0 if left_to_right else 5.0
                cy = (y + ny) / 2.0
                sa = math.pi / 2 if left_to_right else 3 * math.pi / 2
                ea = 3 * math.pi / 2 if left_to_right else 5 * math.pi / 2
                n_arc = max(10, int(math.pi * self.turn_radius / 0.25))
                for j in range(1, n_arc + 1):
                    a = sa + (ea - sa) * j / n_arc
                    pts.append((cx + self.turn_radius * math.cos(a),
                                cy + self.turn_radius * math.sin(a)))

        # 终点直道
        last_lr = ((self.n_segments - 1) % 2 == 0)
        ye = y_end + dy / 2
        waypoints.append((5.0 if last_lr else 45.0, float(y_end)))
        waypoints.append((45.0 if last_lr else 5.0, float(ye)))
        for x in np.arange(5.0, 45.01, 0.25):
            pts.append((float(x), float(ye)))

        # 总长
        total = sum(
            math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            for i in range(1, len(pts))
        )
        return pts, total, waypoints

    # ── 坐标转换 ──

    def _w2p(self, wx, wy):
        """世界坐标 → hfield 像素"""
        px = int(wx / self.map_size * self.hf_res)
        py = int(wy / self.map_size * self.hf_res)
        return (min(max(px, 0), self.hf_res - 1),
                min(max(py, 0), self.hf_res - 1))

    # ── 主生成 ──

    def generate(self):
        """执行生成。结果存于 self.hfield, self.center_line, self.total_len, self.waypoints。"""
        print(f"TrackGen: {self.map_size}×{self.map_size}m, "
              f"hfield {self.hf_res}×{self.hf_res}, "
              f"{self.n_segments}段, 护栏{self.guard_height}m "
              f"(笔刷{self.guard_brush}px)")

        center_line, total_len, waypoints = self._gen_centerline()
        print(f"  中心线: {len(center_line)} 点, {total_len:.0f}m, "
              f"{len(waypoints)} waypoints")

        # 全图路面
        hf = np.full((self.hf_res, self.hf_res), self.ROAD_PIX, dtype=np.uint8)

        # 护栏绘制 — 3px 笔刷
        n_guard = 0
        r2 = self.guard_brush ** 2
        prev_nx, prev_ny = 0.0, 0.0

        for i in range(len(center_line)):
            cx, cy = center_line[i]
            # 法线
            if i < len(center_line) - 1:
                tx = center_line[i + 1][0] - cx
                ty = center_line[i + 1][1] - cy
            elif i > 0:
                tx = cx - center_line[i - 1][0]
                ty = cy - center_line[i - 1][1]
            else:
                continue
            tlen = math.hypot(tx, ty)
            if tlen < 0.001:
                continue
            nx_dir = -ty / tlen
            ny_dir = tx / tlen
            # 防翻转
            if prev_nx * nx_dir + prev_ny * ny_dir < 0:
                nx_dir, ny_dir = -nx_dir, -ny_dir
            if cx < 15.0 and nx_dir > 0:
                nx_dir, ny_dir = -nx_dir, -ny_dir
            elif cx > 35.0 and nx_dir < 0:
                nx_dir, ny_dir = -nx_dir, -ny_dir
            prev_nx, prev_ny = nx_dir, ny_dir

            for side in [-1, 1]:
                gx = cx + side * (self.road_width / 2) * nx_dir
                gy = cy + side * (self.road_width / 2) * ny_dir
                gpx, gpy = self._w2p(gx, gy)
                # 3px 笔刷: 画所有在笔刷半径内的像素
                for dx in range(-self.guard_brush, self.guard_brush + 1):
                    for dy in range(-self.guard_brush, self.guard_brush + 1):
                        if dx * dx + dy * dy <= r2:
                            px, py = gpx + dx, gpy + dy
                            if 0 <= px < self.hf_res and 0 <= py < self.hf_res:
                                if hf[py, px] < self.GUARD_PIX:
                                    hf[py, px] = self.GUARD_PIX
                                    n_guard += 1

        # 弧形帽
        sx, sy = center_line[0]
        for a in np.linspace(np.pi / 2, -np.pi / 2, 80):
            gx = sx + (self.road_width / 2) * np.cos(a)
            gy = sy + (self.road_width / 2) * np.sin(a)
            gpx, gpy = self._w2p(gx, gy)
            for dx in range(-self.guard_brush, self.guard_brush + 1):
                for dy in range(-self.guard_brush, self.guard_brush + 1):
                    if dx * dx + dy * dy <= r2:
                        px, py = gpx + dx, gpy + dy
                        if 0 <= px < self.hf_res and 0 <= py < self.hf_res:
                            if hf[py, px] < self.GUARD_PIX:
                                hf[py, px] = self.GUARD_PIX
                                n_guard += 1

        ex, ey = center_line[-1]
        for a in np.linspace(-np.pi / 2, np.pi / 2, 80):
            gx = ex + (self.road_width / 2) * np.cos(a)
            gy = ey + (self.road_width / 2) * np.sin(a)
            gpx, gpy = self._w2p(gx, gy)
            for dx in range(-self.guard_brush, self.guard_brush + 1):
                for dy in range(-self.guard_brush, self.guard_brush + 1):
                    if dx * dx + dy * dy <= r2:
                        px, py = gpx + dx, gpy + dy
                        if 0 <= px < self.hf_res and 0 <= py < self.hf_res:
                            if hf[py, px] < self.GUARD_PIX:
                                hf[py, px] = self.GUARD_PIX
                                n_guard += 1

        print(f"  护栏: {n_guard} 像素 (笔刷{self.guard_brush}px)")

        self.hfield = hf
        self.center_line = center_line
        self.total_len = total_len
        self.waypoints = waypoints
        return self

    # ── 保存 ──

    def save(self, png_path, preview_path=None):
        """保存 hfield PNG 和可选预览图。

        Args:
            png_path: hfield PNG 输出路径
            preview_path: 彩色预览图路径 (可选)
        """
        if cv2 is None:
            raise RuntimeError("opencv-python-headless 未安装")
        cv2.imwrite(png_path, self.hfield)
        print(f"  PNG: {png_path}")

        if preview_path:
            h, w = self.hfield.shape
            preview = np.zeros((h, w, 3), dtype=np.uint8)
            preview[self.hfield == self.ROAD_PIX] = [180, 180, 185]
            preview[self.hfield == self.GUARD_PIX] = [220, 40, 40]
            cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            print(f"  预览: {preview_path}")

    @property
    def hfield_size(self):
        """MuJoCo hfield size 参数 (half_x, half_y, scale, negative)"""
        return (self.map_size / 2, self.map_size / 2, self.scale, self.negative)


# ── 自测 ──
def _self_test():
    """纯 Python 自测，不依赖 MuJoCo"""
    tg = TrackGenerator(hf_res=500, guard_height=3.0)
    tg.generate()

    hf = tg.hfield
    unique = np.unique(hf)
    assert 128 in unique, f"路面像素 128 缺失"
    assert 255 in unique, f"护栏像素 255 缺失"
    assert len(tg.center_line) > 100
    assert tg.total_len > 500
    print(f"  ✓ 自测通过: shape={hf.shape}, values={unique.tolist()}, "
          f"len={tg.total_len:.0f}m")
    return True


if __name__ == "__main__":
    _self_test()
