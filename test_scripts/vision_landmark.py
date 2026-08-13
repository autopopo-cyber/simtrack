"""视觉地标识别模块：前置相机 → 图像金字塔多尺度 ArUco 检测 → 标牌 ID。

主人设计（2026-08-06）：
- 标准黑白二维码（DICT_7X7），机器狗看码识唯一 ID
- **图像金字塔多尺度检测**（主人指令）：每帧生成多个分辨率（长焦→远距离小码，
  广角→近距离大码），每尺度都跑 ArUco，避免"只拍到局部/码太小"漏检
- 标牌 = 地面平铺 plane（MuJoCo plane 纹理渲染 + ArUco 识别已验证可靠）

用法（在 algo3_headless.py 主循环中）：
    vis = VisionLandmark(m, d, renderer)
    detected = vis.scan_once(step)   # 返回 [(step, idx, ch, slot)]
"""
import os, math, sys
import numpy as np
import mujoco

try:
    import cv2
    CV2_OK = True
except ImportError:
    cv2 = None
    CV2_OK = False

def _load_landmarks():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, ".."))
        from test_scripts import landmarks as LM
        return LM
    except Exception as e:
        print(f"  [VISION] 标牌表初始化失败: {e}", flush=True)
        return None

class VisionLandmark:
    def __init__(self, m, d, renderer, cam_name="bot_cam", detect_every=40, aruco=True, aruco_every=3):
        self.m, self.d, self.renderer = m, d, renderer
        self.cam_name = cam_name
        self.detect_every = detect_every
        self.aruco = aruco   # 场景里没放标牌时关掉 ArUco 金字塔（省 CPU），终点球检测不受影响
        # 2026-08-09 性能：ArUco 每 aruco_every 帧才跑一次（终点球检测仍每帧）。
        # profile：金字塔全尺度 detectMarkers 是视觉最大头（450 次/4000 步=11s）。
        # 40×3=120 步=0.6s 物理一检，通道识别/里程计修正灵敏度足够
        self.aruco_every = max(1, aruco_every)
        self._frame_no = 0
        self.step_count = 0
        self.LM = _load_landmarks()
        # 终点发现（无特权）：每帧检测绿色终点球 → (bearing, dist, step)
        self.finish_obs = None
        # 标牌几何观测：[(step, idx, dist, bearing)]——里程计绝对修正用
        self.geo_obs = []
        try:
            from test_scripts.finish_detect import detect_finish
            self._detect_finish = detect_finish
        except Exception:
            self._detect_finish = None
        # ArUco 检测器（DICT_7X7_1000）
        if CV2_OK and self.LM is not None:
            dict_name = getattr(self.LM, "ARUCO_DICT", "DICT_7X7_1000")
            dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_7X7_1000)
            self.detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(dict_id))
        else:
            self.detector = None
        # 图像金字塔尺度（主人指令：多尺度避免只拍局部）
        # 码太小（远处）→ 放大 (2x)；码太大（近处，占满视野）→ 缩小 (0.5x)
        # 实测：狗 2m 看 1.5m 标牌时 scale=0.5 识别成功（原图码太大检测器失败）
        # 2026-08-09 性能：6 尺度→3 尺度。2m 宽标牌在 1280×720 下 25m 处仍有 ~70px 边长，
        # 1.0 尺度即可检；3x 放大到 4K 的 detectMarkers 是单帧最大开销（~60ms），实际无增益
        self.pyramid_scales = [0.5, 1.0, 2.0]
        # 统计
        self.total_detected = 0
        self.seen_ids = set()
        self.detections = []  # [(step, idx, ch, slot)]

    def _apply_bob(self, step):
        """模拟相机颠簸：行走节律 ±5° 水平 + 垂直微颤"""
        t = step * self.m.opt.timestep
        yaw_bob = 5.0 * math.sin(2 * math.pi * 1.5 * t) + np.random.uniform(-0.5, 0.5)
        pitch_bob = 3.0 * math.sin(2 * math.pi * 3.0 * t + 0.5) + np.random.uniform(-0.5, 0.5)
        return yaw_bob, pitch_bob

    def scan_once(self, step):
        """渲染相机帧 + 金字塔多尺度 ArUco 检测。返回 [(step, idx, ch, slot)]"""
        if self.renderer is None or self.detector is None or self.LM is None:
            return []
        self.step_count += 1
        if self.step_count % self.detect_every != 0:
            return []
        self._frame_no += 1
        _run_aruco = self.aruco and (self._frame_no % self.aruco_every == 0)
        try:
            yaw_bob, pitch_bob = self._apply_bob(step)
            orig_mat = self.m.cam_mat0[0].copy()
            ry, rp = np.deg2rad(yaw_bob), np.deg2rad(pitch_bob)
            Rz = np.array([[math.cos(ry), -math.sin(ry), 0],
                           [math.sin(ry), math.cos(ry), 0],
                           [0, 0, 1]])
            Rx = np.array([[1, 0, 0],
                           [0, math.cos(rp), -math.sin(rp)],
                           [0, math.sin(rp), math.cos(rp)]])
            M = orig_mat.reshape(3, 3)
            self.m.cam_mat0[0] = (M @ Rz @ Rx).flatten()
            self.renderer.update_scene(self.d, camera=self.cam_name)
            img = self.renderer.render()
            self.m.cam_mat0[0] = orig_mat
        except Exception:
            return []
        # 终点发现：绿色终点球（无特权，狗亲眼看到才知终点在哪）→ (bearing, dist, area, step)
        if self._detect_finish is not None:
            try:
                fov = float(self.m.cam_fovy[0]) if hasattr(self.m, "cam_fovy") else 45.0
                obs = self._detect_finish(img, fovy_deg=fov)
                if obs is not None:
                    # 面积归一化到 720p 等效（algo3 到达阈值 12000/25000 是 720p 标定）：
                    # 640×360 渲染下像素面积是 720p 的 1/4，不归一则到达判定偏远 ~2 倍
                    _h = img.shape[0]
                    _area720 = obs[2] * (720.0 / _h) ** 2
                    self.finish_obs = (obs[0], obs[1], _area720, obs[3], step)
            except Exception:
                pass
        if not CV2_OK or self.detector is None or not _run_aruco:
            return []
        return self._detect_pyramid(img, step)

    def _coarse_gate(self, gray):
        """粗筛：0.25x Laplacian 能量 < 200 → 无黑白高对比特征 → 跳过金字塔。
        主人指令（2026-08-06）：低分辨率特征都没有，高分辨率检测就不用做。
        实测（0.25x Laplacian 能量）：无标牌 23-84，有标牌 231-1684，阈值 200 零重叠。
        """
        h, w = gray.shape
        g = cv2.resize(gray, (int(w * 0.25), int(h * 0.25)), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(g, cv2.CV_64F)
        return float((lap**2).mean()) >= 200.0

    def _detect_pyramid(self, img, step):
        """图像金字塔多尺度 ArUco 检测。
        主人指令：生成不同分辨率金字塔，多尺度识别——远处小码用放大尺度，
        近处大码用缩小尺度，避免只拍到局部漏检。
        一级粗筛：Laplacian 能量不足（无黑白高对比特征）→ 跳过全部高分辨率。
        一步内同一标牌去重（多尺度命中只记一次）。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if not self._coarse_gate(gray):
            return []   # 无特征，跳过全部金字塔（省 ~25ms/帧）
        results = []
        step_ids = set()  # 本步已记录标牌（多尺度去重）
        for scale in self.pyramid_scales:
            h, w = gray.shape
            nh, nw = int(h * scale), int(w * scale)
            if nh < 60 or nw < 60:   # 太小无意义
                continue
            # 用默认 INTER_LINEAR：INTER_AREA 缩小会模糊小码致 ArUco 识别失败（实测）
            g = cv2.resize(gray, (nw, nh))
            try:
                corners, ids, rejected = self.detector.detectMarkers(g)
            except Exception:
                continue
            if ids is None or len(ids) == 0:
                continue
            for i, marker_id in enumerate(ids.flatten()):
                idx = int(marker_id)
                if idx >= 30 or idx not in self._pos_map():   # aruco PNG 仅 00-29；idx10-29=中间锚点
                    continue
                if idx in step_ids:   # 本步已记录，跳过（多尺度去重）
                    continue
                step_ids.add(idx)
                ch, slot = self._idx_ch_slot(idx)
                self.total_detected += 1
                # 标牌几何（里程计修正用）：角点换回原始分辨率 → 边长 px → 距离，中心 x → 方位
                # 标牌真实尺寸 2m×2m（landmarks.LM_HALF=1.0 半尺寸）
                try:
                    import math as _m
                    pts = corners[i].reshape(-1, 2) / scale   # 原始图像素坐标
                    side_px = float(np.mean([np.hypot(pts[(j+1) % 4][0]-pts[j][0],
                                                      pts[(j+1) % 4][1]-pts[j][1]) for j in range(4)]))
                    h0, w0 = img.shape[:2]
                    _fy = (h0 / 2.0) / _m.tan(_m.radians(float(self.m.cam_fovy[0])) / 2.0)
                    _dist = 2.0 * _fy / max(side_px, 1.0)
                    _bearing = _m.atan((float(pts[:, 0].mean()) - w0 / 2.0) / _fy)
                    self.geo_obs.append((step, idx, _dist, _bearing))
                except Exception:
                    pass
                self.seen_ids.add(idx)
                self.detections.append((step, idx, ch, slot))
                results.append((step, idx, ch, slot, scale))
                print(f"  [VISION] step={step} 看到标牌#{idx} 通道{ch} 位置{slot} "
                      f"尺度{scale}x", flush=True)
        return results

    def _pos_map(self):
        """idx → 世界坐标（从 landmarks 构建）"""
        if not hasattr(self, "_pm"):
            self._pm = {}
            for idx, ch, side, wx, wy, wz, quat in self.LM.landmark_positions():
                self._pm[idx] = (wx, wy)
        return self._pm

    def _idx_ch_slot(self, idx):
        """idx → (ch, side)：idx 直接 = 通道号"""
        return idx, 0
