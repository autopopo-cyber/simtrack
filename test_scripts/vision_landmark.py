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
    def __init__(self, m, d, renderer, cam_name="bot_cam", detect_every=40):
        self.m, self.d, self.renderer = m, d, renderer
        self.cam_name = cam_name
        self.detect_every = detect_every
        self.step_count = 0
        self.LM = _load_landmarks()
        # ArUco 检测器（DICT_7X7_1000）
        if CV2_OK and self.LM is not None:
            dict_name = getattr(self.LM, "ARUCO_DICT", "DICT_7X7_1000")
            dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_7X7_1000)
            self.detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(dict_id))
        else:
            self.detector = None
        # 图像金字塔尺度（主人指令：多尺度避免只拍局部）
        # 码太小（远处）→ 放大 (2x,3x)；码太大（近处，占满视野）→ 缩小 (0.25x,0.5x)
        # 实测：狗 2m 看 1.5m 标牌时 scale=0.5 识别成功（原图码太大检测器失败）
        self.pyramid_scales = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # 远小码→近大码
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
        if not CV2_OK:
            return []
        return self._detect_pyramid(img, step)

    def _detect_pyramid(self, img, step):
        """图像金字塔多尺度 ArUco 检测。
        主人指令：生成不同分辨率金字塔，多尺度识别——远处小码用放大尺度，
        近处大码用缩小尺度，避免只拍到局部漏检。
        一步内同一标牌去重（多尺度命中只记一次）。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        results = []
        step_ids = set()  # 本步已记录标牌（多尺度去重）
        for scale in self.pyramid_scales:
            h, w = gray.shape
            nh, nw = int(h * scale), int(w * scale)
            if nh < 60 or nw < 60:   # 太小无意义
                continue
            if scale < 1.0:
                g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
            try:
                corners, ids, rejected = self.detector.detectMarkers(g)
            except Exception:
                continue
            if ids is None or len(ids) == 0:
                continue
            for i, marker_id in enumerate(ids.flatten()):
                idx = int(marker_id)
                if idx >= 30 or idx not in self._pos_map():
                    continue
                if idx in step_ids:   # 本步已记录，跳过（多尺度去重）
                    continue
                step_ids.add(idx)
                ch, slot = self._idx_ch_slot(idx)
                self.total_detected += 1
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
            for idx, ch, slot, wx, wy, wz in self.LM.landmark_positions():
                self._pm[idx] = (wx, wy)
        return self._pm

    def _idx_ch_slot(self, idx):
        """idx → (ch, slot)：idx = ch*3 + slot"""
        return idx // 3, idx % 3
