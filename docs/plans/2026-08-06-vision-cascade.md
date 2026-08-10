# 视觉金字塔级联粗筛 + 2m 贴墙标牌 实现计划

> **For Hermes:** 用 subagent-driven-development 逐任务实现。
> **Goal:** 标牌放大 2m 贴三面墙场景 + Laplacian 能量粗筛（无特征跳过金字塔）。
> **Architecture:** landmarks.py 定义 2m 标牌+三面墙 XML；vision_landmark.py 加 _coarse_gate（0.25x Laplacian<200 返回空）；algo3 接入。
> **Tech Stack:** Python, MuJoCo, OpenCV(cv2.aruco/Laplacian)

---

### Task 1: landmarks.py — 标牌 2m + 三面墙场景

**Objective:** LM_HALF=1.0（2m 标牌）、标牌贴横墙内表面凸出 0.3m、场景加三面 box 墙。

**Files:**
- Modify: `test_scripts/landmarks.py`

**Step 1: 修改 LM_HALF**
```python
LM_HALF = 1.0   # 2m×2m（主人指令放大 2 倍）
LM_Z = HF_SURF + 1.0   # 中心离地 1m = 相机高度
```

**Step 2: 新增 wall_xml() 生成三面 box 墙**（通道两侧 y=0/y=5 + 横墙 x=50，contype=0 纯可视化）

**Step 3: landmark_xml() 标牌贴横墙内表面**（x=49.6 凸出 0.3m，quat 绕 y -90° 法线朝 -x）

**Step 4: 验证**
```bash
/home/qin/mujoco-venv/bin/python -c "from test_scripts.landmarks import landmark_xml, wall_xml; a,w=landmark_xml(); print('标牌数', len(w.splitlines()), '墙', len(wall_xml().splitlines()))"
```
Expected: 标牌数 30, 墙 3

**Step 5: Commit**
```bash
git add test_scripts/landmarks.py && git commit -m "feat: 标牌放大2m贴三面墙场景"
```

---

### Task 2: vision_landmark.py — Laplacian 能量粗筛

**Objective:** _detect_pyramid 前加 _coarse_gate：0.25x Laplacian 能量 < 200 → return []。

**Files:**
- Modify: `test_scripts/vision_landmark.py`

**Step 1: 添加 _coarse_gate 方法**
```python
def _coarse_gate(self, gray):
    """粗筛：0.25x Laplacian 能量 < 200 → 无黑白高对比特征 → 跳过金字塔。
    实测：无标牌 23-84，有标牌 231-1684，阈值 200 零重叠。"""
    h, w = gray.shape
    g = cv2.resize(gray, (int(w*0.25), int(h*0.25)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    return float((lap**2).mean()) >= 200.0
```

**Step 2: _detect_pyramid 开头接入**
```python
if not self._coarse_gate(gray):
    return []   # 无特征，跳过全部高分辨率
```

**Step 3: 验证**（复用 /tmp/test_2m.py 的帧，无标牌跳过/有标牌继续）

**Step 4: Commit**

---

### Task 3: algo3_headless.py — 三面墙接入

**Objective:** XML 加 wall_xml()，标牌场景完整。

**Files:**
- Modify: `test_scripts/algo3_headless.py:505-538`

**Step 1: import wall_xml，XML 的 {LM_WORLD} 前加 {WALL_XML}**

**Step 2: 验证**
```bash
timeout 60 /home/qin/mujoco-venv/bin/python test_scripts/algo3_headless.py --landmarks 1 --vision 1 --timeout 20
```
Expected: landmarks_unique ≥ 2，无报错

**Step 3: Commit**

---

### Task 4: 性能验证

**Objective:** 无标牌帧检测 < 2ms（原 25ms），有标牌帧识别不降。

**Files:**
- Test: `/tmp/test_perf_baseline.py`（改调用级联版）

**Step 1: 无标牌帧计时**
Expected: 粗筛跳过，单帧 < 2ms

**Step 2: 有标牌帧 30s 导航**
Expected: landmarks_unique ≥ 2（2m 标牌）

**Step 3: 文档更新** docs/2026-08-05-dog50-maze-pitfalls.md 记粗筛经验

**Step 4: Commit**
