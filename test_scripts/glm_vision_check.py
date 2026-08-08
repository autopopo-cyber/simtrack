#!/usr/bin/env python3
"""GLM-5V 视觉巡检 — 用智谱 glm-5v-turbo 判断仿真帧

输入: 渲染帧 PNG 目录
输出: JSON 巡检报告（每帧：问题 + GLM 回答）
"""
import os, sys, json, base64, urllib.request, urllib.error, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--frames-dir", default="/tmp/firefly_s170456", help="渲染帧目录")
ap.add_argument("--max-frames", type=int, default=5, help="最多巡检几帧")
ap.add_argument("--out", default="/tmp/firefly_glm_report.json", help="输出 JSON")
args = ap.parse_args()

# 读 GLM key
key = None
with open(os.path.expanduser("~/junxiu/PASSBOOK.md")) as f:
    for line in f:
        if "GLM_API_KEY" in line and "=" in line:
            key = line.strip().split("=", 1)[1]
            break
if not key:
    print("❌ 找不到 GLM_API_KEY")
    sys.exit(1)

API_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

def ask_glm(image_b64: str, question: str) -> str:
    """调智谱 glm-5v-turbo 看图回答"""
    payload = {
        "model": "glm-5v-turbo",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}}
            ]
        }],
        "max_tokens": 300,
        "thinking": {"type": "disabled"}
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=90)
        data = json.loads(resp.read().decode())
        return data["choices"][0]["message"].get("content", "").strip()
    except urllib.error.HTTPError as e:
        return f"[HTTP {e.code}] {e.read().decode()[:200]}"
    except Exception as e:
        return f"[ERR] {e}"

# 收集帧
frames = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
if not frames:
    print("❌ 无帧文件")
    sys.exit(1)

# 均匀抽样
total = len(frames)
step = max(1, total // args.max_frames)
sample = frames[::step][:args.max_frames]
if frames[-1] not in sample:
    sample.append(frames[-1])

report = []
for fname in sample:
    fpath = os.path.join(args.frames_dir, fname)
    with open(fpath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    step_num = fname.replace("frame_", "").replace(".png", "")

    q1 = "这是一张机器人仿真俯视图。画面里机器人（橙色圆柱）在什么位置？前方看起来是通道还是死路/墙壁？用两句话回答。"
    a1 = ask_glm(b64, q1)

    q2 = "机器人是否看起来在贴着墙转圈、卡住不动，或者在正常直线前进？"
    a2 = ask_glm(b64, q2)

    entry = {"frame": fname, "step": step_num, "q1_pos_channel": a1, "q2_stuck_check": a2}
    report.append(entry)
    print(f"[{fname}] Q1: {a1[:80]}...")
    print(f"          Q2: {a2[:80]}...")
    sys.stdout.flush()

with open(args.out, "w") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"\n✅ 巡检完成 → {args.out}")
