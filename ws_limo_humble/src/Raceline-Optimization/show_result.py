import sys, glob, os
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1] if len(sys.argv) > 1 else "track_map"
if len(sys.argv) > 2:
    csv = sys.argv[2]
else:
    cands = glob.glob(f"outputs/{name}/traj_race_cl-*.csv") + glob.glob("outputs/traj_race_cl.csv")
    if not cands:
        raise SystemExit(f"결과 파일이 없습니다: outputs/{name}/traj_race_cl-*.csv 를 찾지 못했습니다.")
    csv = max(cands, key=os.path.getmtime)
print("result file:", csv)

d = None
for delim in (",", ";", None):                     # 쉼표 -> 세미콜론 -> 공백 순으로 시도
    try:
        a = np.loadtxt(csv, delimiter=delim, comments="#", ndmin=2)
        if a.shape[1] >= 3:
            d = a; break
    except Exception:
        pass
if d is None:
    raise SystemExit("csv를 읽지 못했습니다. head -3 으로 파일 앞부분을 확인해 주세요.")

if d.shape[1] == 3:                                # x, y, v
    x, y, v = d[:, 0], d[:, 1], d[:, 2]
    print("형식: x, y, v (3열) /", len(d), "점")
else:                                              # s, x, y, psi, kappa, vx, ax ...
    x, y = d[:, 1], d[:, 2]
    v = d[:, 5] if d.shape[1] > 5 else d[:, -1]
    print("형식:", d.shape[1], "열 (s, x, y, ..., vx 순서로 가정) /", len(d), "점")
s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]

area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
print("방향:", "반시계(CCW)" if area2 > 0 else "시계(CW)", "/ 시작점 x=%.2f y=%.2f" % (x[0], y[0]))
print("경로 길이 %.1f m, 속도 min/max %.2f/%.2f m/s" % (s[-1], v.min(), v.max()))

cfg = yaml.safe_load(open(f"maps/{name}.yaml")); res = cfg["resolution"]; ox, oy = cfg["origin"][:2]
img = np.array(Image.open(f"maps/{name}.pgm")); H, W = img.shape
fig, ax = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
ax[0].imshow(img, cmap="gray", extent=[ox, ox + W * res, oy, oy + H * res], origin="upper")
tp = f"inputs/tracks/{name}.csv"
if os.path.exists(tp):
    c = np.loadtxt(tp, delimiter=",", comments="#")
    ax[0].plot(c[:, 0], c[:, 1], "--", color="tab:gray", lw=1, label="centerline")
sc = ax[0].scatter(x, y, c=v, s=6, cmap="viridis"); plt.colorbar(sc, ax=ax[0], label="v [m/s]")
k = max(1, len(x) // 40)
ax[0].quiver(x[::k], y[::k], (np.roll(x, -k) - x)[::k], (np.roll(y, -k) - y)[::k],
             angles="xy", scale_units="xy", scale=1.5, width=0.003, color="tab:red")
ax[0].plot(x[0], y[0], "yo", ms=10); ax[0].annotate("START", (x[0], y[0]), color="yellow", weight="bold",
             xytext=(6, 8), textcoords="offset points"); ax[0].legend()
ax[1].plot(s, v); ax[1].set_xlabel("s [m]"); ax[1].set_ylabel("v [m/s]"); ax[1].grid(True)
out = f"result_{name}.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("저장:", out)
