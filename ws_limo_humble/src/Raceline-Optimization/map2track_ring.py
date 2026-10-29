import sys
import numpy as np, yaml
from PIL import Image
from scipy import ndimage
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1] if len(sys.argv) > 1 else "track_map"
PGM, YAML = f"maps/{name}.pgm", f"maps/{name}.yaml"
OUT = f"inputs/tracks/{name}.csv"
print("map:", PGM, "->", OUT)

STEP_M = 0.1         # 출력 점 간격 [m]
SMOOTH_M = 0.8       # 중심선 평활 길이 [m]
MAX_HALF_W = 1.0     # 한쪽 폭 상한 [m]
MIN_BLOB_PX = 150    # 이보다 작은 장애물 덩어리는 노이즈로 보고 메움

cfg = yaml.safe_load(open(YAML)); res = cfg["resolution"]; ox, oy = cfg["origin"][:2]
img = np.array(Image.open(PGM)); H, W = img.shape

free = img >= 250
lab, n = ndimage.label(free)
sizes = ndimage.sum(free, lab, range(1, n + 1))
free = lab == (1 + int(np.argmax(sizes)))          # 가장 큰 자유 공간만

obst = ~free
lab, n = ndimage.label(obst)
border = np.zeros_like(obst)
border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
outer = np.zeros_like(obst); inner = np.zeros_like(obst); isl = []
for i in range(1, n + 1):
    m = lab == i
    if (m & border).any():
        outer |= m                                  # 바깥 벽/미지 영역
    elif m.sum() >= MIN_BLOB_PX:
        inner |= m; isl.append(int(m.sum()))        # 안쪽 섬
    else:
        free |= m                                   # 작은 노이즈는 메움
print("islands:", len(isl), "sizes(px):", isl)
if not isl:
    raise SystemExit("안쪽 섬을 못 찾았습니다. 섬이 있는 고리 모양 트랙에서만 동작합니다.")

d_in = ndimage.distance_transform_edt(~inner) * res
d_out = ndimage.distance_transform_edt(~outer) * res
cs = measure.find_contours(d_in - d_out, 0.0)      # 섬/바깥벽 거리가 같은 선
c = max(cs, key=len)
print("contours:", len(cs), "longest px:", len(c), "closed:", bool(np.allclose(c[0], c[-1])))

x = ox + (c[:, 1] + 0.5) * res
y = oy + (H - 1 - c[:, 0] + 0.5) * res
s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
sn = np.arange(0, s[-1], STEP_M)
xs, ys = np.interp(sn, s, x), np.interp(sn, s, y)

def smooth(a, k):
    k += (k % 2 == 0); p = k // 2
    return np.convolve(np.r_[a[-p:], a, a[:p]], np.ones(k) / k, mode="valid")
k = max(3, int(SMOOTH_M / STEP_M))
xs, ys = smooth(xs, k), smooth(ys, k)

col = (xs - ox) / res - 0.5
row = H - 0.5 - (ys - oy) / res
w = np.minimum(ndimage.map_coordinates(np.minimum(d_in, d_out), [row, col], order=1), MAX_HALF_W)

L = np.hypot(np.diff(np.r_[xs, xs[0]]), np.diff(np.r_[ys, ys[0]])).sum()
print("length %.2f m, points %d, full width min/max %.2f/%.2f m" % (L, len(xs), 2 * w.min(), 2 * w.max()))

np.savetxt(OUT, np.c_[xs, ys, w, w], delimiter=",", fmt="%.4f", header="x_m,y_m,w_tr_right_m,w_tr_left_m")
png = f"centerline_{name}.png"
plt.figure(figsize=(14, 6))
plt.imshow(img, cmap="gray", extent=[ox, ox + W * res, oy, oy + H * res], origin="upper")
plt.plot(xs, ys, "r-", lw=1.2); plt.plot(xs[0], ys[0], "yo")
plt.savefig(png, dpi=110, bbox_inches="tight")
print("중심선 확인 그림:", png)

# ---- post-process: smooth centerline + cap width by local radius (avoid crossed normals)
import sys as _sys
_name = _sys.argv[1]
_f = 'inputs/tracks/%s.csv' % _name
_d = np.loadtxt(_f, delimiter=',', comments='#')
_w = 7
for _c in (0, 1):
    _p = np.r_[_d[-_w:, _c], _d[:, _c], _d[:_w, _c]]
    _d[:, _c] = np.convolve(_p, np.ones(2 * _w + 1) / (2 * _w + 1), mode='same')[_w:-_w]
_xy = _d[:, :2]; _n = len(_d)
_dx = (np.roll(_xy, -1, 0) - np.roll(_xy, 1, 0)) / 2
_dd = np.roll(_xy, -1, 0) - 2 * _xy + np.roll(_xy, 1, 0)
_k = np.abs(_dx[:, 0] * _dd[:, 1] - _dx[:, 1] * _dd[:, 0]) / np.maximum(np.hypot(_dx[:, 0], _dx[:, 1]) ** 3, 1e-9)
_r = 1 / np.maximum(_k, 1e-3)
_rmin = np.array([_r[np.arange(i - 10, i + 11) % _n].min() for i in range(_n)])
_cap = np.clip(0.8 * _rmin, 0.35, 0.60)
_d[:, 2] = np.minimum(_d[:, 2], _cap); _d[:, 3] = np.minimum(_d[:, 3], _cap)
np.savetxt(_f, _d, delimiter=',', fmt='%.6f', header='x_m,y_m,w_tr_right_m,w_tr_left_m')
print('post-process: min radius %.2f m, width %.2f..%.2f m' % (_rmin.min(), _d[:, 2].min(), _d[:, 2].max()))
