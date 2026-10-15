#!/usr/bin/env python3
"""Cadence analyzer: how "on ones / twos / threes" is a clip?

Reads frames with ffmpeg (grayscale, downscaled), measures the per-transition
change, splits transitions into "hold" vs "new drawing" with a threshold that is
relative to the clip's own motion (a held drawing still wobbles from codec/VAE
noise, so a fixed epsilon is wrong), then reports the run-length histogram of
drawings.  Run length 1 = ones, 2 = twos, 3 = threes.

Two views:
  global  - whole-frame diff (camera pans read as ones even when characters are on twos)
  tiles   - per-tile diff on a grid; histogram pooled over tiles that actually move.
            This is closer to what an animator means (character layer cadence).

Usage: cadence.py video.mp4 [--start S] [--dur D] [--json]
"""
import argparse, json, subprocess, sys
import numpy as np

W, H = 256, 144


def read_frames(path, start=None, dur=None, w=W, h=H):
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def runs_from_changes(changed):
    """changed[t] = True if frame t+1 is a new drawing vs frame t. Returns drawing run lengths."""
    runs, cur = [], 1
    for c in changed:
        if c:
            runs.append(cur)
            cur = 1
        else:
            cur += 1
    runs.append(cur)
    # drop the partial first/last runs
    return runs[1:-1] if len(runs) > 2 else []


def classify(d, rel=0.25, floor=0.6, pct=90):
    """d: per-transition diffs. Returns bool changed array and threshold."""
    if len(d) == 0:
        return np.zeros(0, bool), 0.0
    hi = np.percentile(d, pct)
    thr = max(rel * hi, floor)
    return d > thr, thr


def hist(runs, maxk=5):
    h = {k: 0 for k in range(1, maxk + 1)}
    for r in runs:
        if r >= 9:  # long holds = static, not cadence
            continue
        h[min(r, maxk)] += 1
    tot = sum(h.values()) or 1
    return {k: round(v / tot, 3) for k, v in h.items()}, sum(h.values())


def laplacian_var(f):
    lap = (-4 * f[1:-1, 1:-1] + f[:-2, 1:-1] + f[2:, 1:-1] + f[1:-1, :-2] + f[1:-1, 2:])
    return float(lap.var())


def smear_ratio(frames, g):
    """Line sharpness of the fastest-moving frames relative to the calmest ones (1.0 = no smear,
    lower = motion blur / ghosting / melted lines during action). Uses the per-transition motion g."""
    if len(g) < 10:
        return None
    sharp = np.array([laplacian_var(f) for f in frames[1:]])
    order = np.argsort(g)
    k = max(2, len(g) // 5)
    calm, fast = sharp[order[:k]], sharp[order[-k:]]
    return round(float(np.median(fast) / (np.median(calm) + 1e-6)), 3)


def moving_sharpness(path, w=512, h=288, thr=18.0):
    """Mean gradient magnitude inside the regions that are moving (|f_t - f_t-1| > thr, dilated).
    Comparable between variants of the SAME prompt/seed: higher = crisper lines during action,
    lower = motion blur / ghosting / melted shapes. Also returns the 10th percentile (worst frames)."""
    fr = read_frames(path, w=w, h=h)
    vals = []
    for t in range(1, len(fr)):
        d = np.abs(fr[t] - fr[t - 1])
        m = d > thr
        if m.mean() < 0.01:
            continue
        # dilate mask 2px (cheap max filter)
        mm = m.copy()
        for s in (1, 2):
            mm[s:] |= m[:-s]; mm[:-s] |= m[s:]; mm[:, s:] |= m[:, :-s]; mm[:, :-s] |= m[:, s:]
        f = fr[t]
        gx = np.abs(f[:, 1:] - f[:, :-1])[:-1]
        gy = np.abs(f[1:] - f[:-1])[:, :-1]
        g = np.sqrt(gx ** 2 + gy ** 2)
        vals.append(float(g[mm[:-1, :-1]].mean()))
    if not vals:
        return None, None
    return round(float(np.median(vals)), 2), round(float(np.percentile(vals, 10)), 2)


def phase_shift(a, b):
    """integer (dy, dx) that best aligns b onto a (phase correlation)."""
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-9
    r = np.fft.ifft2(R).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    H, W = a.shape

    def para(m1, c, p1):  # sub-pixel parabolic peak offset
        den = m1 - 2 * c + p1
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m1 - p1) / den
    dy = iy + para(r[(iy - 1) % H, ix], r[iy, ix], r[(iy + 1) % H, ix])
    dx = ix + para(r[iy, (ix - 1) % W], r[iy, ix], r[iy, (ix + 1) % W])
    if dy > H // 2: dy -= H
    if dx > W // 2: dx -= W
    return float(dy), float(dx)


def fourier_shift(b, dy, dx):
    H, W = b.shape
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.fftfreq(W)[None, :]
    return np.fft.ifft2(np.fft.fft2(b) * np.exp(-2j * np.pi * (ky * dy + kx * dx))).real


def comp_diffs(frames, max_shift=24, border=24):
    """|f_t - shift(f_{t-1})| after removing global camera translation; border cropped."""
    out = []
    for t in range(1, len(frames)):
        a, b = frames[t], frames[t - 1]
        dy, dx = phase_shift(a, b)
        if abs(dy) > max_shift or abs(dx) > max_shift:
            dy = dx = 0
        bs = fourier_shift(b, dy, dx)
        d = np.abs(a - bs)
        out.append(d[border:-border, border:-border])
    return np.stack(out)


def analyze(frames, grid=(6, 8), rel=0.25, camera_comp=False):
    diffs = comp_diffs(frames) if camera_comp else np.abs(frames[1:] - frames[:-1])  # (T-1,H,W)
    g = diffs.mean(axis=(1, 2))
    ch, thr = classify(g, rel=rel)
    gruns = runs_from_changes(ch)
    gh, gn = hist(gruns)

    gy, gx = grid
    T1, Hh, Ww = diffs.shape
    th, tw = Hh // gy, Ww // gx
    tiles = diffs[:, :th * gy, :tw * gx].reshape(T1, gy, th, gx, tw).mean(axis=(2, 4))
    pooled = []
    moving_tiles = 0
    for i in range(gy):
        for j in range(gx):
            d = tiles[:, i, j]
            if np.percentile(d, 90) < 2.0:  # tile basically static
                continue
            moving_tiles += 1
            c, _ = classify(d, rel=rel, floor=1.0)
            pooled += runs_from_changes(c)
    th_, tn = hist(pooled)

    def summary(h):
        mode = max(h, key=h.get) if any(h.values()) else 0
        avg = sum(k * v for k, v in h.items())
        return mode, round(avg, 2)

    gm, ga = summary(gh)
    tm, ta = summary(th_)
    smear = smear_ratio(frames, g)
    return {
        "smear": smear,
        "frames": int(frames.shape[0]),
        "global": {"hist": gh, "n_runs": gn, "mode": gm, "mean_run": ga, "thr": round(float(thr), 2),
                   "mean_diff": round(float(g.mean()), 2)},
        "tiles": {"hist": th_, "n_runs": tn, "mode": tm, "mean_run": ta, "moving_tiles": moving_tiles},
        "diff_series": [round(float(x), 2) for x in g],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--series", action="store_true", help="print per-transition diff series")
    a = ap.parse_args()
    fr = read_frames(a.video, a.start, a.dur)
    r = analyze(fr)
    if a.json:
        print(json.dumps(r))
        return
    print(f"{a.video}  frames={r['frames']}")
    for k in ("global", "tiles"):
        v = r[k]
        hs = " ".join(f"{n}s:{p:.2f}" for n, p in v["hist"].items())
        print(f"  {k:6s} mode={v['mode']} mean_run={v['mean_run']} runs={v['n_runs']}  [{hs}]"
              + (f" thr={v['thr']} mean_diff={v['mean_diff']}" if k == "global" else f" moving_tiles={v['moving_tiles']}"))
    if a.series:
        print("  series:", " ".join(f"{x:.1f}" for x in r["diff_series"]))


if __name__ == "__main__":
    main()
