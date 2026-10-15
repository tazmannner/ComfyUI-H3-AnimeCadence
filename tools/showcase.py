#!/usr/bin/env python3
"""Before/after showcase for the Anime Cadence node, built for phone viewing.
Two panels (left = raw H3, right = after Anime Cadence) with a drawing-change ticker under each:
a dot that lights when the picture actually changes, a strip of the last 48 frames, and a live
"drawings per second" counter. Sections: real speed, half speed, then frame-by-frame steps through
the busiest second.
  showcase.py out.mp4 left.mp4 right.mp4 [--left-label ..] [--right-label ..] [--title ..]
"""
import argparse, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(__file__))
import cadence_analyze as cadence

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PW, PH = 640, 360          # panel size
TICK_H = 92                # ticker area under each panel
TITLE_H = 54


def rgb_frames(path, w=PW, h=PH):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def changes(path):
    """bool per frame: is this frame a new drawing (vs the previous frame)?"""
    g = cadence.read_frames(path)
    d = np.abs(g[1:] - g[:-1]).mean(axis=(1, 2))
    thr = max(0.25 * np.percentile(d, 90), 0.6)
    return np.concatenate([[True], d > thr])


def draw_ticker(dr, x0, y0, ch, t, label, font_s, font_b, accent):
    # live dot
    on = ch[t]
    dr.ellipse([x0 + 14, y0 + 14, x0 + 44, y0 + 44], fill=accent if on else (60, 60, 60))
    # drawings/sec over the trailing second
    a = max(0, t - 23)
    dps = int(ch[a:t + 1].sum() * 24 / (t + 1 - a))
    dr.text((x0 + 56, y0 + 10), f"{dps:2d} drawings/sec", fill=(240, 240, 240), font=font_b)
    dr.text((x0 + 56, y0 + 40), label, fill=(170, 170, 170), font=font_s)
    # strip of the last 48 frames: tall bar = new drawing, short = held
    n, bw = 48, (PW - 28) / 48
    for i in range(n):
        f = t - (n - 1 - i)
        if f < 0:
            continue
        hh = 22 if ch[f] else 5
        x = x0 + 14 + i * bw
        col = accent if ch[f] else (90, 90, 90)
        if f == t:
            col = (255, 255, 255)
        dr.rectangle([x, y0 + 86 - hh, x + bw - 2, y0 + 86], fill=col)


def compose(L, R, chL, chR, t, title, sub, labels, fonts):
    W, H = PW * 2 + 12, TITLE_H + PH + TICK_H
    img = Image.new("RGB", (W, H), (16, 16, 18))
    dr = ImageDraw.Draw(img)
    dr.text((12, 6), title, fill=(255, 230, 80), font=fonts[2])
    dr.text((12, 32), sub, fill=(200, 200, 200), font=fonts[0])
    img.paste(Image.fromarray(L[t]), (0, TITLE_H))
    img.paste(Image.fromarray(R[t]), (PW + 12, TITLE_H))
    dr.text((8, TITLE_H + 6), labels[0], fill=(255, 255, 255), font=fonts[1], stroke_width=2, stroke_fill=(0, 0, 0))
    dr.text((PW + 20, TITLE_H + 6), labels[1], fill=(255, 255, 255), font=fonts[1], stroke_width=2, stroke_fill=(0, 0, 0))
    dr.text((W - 92, TITLE_H + PH - 26), f"f{t:03d}", fill=(255, 220, 0), font=fonts[0], stroke_width=2, stroke_fill=(0, 0, 0))
    draw_ticker(dr, 0, TITLE_H + PH, chL, t, "red bar = picture changed this frame", fonts[0], fonts[1], (255, 90, 90))
    draw_ticker(dr, PW + 12, TITLE_H + PH, chR, t, "green bar = new drawing, gap = drawing held", fonts[0], fonts[1], (90, 220, 120))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--left-label", default="raw H3")
    ap.add_argument("--right-label", default="+ Anime Cadence (our node)")
    ap.add_argument("--title", default="Same render. Only the timing changed.")
    ap.add_argument("--realtime-only", action="store_true", help="skip the half-speed and frame-by-frame sections")
    a = ap.parse_args()
    global PH  # keep the source aspect ratio (panel width fixed)
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", a.left], capture_output=True, text=True).stdout.strip().split(",")
    PH = int(round(PW * int(wh[1]) / int(wh[0]) / 2)) * 2
    L, R = rgb_frames(a.left, PW, PH), rgb_frames(a.right, PW, PH)
    T = min(len(L), len(R))
    chL, chR = changes(a.left)[:T], changes(a.right)[:T]
    fonts = (ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 24))
    labels = (a.left_label, a.right_label)
    # busiest second = window with most raw changes weighted by motion
    g = cadence.read_frames(a.left)
    mot = np.concatenate([[0], np.abs(g[1:] - g[:-1]).mean(axis=(1, 2))])[:T]
    s0 = int(np.argmax(np.convolve(mot, np.ones(24), "valid"))) if T > 24 else 0
    plan = [(t, 1, "real speed") for t in range(T)]
    if not a.realtime_only:
        plan += [(t, 2, "half speed") for t in range(T)]
    plan += [] if a.realtime_only else [(t, 8, f"frame by frame, frames {s0}-{s0 + 23}: right side repeats each drawing") for t in range(s0, min(T, s0 + 24))]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{PW * 2 + 12}x{TITLE_H + PH + TICK_H}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for t, rep, sub in plan:
        fr = np.asarray(compose(L, R, chL, chR, t, a.title, sub, labels, fonts))
        for _ in range(rep):
            p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    print(f"wrote {a.out}  raw drawings/s={chL.sum() * 24 / T:.1f}  cadence drawings/s={chR.sum() * 24 / T:.1f}")


if __name__ == "__main__":
    main()
