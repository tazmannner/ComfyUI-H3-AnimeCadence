"""Cadence nodes: turn smooth AI video into held drawings (ones/twos/threes/adaptive).

Video models "hedge" between poses: under uncertainty the lowest-error frame is a blend,
so fast motion becomes a smear and the whole clip moves on ones like CG.  Hand-drawn
anime holds a drawing for 2-3 frames and then snaps.  These nodes impose that cadence
after decode, and (pick="sharpest") keep the crispest frame of each hold so the blended
in-between frames are simply never shown.
"""
import torch
import torch.nn.functional as F


def _gray_small(images, size=128):
    # images: (T,H,W,C) float 0..1  ->  (T,h,w) on images.device
    x = images[..., :3].mean(-1, keepdim=False)[:, None]
    T, _, H, W = x.shape
    s = size / max(H, W)
    h, w = max(8, int(H * s)), max(8, int(W * s))
    return F.interpolate(x, size=(h, w), mode="area")[:, 0]


def _sharpness(images, size=384):
    """Laplacian variance per frame (higher = crisper lines, lower = smeared/ghosted)."""
    x = images[..., :3].mean(-1)[:, None]
    T, _, H, W = x.shape
    s = min(1.0, size / max(H, W))
    if s < 1.0:
        x = F.interpolate(x, size=(int(H * s), int(W * s)), mode="area")
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype, device=x.device)[None, None]
    lap = F.conv2d(x, k)
    return lap.flatten(1).var(dim=1)


def _region_diff(a, b, q=0.9):
    """change between two small gray frames, robust to static backgrounds: mean of the
    top (1-q) fraction of per-pixel differences, so a small moving character still counts."""
    d = (a - b).abs().flatten()
    k = max(1, int(d.numel() * (1 - q)))
    return d.topk(k).values.mean().item()


def token_twos_starts(T):
    """Drawing starts aligned to H3's latent token grid: every 17 frames are 5 tokens
    covering (1,4,4,4,4) frames, so pairs (1,2),(3,4)..(15,16) each sit inside one token and
    the 17k singleton joins the previous pair as a 3-hold."""
    starts = [0]
    for f in range(3, T):
        if f % 17 % 2 == 1:
            starts.append(f)
    return starts


def token_fours_starts(T):
    """One drawing per H3 latent token: [0-4],[5-8],[9-12],[13-16],[17-21],... (the 17k singleton
    joins the following 4-frame token)."""
    return [0] + [f for f in range(5, T) if f % 17 in (0, 5, 9, 13)]


def plan_holds(images, mode="twos", min_hold=2, max_hold=4, sensitivity=1.0, offset=0):
    """Return list of drawing start indices."""
    T = images.shape[0]
    if mode == "ones":
        return list(range(T))
    if mode in ("twos", "threes", "fours"):
        n = {"twos": 2, "threes": 3, "fours": 4}[mode]
        starts = list(range(offset % n, T, n))
        if starts[0] != 0:
            starts = [0] + starts
        return starts
    if mode == "h3_token_twos":
        return token_twos_starts(T)
    if mode == "h3_token_fours":
        return token_fours_starts(T)
    if mode in ("auto_twos", "auto_threes", "auto"):
        # follow the cadence the clip already has: pick the (hold, phase) whose hold slots contain
        # the least motion (changes should land on slot boundaries, not inside them)
        n_opts = {"auto_twos": (2,), "auto_threes": (3,), "auto": (2, 3)}[mode]
        g = _gray_small(images)
        d = torch.tensor([0.0] + [_region_diff(g[t], g[t - 1]) for t in range(1, T)])
        tot = d.sum().item() + 1e-6
        best = None
        for n in n_opts:
            for off in range(n):
                inside = sum(d[t].item() for t in range(1, T) if (t - off) % n != 0) / tot
                # a threes grid hides more frames, so it must fit clearly better to win
                score = inside * (1.25 if n == 3 else 1.0)
                if best is None or score < best[0]:
                    best = (score, n, off)
        _, n, off = best
        starts = list(range(off, T, n))
        return starts if starts and starts[0] == 0 else [0] + starts
    # adaptive: new drawing when the source has moved far enough from the held drawing
    g = _gray_small(images)
    steps = torch.tensor([_region_diff(g[t], g[t - 1]) for t in range(1, T)]) if T > 1 else torch.zeros(1)
    # threshold relative to this clip's own motion: the typical size of one *moving* step
    # (the source may already contain holds, so ignore near-zero steps when estimating it)
    hi = torch.quantile(steps, 0.9).item() if steps.numel() > 3 else steps.max().item()
    moving = steps[steps > 0.3 * hi]
    ref = moving.median().item() if moving.numel() else hi
    thr = max(ref * 1.2 / max(sensitivity, 1e-3), 1.5 / 255)
    starts, s = [0], 0
    for t in range(1, T):
        held = t - s
        if held < min_hold:
            continue
        if held >= max_hold or _region_diff(g[t], g[s]) > thr:
            starts.append(t)
            s = t
    return starts


def apply_holds(images, starts, pick="sharpest", sharp=None):
    T = images.shape[0]
    bounds = list(starts) + [T]
    out = images.clone()
    chosen = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if pick == "first":
            i = a
        elif pick == "middle":
            i = (a + b - 1) // 2
        else:  # sharpest
            i = a + int(torch.argmax(sharp[a:b]).item())
        out[a:b] = images[i]
        chosen.append(i)
    return out, chosen


def runs_str(starts, T):
    b = list(starts) + [T]
    runs = [y - x for x, y in zip(b[:-1], b[1:])]
    hist = {}
    for r in runs:
        hist[r] = hist.get(r, 0) + 1
    return runs, ", ".join(f"{k}s x{v}" for k, v in sorted(hist.items()))


class AnimeCadence:
    """Impose animation cadence on decoded frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "mode": (["adaptive", "twos", "threes", "fours", "h3_token_twos", "h3_token_fours", "auto", "auto_twos", "auto_threes", "ones"], {"default": "adaptive",
                     "tooltip": "adaptive: hold until the picture has moved enough (fast action -> 2s, slow -> 3s/4s). "
                                "h3_token_twos: twos aligned to H3 latent tokens (use before a v2v re-encode)."}),
            "pick": (["sharpest", "first", "middle"], {"default": "sharpest",
                     "tooltip": "which source frame becomes the held drawing. sharpest = highest line contrast, "
                                "skips smeared/ghosted in-betweens"}),
        }, "optional": {
            "min_hold": ("INT", {"default": 2, "min": 1, "max": 8}),
            "max_hold": ("INT", {"default": 4, "min": 1, "max": 24}),
            "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05,
                            "tooltip": "adaptive only. higher = new drawings sooner (closer to min_hold)"}),
            "offset": ("INT", {"default": 0, "min": 0, "max": 3, "tooltip": "phase for fixed modes"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "report", "drawing_starts")
    FUNCTION = "run"
    CATEGORY = "AnimeCadence"

    def run(self, images, mode, pick, min_hold=2, max_hold=4, sensitivity=1.0, offset=0):
        T = images.shape[0]
        starts = plan_holds(images, mode, min_hold, max_hold, sensitivity, offset)
        sharp = _sharpness(images) if pick == "sharpest" else None
        out, chosen = apply_holds(images, starts, pick, sharp)
        _, h = runs_str(starts, T)
        report = f"{T} frames -> {len(starts)} drawings ({len(starts) / T * 24:.1f} drawings/s @24fps). holds: {h}"
        return (out, report, ",".join(map(str, starts)))


class AnimeCadenceMeasure:
    """Measure the cadence a clip actually has (held-frame run lengths, relative threshold)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "AnimeCadence"
    OUTPUT_NODE = True

    def run(self, images):
        g = _gray_small(images, 256)
        T = g.shape[0]
        d = torch.tensor([_region_diff(g[t], g[t - 1]) for t in range(1, T)])
        thr = max(0.25 * torch.quantile(d, 0.9).item(), 0.6 / 255)
        changed = (d > thr).tolist()
        runs, cur = [], 1
        for c in changed:
            if c:
                runs.append(cur)
                cur = 1
            else:
                cur += 1
        runs = runs[1:]
        hist = {}
        for r in runs:
            if r < 9:
                hist[min(r, 5)] = hist.get(min(r, 5), 0) + 1
        tot = sum(hist.values()) or 1
        s = " ".join(f"{k}s:{hist.get(k, 0) / tot:.2f}" for k in range(1, 6))
        txt = f"cadence over {T} frames: {s}"
        return {"ui": {"text": [txt]}, "result": (txt,)}
