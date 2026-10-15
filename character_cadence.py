"""Character-only cadence: hold the CHARACTER on 2s/3s while the background / camera keeps moving
every frame (how real anime is shot: cel held, background pans on ones).

AnimeCadenceMaskTrack       - per-frame character mask with the local SAM (sam_vit_b in models/sams),
                            tracked frame to frame (previous mask -> box + point prompt). No downloads.
AnimeCadenceCharacter - for each hold slot, paste the held drawing of the character (plus a
                            feathered halo) over the CURRENT frame's background.
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import cadence

_SAM_CACHE = {}


def _load_sam(name, device):
    import folder_paths
    from segment_anything import sam_model_registry, SamPredictor
    key = (name, str(device))
    if key not in _SAM_CACHE:
        kind = "vit_h" if "vit_h" in name else "vit_l" if "vit_l" in name else "vit_b"
        sam = sam_model_registry[kind](checkpoint=folder_paths.get_full_path_or_raise("sams", name))
        sam.to(device).eval()
        _SAM_CACHE.clear()
        _SAM_CACHE[key] = SamPredictor(sam)
    return _SAM_CACHE[key]


def _bbox(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def _inner_point(m):
    """a point well inside the mask (max of a cheap distance proxy), robust for non-convex shapes"""
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    best = t
    for _ in range(8):
        t = -F.max_pool2d(-t, 5, 1, 2)  # erode
        if t.sum() < 20:
            break
        best = t
    ys, xs = np.nonzero(best[0, 0].numpy() > 0.5)
    if len(xs) == 0:
        ys, xs = np.nonzero(m)
    order = np.lexsort((xs, ys))
    i = order[len(order) // 2]
    return xs[i], ys[i]


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def track(images, sam_name, x, y, device=None, expand=0.15, progress=None):
    """images: (T,H,W,3) float 0..1. x,y: fractions of the frame for the first-frame click."""
    import comfy.model_management as mm
    device = device or mm.get_torch_device()
    pred = _load_sam(sam_name, device)
    T, H, W, _ = images.shape
    masks = np.zeros((T, H, W), bool)
    prev = None
    for t in range(T):
        img = (images[t].cpu().numpy() * 255).astype(np.uint8)
        pred.set_image(img)
        if prev is None:
            pts = np.array([[x * W, y * H]])
            out, sc, _ = pred.predict(point_coords=pts, point_labels=np.array([1]), multimask_output=True)
            # prefer the largest plausible mask (whole person, not just the shirt)
            cand = [(m.sum() * (0.5 + s), m) for m, s in zip(out, sc) if 0.005 < m.mean() < 0.6]
            m = max(cand, key=lambda c: c[0])[1] if cand else out[int(np.argmax(sc))]
        else:
            bx = _bbox(prev)
            px, py = _inner_point(prev)
            if bx is None:
                m = prev
            else:
                x0, y0, x1, y1 = bx
                dx, dy = (x1 - x0) * expand, (y1 - y0) * expand
                box = np.array([max(0, x0 - dx), max(0, y0 - dy), min(W - 1, x1 + dx), min(H - 1, y1 + dy)])
                # positive clicks spread over the body: an inner point in the top, middle and bottom third
                pts = []
                for k in range(3):
                    band = np.zeros_like(prev)
                    ya, yb = y0 + (y1 - y0) * k // 3, y0 + (y1 - y0) * (k + 1) // 3 + 1
                    band[ya:yb] = prev[ya:yb]
                    if band.sum() > 30:
                        pts.append(_inner_point(band))
                pts = np.array(pts or [(px, py)], dtype=np.float32)
                lab = np.ones(len(pts), dtype=int)
                cands = []
                for use_box in (True, False):
                    out, sc, _ = pred.predict(point_coords=pts, point_labels=lab,
                                              box=box[None] if use_box else None, multimask_output=True)
                    cands += list(zip(out, sc))
                pa = prev.sum()
                ok = [(m_, s_) for m_, s_ in cands
                      if 0.8 * pa < m_.sum() < 1.4 * pa and _iou(m_, prev) > 0.5]
                # temporal consistency: best IoU with the previous mask, weighted by SAM's score;
                # if SAM only offers implausible jumps (character fills the frame, background grabbed), keep prev
                m = max(ok, key=lambda c: _iou(c[0], prev) * (0.5 + c[1]))[0] if ok else prev
        masks[t] = m
        prev = m
        if progress:
            progress(t)
    return torch.from_numpy(masks.astype(np.float32))


def _grow_feather(m, grow, feather):
    m = m[:, None]
    if grow > 0:
        k = grow * 2 + 1
        m = F.max_pool2d(m, k, 1, grow)
    if feather > 0:
        k = feather * 2 + 1
        m = F.avg_pool2d(F.pad(m, (feather,) * 4, mode="replicate"), k, 1)
    return m[:, 0].clamp(0, 1)


def character_cadence(images, masks, mode="auto", pick="sharpest", grow=6, feather=8):
    T = images.shape[0]
    masks = masks.to(images.device).float()
    if masks.shape[1:] != images.shape[1:3]:
        masks = F.interpolate(masks[:, None], size=images.shape[1:3], mode="bilinear")[:, 0]
    # plan holds from the character's own motion (background removed)
    char_only = images * masks[..., None]
    starts = cadence.plan_holds(char_only, mode)
    sharp = cadence._sharpness(char_only) if pick == "sharpest" else None
    bounds = list(starts) + [T]
    out = images.clone()
    for a, b in zip(bounds[:-1], bounds[1:]):
        i = a if pick == "first" else (a + b - 1) // 2 if pick == "middle" else a + int(torch.argmax(sharp[a:b]))
        for t in range(a, b):
            if t == i:
                continue
            region = torch.maximum(masks[i], masks[t])[None]          # held char + where the char is now
            M = _grow_feather(region, grow, feather)[0][..., None]
            out[t] = M * images[i] + (1 - M) * images[t]
    return out, starts


class AnimeCadenceMaskTrack:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "images": ("IMAGE",),
            "sam_name": (folder_paths.get_filename_list("sams"),),
            "click_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                        "tooltip": "first-frame click on the character, fraction of width"}),
            "click_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "AnimeCadence"

    def run(self, images, sam_name, click_x, click_y):
        import comfy.utils
        pbar = comfy.utils.ProgressBar(images.shape[0])
        return (track(images, sam_name, click_x, click_y, progress=lambda t: pbar.update(1)),)


class AnimeCadenceCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "mode": (["auto", "auto_twos", "auto_threes", "twos", "threes", "adaptive"], {"default": "auto"}),
            "pick": (["sharpest", "first", "middle"], {"default": "sharpest"}),
            "grow": ("INT", {"default": 6, "min": 0, "max": 64, "tooltip": "px the held character region is grown"}),
            "feather": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "px soft edge into the live background"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "run"
    CATEGORY = "AnimeCadence"

    def run(self, images, masks, mode, pick, grow, feather):
        out, starts = character_cadence(images, masks, mode, pick, grow, feather)
        _, h = cadence.runs_str(starts, images.shape[0])
        return (out, f"character held: {len(starts)} drawings over {images.shape[0]} frames ({h}); background live every frame")
