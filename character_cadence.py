"""Character-only cadence: hold the CHARACTER on 2s/3s while the background / camera keeps moving every frame
(how real anime is shot: cel held, background pans on ones).

AnimeCadenceCharacter - for each hold slot, paste the held drawing of the character (plus a feathered halo) over the
                        CURRENT frame's background. Takes any per-frame MASK: SAM2 nodes, the optional SAM v1 tracker
                        in this pack (sam_track.py), a painted mask, ...
"""
import torch
import torch.nn.functional as F

from . import cadence


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
