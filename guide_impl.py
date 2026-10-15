"""Hot-reloadable body of AnimeCadenceH3HoldGuide (reloaded every time the node runs)."""
import logging
import torch
import comfy.model_management as mm

from . import cadence

log = logging.getLogger("AnimeCadence")


def make_hold(pix, starts, pick, sharp=None):
    F_ = pix.shape[2]
    bounds = list(starts) + [F_]
    held = pix.clone()
    for a, c in zip(bounds[:-1], bounds[1:]):
        if pick == "first":
            i = a
        elif pick == "middle":
            i = (a + c - 1) // 2
        else:
            i = a + int(torch.argmax(sharp[a:c]))
        held[:, :, a:c] = pix[:, :, i:i + 1]
    return held


def guide_step(args, cfg, state):
    den = args["denoised"]
    sigma = float(args["sigma"].flatten()[0])
    if cfg["strength"] == 0 or not (cfg["s_lo"] <= sigma <= cfg["s_hi"]):
        return den
    T, h, w = cfg["T"], cfg["h"], cfg["w"]
    n_vid = 24 * T * h * w
    B = den.shape[0]
    flat = den.reshape(B, -1)
    if flat.shape[1] < n_vid:
        log.warning("HoldGuide: latent smaller than width/height/length implies; skipping")
        return den
    v = flat[:, :n_vid].reshape(B, 24, T, h, w)
    tae = cfg["tae"]
    mm.load_models_gpu([tae.patcher])
    enc = tae.first_stage_model
    dev = mm.get_torch_device()
    dt = tae.vae_dtype
    opts = cfg["opts"]
    mode = opts.get("mode", cfg["mode"])
    strength = cfg["strength"]
    # phases: [[end_percent, mode, strength?], ...] -> first phase whose end sigma is below current sigma
    if opts.get("phases"):
        ms = args["model"].model_sampling
        for ph in sorted(opts["phases"], key=lambda p: p[0]):
            if sigma >= float(ms.percent_to_sigma(ph[0])):
                mode = ph[1]
                if len(ph) > 2:
                    strength = ph[2]
                break
        else:
            return den
    with torch.no_grad():
        out_v = []
        for b in range(B):
            vb = v[b:b + 1].to(dev, dt)
            pix = enc.decode(vb).to(dev, dt)  # [1,3,F,H,W] 0..1
            frames = pix[0].permute(1, 2, 3, 0).float()
            if mode == "adaptive" and opts.get("freeze_plan", True) and state.get("mode") == mode:
                starts = state["starts"]
            else:
                state["mode"] = mode
                starts = cadence.plan_holds(frames, mode, opts.get("min_hold", 2), opts.get("max_hold", 4),
                                            opts.get("sensitivity", 1.0))
                state["starts"] = starts
            sharp = cadence._sharpness(frames) if cfg["pick"] == "sharpest" else None
            held = make_hold(pix, starts, cfg["pick"], sharp)
            if opts.get("direct", False):
                target = enc.encode(held)
                d = target[:, :, :T] - vb
            else:
                d = (enc.encode(held) - enc.encode(pix))[:, :, :T]
            out_v.append(v[b:b + 1] + strength * d.to(v.device, v.dtype))
        v_new = torch.cat(out_v, 0)
    state["calls"] = state.get("calls", 0) + 1
    if state["calls"] == 1:
        log.info(f"HoldGuide active from sigma {sigma:.3f}: {len(starts)} drawings over {frames.shape[0]} frames")
    flat = flat.clone()
    flat[:, :n_vid] = v_new.reshape(B, -1)
    return flat.reshape(den.shape)
