"""In-sampler cadence guidance for MiniMax H3.

At each sampling step the model predicts a clean video x0.  We decode that prediction with
the tiny H3 autoencoder (taeh3), replace every hold slot with one drawing (so the prediction
"shows" held drawings that snap), re-encode, and nudge x0 toward it by the *difference*
E(held) - E(pred), which cancels the TAE's own reconstruction bias.  Applied only over an
early/mid sigma window: structure (which pose, when) is decided there, and the remaining
steps refine linework on top of the held poses.
"""
import importlib
import json

import comfy.sd
import comfy.utils
import folder_paths

from . import cadence, guide_impl


def h3_tokens(length):
    return (length - 5) // 17 * 5 + 2


class AnimeCadenceH3HoldGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tae_name": (folder_paths.get_filename_list("vae_approx"), {"tooltip": "madebyollin taeh3 (encoder+decoder)"}),
            "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 32}),
            "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17}),
            "mode": (["h3_token_twos", "auto", "auto_twos", "auto_threes", "twos", "threes", "h3_token_fours", "adaptive"], {"default": "h3_token_twos"}),
            "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "pick": (["first", "sharpest", "middle"], {"default": "first"}),
        }, "optional": {
            "options": ("STRING", {"default": "{}", "tooltip": "JSON extras: min_hold, max_hold, sensitivity, direct, freeze_plan"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "AnimeCadence"

    _tae_cache = {}

    @classmethod
    def load_tae(cls, name):
        if name not in cls._tae_cache:
            sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", name))
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            cls._tae_cache = {name: vae}
        return cls._tae_cache[name]

    def patch(self, model, tae_name, width, height, length, mode, strength, start_percent, end_percent, pick,
              options="{}"):
        importlib.reload(cadence)
        impl = importlib.reload(guide_impl)
        m = model.clone()
        ms = m.get_model_object("model_sampling")
        cfg = {
            "tae": self.load_tae(tae_name), "mode": mode, "strength": strength, "pick": pick,
            "s_hi": float(ms.percent_to_sigma(start_percent)), "s_lo": float(ms.percent_to_sigma(end_percent)),
            "T": h3_tokens(length), "h": height // 16, "w": width // 16,
            "opts": json.loads(options or "{}"),
        }
        state = {}

        def post_cfg(args):
            return impl.guide_step(args, cfg, state)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)
