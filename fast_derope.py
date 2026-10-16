"""Helpers for the cheap ("scout") de-rope recipe.

The de-rope pass 2 starts at sigma ~0.92 (inject 0.5 on a shift-12 schedule), so its init latent
only has to carry layout and choreography. The full H3 VAE round trip on the dilated clip
(decode pass 1, encode ~2.3x as many frames) is wasted precision there; the tiny taeh3
autoencoder does the same job ~10x faster. The final decode still uses the full VAE.
"""
import logging

import comfy.sd
import comfy.utils
import folder_paths


class AnimeCadenceTAELoader:
    """Load a video TAE from models/vae_approx as a normal VAE (stock VAELoader only accepts the
    exact file name taeh3.safetensors there, and that file is a decoder-only preview model)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "tae_name": (folder_paths.get_filename_list("vae_approx"),
                         {"tooltip": "madebyollin taeh3 with encoder + decoder (e.g. taeh3(1).safetensors)"}),
        }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "AnimeCadence"
    DESCRIPTION = ("Loads the tiny H3 autoencoder as a VAE for VAEEncode / VAEDecode. Use it for the "
                   "throwaway steps of a de-rope (decode the scout pass, encode the time-smeared init); "
                   "keep the full H3 VAE for the final decode.")

    _cache = {}

    def load(self, tae_name):
        if tae_name not in self._cache:
            sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", tae_name))
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            type(self)._cache = {tae_name: vae}
        return (self._cache[tae_name],)


class AnimeCadenceScoutSigmas:
    """Cut pass 1's sigma schedule so it stops right after the model has predicted x0 at x0_sigma.

    The de-rope pass 2 starts at sigma ~0.92 (inject 0.5, 8-step turbo schedule, shift 12) and only
    keeps layout + choreography from its init, which a same-seed run has already decided by then.
    Wire the output into pass 1's SamplerCustomAdvanced and take its *denoised_output* (x0) for the
    Jerk Oracle and the time smear: same acting as the full render, for about half its steps.
    20-step beta57 at shift 12 -> 10 steps for x0_sigma 0.91 (last x0 at sigma 0.900)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "x0_sigma": ("FLOAT", {"default": 0.91, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "stop once the model has been evaluated at a sigma <= this. "
                                              "Lower = sharper scout (more steps), higher = cheaper. "
                                              "0.91 (= 10 of 20 beta57 steps) matched the full render's choreography in our tests"}),
        }}

    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("sigmas", "steps")
    FUNCTION = "split"
    CATEGORY = "AnimeCadence"
    DESCRIPTION = ("Stops pass 1 early for a cheap de-rope: use SamplerCustomAdvanced's denoised_output "
                   "(x0 prediction) as the de-rope source. 'steps' = how many steps will run.")

    def split(self, sigmas, x0_sigma):
        s = sigmas.flatten()
        n = s.shape[0] - 1  # steps in the full schedule
        i = next((k for k in range(n) if float(s[k]) <= x0_sigma), n - 1)  # first evaluated sigma <= target
        logging.info(f"ScoutSigmas: running {i + 1} of {n} steps, x0 predicted at sigma {float(s[i]):.3f}")
        return (sigmas[:i + 2], i + 1)
