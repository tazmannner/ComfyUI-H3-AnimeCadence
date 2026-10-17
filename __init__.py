"""ComfyUI-H3-AnimeCadence: make MiniMax H3 video move like hand-drawn anime, animated on 2s/3s.

Core:          Anime Cadence (hold on 2s/3s), Measure, Character Only (any mask)
Optional:      Character Mask Track (SAM v1) -- only registered when `segment-anything` is installed
Experimental:  H3 Hold Guide (in-sampler nudge)
Fast-motion smear (de-rope) lives in its own pack: ComfyUI-H3-FastDerope."""
import importlib.util
import logging

from .cadence import AnimeCadence, AnimeCadenceMeasure
from .character_cadence import AnimeCadenceCharacter
from .hold_guide import AnimeCadenceH3HoldGuide

NODE_CLASS_MAPPINGS = {
    "AnimeCadence": AnimeCadence,
    "AnimeCadenceMeasure": AnimeCadenceMeasure,
    "AnimeCadenceCharacter": AnimeCadenceCharacter,
    "AnimeCadenceH3HoldGuide": AnimeCadenceH3HoldGuide,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeCadence": "Anime Cadence: Hold on 2s/3s",
    "AnimeCadenceMeasure": "Anime Cadence: Measure",
    "AnimeCadenceCharacter": "Anime Cadence: Character Only",
    "AnimeCadenceH3HoldGuide": "H3 Hold Guide (experimental)",
}

# optional SAM v1 tracker: no hard dependency on segment-anything
if importlib.util.find_spec("segment_anything") is not None:
    from .sam_track import AnimeCadenceMaskTrack
    NODE_CLASS_MAPPINGS["AnimeCadenceMaskTrack"] = AnimeCadenceMaskTrack
    NODE_DISPLAY_NAME_MAPPINGS["AnimeCadenceMaskTrack"] = "Character Mask Track (SAM, optional)"
else:
    logging.info("AnimeCadence: segment-anything not installed; the optional Character Mask Track (SAM) node is "
                 "disabled. Character Only works with any mask (e.g. SAM2 nodes).")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
