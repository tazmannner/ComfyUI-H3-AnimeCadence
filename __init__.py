"""ComfyUI-H3-AnimeCadence: experiments for making MiniMax H3 animate like hand-drawn anime
(cadence on twos/threes, less smear/morph distortion in 2D motion)."""
import os
import folder_paths

# SAM v1 checkpoints live in models/sams (same folder Impact Pack uses); register it if nobody else did
if "sams" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("sams", os.path.join(folder_paths.models_dir, "sams"))

from .cadence import AnimeCadence, AnimeCadenceMeasure
from .hold_guide import AnimeCadenceH3HoldGuide
from .character_cadence import AnimeCadenceMaskTrack, AnimeCadenceCharacter

NODE_CLASS_MAPPINGS = {
    "AnimeCadence": AnimeCadence,
    "AnimeCadenceMeasure": AnimeCadenceMeasure,
    "AnimeCadenceH3HoldGuide": AnimeCadenceH3HoldGuide,
    "AnimeCadenceMaskTrack": AnimeCadenceMaskTrack,
    "AnimeCadenceCharacter": AnimeCadenceCharacter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeCadence": "Anime Cadence: Hold on 2s/3s",
    "AnimeCadenceMeasure": "Anime Cadence: Measure",
    "AnimeCadenceH3HoldGuide": "H3 Hold Guide (experimental)",
    "AnimeCadenceMaskTrack": "Character Mask Track (SAM)",
    "AnimeCadenceCharacter": "Anime Cadence: Character Only",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
