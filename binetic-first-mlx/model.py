"""Bonic-First model for mlx-lm compatibility.

Enables `mlx_lm.load("binetic-first-mlx")` and `mlx_lm.generate()` to work
with the Bonsai 2 ternary packed weights format.

mlx-lm's load_model() pipeline:
  1. Reads config.json (looks for "model_file" key)
  2. If model_file exists, imports Model + ModelArgs classes from that file
  3. ModelArgs.from_dict(config)  (filters to known dataclass fields)
  4. Model(model_args)
  5. model.sanitize(weights) if it exists  ← we transform packed -> unpacked
  6. model.load_weights(sanitized_weights)

Our sanitize() mirrors bonsai2-27b-mlx/runtime/artifact.py::load_model():
  - Reads modules config from config.json
  - Instantiates Packed modules and sets them on the model
  - Returns non-packed weights for standard load_weights

NOTE: Hardware target is M3 Max MacBook (64 GB unified memory).
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

# Import the Packed module from the Bonsai runtime
RUNTIME_DIR = Path(__file__).resolve().parent.parent / "bonsai2-27b-mlx" / "runtime"
sys.path.insert(0, str(RUNTIME_DIR))
from runtime import Packed

_PACKED_SUFFIXES = ("weight", "scales", "biases")


class Model(TextModel):
    """Bonic-First model with packed ternary weights."""

    def sanitize(self, weights: dict) -> dict:
        """Transform packed weights into standard format.

        mlx-lm calls this after loading safetensors. Reads the modules
        config from config.json (alongside this file), instantiates Packed
        modules, attaches them to the model, and returns only the
        non-packed weights for standard load_weights.
        """
        config_path = Path(__file__).parent / "config.json"
        config = json.loads(config_path.read_text())

        modules = config.get("modules", [])
        weight_prefix = ""
        if any(k.startswith("language_model.") for k in weights):
            weight_prefix = "language_model."

        for record in modules:
            path = record["path"]

            def _get(suffix):
                for key in (
                    weight_prefix + path + "." + suffix,
                    path + "." + suffix,
                ):
                    if key in weights:
                        return weights[key]
                raise KeyError(f"Cannot find packed tensor: {path}.{suffix}")

            arrays = [_get(s) for s in _PACKED_SUFFIXES]

            signs = None
            if record["block"]:
                for key in (
                    weight_prefix + path + ".signs",
                    path + ".signs",
                ):
                    if key in weights:
                        signs = weights[key]
                        break

            dtype = mx.float16
            parts = path.split(".")
            parent = self
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)

            packed = Packed(
                arrays=[np.asarray(a) for a in arrays],
                block=record["block"],
                signs=np.asarray(signs) if signs is not None else None,
                embedding=record["embedding"],
                dtype=dtype,
            )
            setattr(parent, parts[-1], packed)

        # Return only non-packed weights (stripped of weight_prefix)
        packed_prefixes = [m["path"] for m in modules]
        packed_suffixes_full = _PACKED_SUFFIXES + ("signs",)
        result = {}
        for key, value in weights.items():
            stripped = key
            if weight_prefix and key.startswith(weight_prefix):
                stripped = key[len(weight_prefix):]

            is_packed = any(
                stripped == f"{prefix}.{s}" or stripped.startswith(f"{prefix}.")
                for prefix in packed_prefixes
                for s in packed_suffixes_full
            )
            if not is_packed:
                # Drop vision tower weights; strip prefix from language model weights
                if weight_prefix and key.startswith(weight_prefix):
                    result[stripped] = value
                elif not key.startswith("vision_tower"):
                    result[key] = value

        return result


class ModelArgs(TextModelArgs):
    """Model arguments for Bonic-First (Qwen3.5 compatible).

    mlx-lm calls ModelArgs.from_dict(config) with the FULL config dict
    (not just text_config), so we must extract text_config first.
    """

    @classmethod
    def from_dict(cls, params):
        """Extract text_config from the full config, then pass to parent."""
        if "text_config" in params:
            params = params["text_config"]
        return super().from_dict(params)
