"""Internal save specification for ndevio OME-Zarr writers.

A :class:`LayerSaveSpec` is the normalized, validated form of one napari
layer.  The normalization step (``_save_normalize``) is responsible for
converting the loose FullLayerData tuple into this typed structure before
it is handed to the writer backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayerSaveSpec:
    """Normalized representation of one napari layer for writing."""

    data_levels: tuple[Any, ...]  # level 0 first (highest resolution)
    layer_type: str
    name: str
    scale: tuple[float, ...]
    axis_labels: tuple[str, ...]
    units: tuple[str | None, ...]
    # Display, written to omero.channels (images) or image-label (labels).
    color: str | None = None  # 'RRGGBB', the colormap's end colour
    contrast_limits: tuple[float, float] | None = None
    visible: bool = True
    label_colors: dict[int, tuple[int, int, int, int]] = field(
        default_factory=dict
    )
