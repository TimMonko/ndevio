"""Normalize napari FullLayerData tuples into :class:`LayerSaveSpec` objects.

A FullLayerData tuple is ``(data, attributes, layer_type)`` where ``data``
is either a single array (single-resolution) or a list of arrays ordered
from highest to lowest resolution (multiscale).  This module converts
that loose structure into a typed, validated :class:`LayerSaveSpec`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._save_model import LayerSaveSpec

SUPPORTED_LAYER_TYPES = frozenset({'image', 'labels'})

# Axes a layer may have; the writer adds C in front when it stacks layers.
_DEFAULT_AXES = ('T', 'Z', 'Y', 'X')

# napari names unnamed axes '-1', '-2', ...
_PLACEHOLDER_AXIS = re.compile(r'^-\d+$')

# napari's default unit says nothing physical.
_NO_UNIT = frozenset({'pixel', 'px', ''})

_HEX = re.compile(r'^#?([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$')


class SaveNormalizationError(ValueError):
    """Raised when napari layer data cannot be normalized for saving."""


def normalize_layer_data(
    layer_data: list[tuple[Any, dict[str, Any], str]],
) -> list[LayerSaveSpec]:
    """Normalize a list of FullLayerData tuples into LayerSaveSpec objects.

    RGB layers (``rgb=True`` in attributes) are expanded into one spec per
    colour channel (R, G, B, and optionally A) so they map onto the C-first
    OME-Zarr layout without adding a spurious sample axis.
    """
    specs: list[LayerSaveSpec] = []
    for entry in layer_data:
        _, attrs, _ = entry
        if (attrs or {}).get('rgb'):
            specs.extend(_normalize_rgb_as_channel_specs(entry))
        else:
            specs.append(_normalize_single_layer(entry))
    _validate_compatible_layers(specs)
    return specs


def _normalize_rgb_as_channel_specs(
    entry: tuple[Any, dict[str, Any], str],
) -> list[LayerSaveSpec]:
    """Expand a single RGB layer (\u2026, S) into one LayerSaveSpec per colour channel.

    napari stores RGB images with the samples axis last ``(Y, X, S)`` and sets
    ``rgb=True`` in layer attributes.  OME-Zarr has no Samples axis; each
    colour channel is written as a separate C slice so the on-disk shape is
    ``(S, Y, X)``, stacked naturally by the writer.
    """
    data, attrs, _ = entry
    attrs = attrs or {}
    data_levels = _normalize_levels(data)
    arr0 = np.asarray(data_levels[0])
    n_channels = arr0.shape[-1]
    channel_labels = ('R', 'G', 'B', 'A')[:n_channels]
    spatial_ndim = arr0.ndim - 1
    axis_labels = _normalize_axis_labels(
        attrs.get('axis_labels'), spatial_ndim
    )
    scale = _normalize_scale(attrs.get('scale'), len(axis_labels))
    units = _normalize_units(attrs.get('units'), len(axis_labels))
    base_name = str(attrs.get('name') or 'Image')
    colors = {'R': 'FF0000', 'G': '00FF00', 'B': '0000FF', 'A': 'FFFFFF'}
    return [
        LayerSaveSpec(
            data_levels=tuple(np.asarray(lvl)[..., i] for lvl in data_levels),
            layer_type='image',
            name=f'{base_name}_{ch}',
            scale=scale,
            axis_labels=axis_labels,
            units=units,
            color=colors[ch],
            visible=bool(attrs.get('visible', True)),
        )
        for i, ch in enumerate(channel_labels)
    ]


def _normalize_single_layer(
    entry: tuple[Any, dict[str, Any], str],
) -> LayerSaveSpec:
    data, attributes, layer_type = entry
    attrs = attributes or {}
    normalized_type = (layer_type or 'image').lower()

    if normalized_type not in SUPPORTED_LAYER_TYPES:
        raise SaveNormalizationError(
            f'Unsupported layer type for OME-Zarr export: {normalized_type!r}'
        )

    data_levels = _normalize_levels(data)
    array = data_levels[0]
    ndim = (
        int(array.ndim)
        if hasattr(array, 'ndim')
        else int(np.asarray(array).ndim)
    )
    axis_labels = _normalize_axis_labels(attrs.get('axis_labels'), ndim)
    scale = _normalize_scale(attrs.get('scale'), len(axis_labels))
    units = _normalize_units(attrs.get('units'), len(axis_labels))

    display: dict[str, Any] = {'visible': bool(attrs.get('visible', True))}
    if normalized_type == 'image':
        display['color'] = colormap_end_color(attrs.get('colormap'))
        display['contrast_limits'] = _normalize_limits(
            attrs.get('contrast_limits')
        )
    else:
        display['label_colors'] = label_colors(attrs.get('colormap'))

    return LayerSaveSpec(
        data_levels=data_levels,
        layer_type=normalized_type,
        name=str(attrs.get('name') or normalized_type.title()),
        scale=scale,
        axis_labels=axis_labels,
        units=units,
        **display,
    )


def colormap_end_color(colormap: Any) -> str | None:
    """The colour a linear colormap ends on, as OMERO's ``'RRGGBB'``.

    OMERO stores one colour per channel, which is what a single-hue napari
    colormap is.  For a many-colour LUT (``'magma'``) the end colour is the
    closest OMERO can say.  Accepts napari's layer state (a dict with
    ``colors``), a ``Colormap``, a hex string, or a colormap name (looked up
    in napari, if it is installed).
    """
    if colormap is None:
        return None
    if isinstance(colormap, str):
        match = _HEX.match(colormap)
        if match:
            return match.group(1).upper()
        try:
            from napari.utils.colormaps import ensure_colormap
        except ImportError:
            return None
        try:
            colormap = ensure_colormap(colormap)
        except KeyError:
            return None
    colors = (
        colormap.get('colors')
        if isinstance(colormap, Mapping)
        else getattr(colormap, 'colors', None)
    )
    if colors is None:
        return None
    end = np.asarray(colors, dtype=float)[-1]
    return ''.join(f'{round(float(c) * 255):02X}' for c in end[:3])


def label_colors(colormap: Any) -> dict[int, tuple[int, int, int, int]]:
    """Label value to RGBA (0-255) for colours set per label.

    Only a direct colormap (napari's ``DirectLabelColormap``, or a dict)
    names colours per label; a cyclic one is a palette, which
    ``image-label`` cannot say, so it gives nothing.  The background and
    the default (``None``) entry are left out.
    """
    color_dict = getattr(colormap, 'color_dict', None)
    if color_dict is None and isinstance(colormap, Mapping):
        color_dict = colormap.get('color_dict', colormap)
    if not isinstance(color_dict, Mapping):
        return {}
    out: dict[int, tuple[int, int, int, int]] = {}
    for key, value in color_dict.items():
        if key is None or isinstance(key, str):
            continue
        try:
            label = int(key)
            rgba = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if label == 0 or rgba.size not in (3, 4):
            continue
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        r, g, b, a = (round(float(c) * 255) for c in rgba)
        out[label] = (r, g, b, a)
    return out


def _normalize_limits(limits: Any) -> tuple[float, float] | None:
    try:
        low, high = (float(v) for v in limits)
    except (TypeError, ValueError):
        return None
    return (low, high) if low < high else None


def _normalize_levels(data: Any) -> tuple[Any, ...]:
    """Return a tuple of resolution-level arrays from single or multiscale data."""
    if isinstance(data, Sequence) and not hasattr(data, 'dtype'):
        if not data:
            raise SaveNormalizationError(
                'Layer data sequence cannot be empty.'
            )
        if hasattr(data[0], 'shape'):
            return tuple(data)  # list of arrays → multiscale pyramid
    return (data,)


def _normalize_axis_labels(
    axis_labels: Sequence[str] | None,
    ndim: int,
) -> tuple[str, ...]:
    if ndim > len(_DEFAULT_AXES):
        raise SaveNormalizationError(
            f'OME-Zarr layers have at most {len(_DEFAULT_AXES)} axes '
            f'({"".join(_DEFAULT_AXES)}); this layer has {ndim}.'
        )
    if axis_labels is None or all(
        _PLACEHOLDER_AXIS.match(str(label)) for label in axis_labels
    ):
        return _DEFAULT_AXES[-ndim:]
    if len(axis_labels) != ndim:
        raise SaveNormalizationError(
            'axis_labels length must match the saved data ndim.'
        )
    return tuple(str(label).upper() for label in axis_labels)


def _normalize_scale(
    scale: Sequence[float] | None,
    ndim: int,
) -> tuple[float, ...]:
    if scale is None:
        return (1.0,) * ndim
    if len(scale) != ndim:
        raise SaveNormalizationError('scale length must match axis_labels.')
    return tuple(float(v) for v in scale)


def _normalize_units(
    units: Sequence[str | None] | None,
    ndim: int,
) -> tuple[str | None, ...]:
    if units is None:
        return (None,) * ndim
    if len(units) != ndim:
        raise SaveNormalizationError('units length must match axis_labels.')
    return tuple(
        None if u is None or str(u) in _NO_UNIT else str(u) for u in units
    )


def _validate_compatible_layers(specs: list[LayerSaveSpec]) -> None:
    """Validate specs for writing to a single OME-Zarr store.

    Image specs are stacked along the C axis at the store root; they must all
    share axis_labels, scale, units, and resolution level count.

    Labels specs each go to a ``labels/<name>/`` NGFF subgroup when images are
    also present.  In a pure-labels export they are stacked to the root and are
    subject to the same C-axis compatibility rules.
    """
    if not specs:
        raise SaveNormalizationError('At least one layer is required.')

    image_specs = [s for s in specs if s.layer_type == 'image']
    labels_specs = [s for s in specs if s.layer_type == 'labels']

    # Image specs stacked to root must be mutually compatible.
    if len(image_specs) > 1:
        _check_group_compatible(image_specs)
    # Pure-labels: stacked to root, must also be mutually compatible.
    if not image_specs and len(labels_specs) > 1:
        _check_group_compatible(labels_specs)
    # Mixed: each label image lives under the image and shares its grid.
    # Its pyramid is rebuilt to the image's levels, so only level 0 must
    # match.
    if image_specs and labels_specs:
        first = image_specs[0]
        shape = tuple(np.shape(first.data_levels[0]))
        for spec in labels_specs:
            if spec.axis_labels != first.axis_labels or (
                tuple(np.shape(spec.data_levels[0])) != shape
            ):
                raise SaveNormalizationError(
                    f'Labels {spec.name!r} must have the same axes and shape '
                    f'as the image to be saved under it: '
                    f'{spec.axis_labels} {np.shape(spec.data_levels[0])} vs '
                    f'{first.axis_labels} {shape}.'
                )
    names = [s.name for s in labels_specs]
    if image_specs and len(set(names)) != len(names):
        raise SaveNormalizationError(f'Labels names must be unique: {names}')


def _check_group_compatible(specs: list[LayerSaveSpec]) -> None:
    """Raise SaveNormalizationError if specs cannot be stacked along the C axis."""
    first = specs[0]
    for spec in specs[1:]:
        if spec.axis_labels != first.axis_labels:
            raise SaveNormalizationError(
                'All layers must share axis_labels to be exported together.'
            )
        if spec.scale != first.scale:
            raise SaveNormalizationError(
                'All layers must share scale to be exported together.'
            )
        if spec.units != first.units:
            raise SaveNormalizationError(
                'All layers must share units to be exported together.'
            )
        if len(spec.data_levels) != len(first.data_levels):
            raise SaveNormalizationError(
                'All layers must have the same number of resolution levels.'
            )
