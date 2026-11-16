"""Write napari layers to OME-Zarr (NGFF 0.5, Zarr v3).

Public API
----------
write_layers_to_ome_zarr  -- napari writer; accepts FullLayerData tuples.
write_file_to_ome_zarr    -- convert a single source file to OME-Zarr.
write_files_to_ome_zarr   -- batch-convert multiple source files to OME-Zarr.

What is written, so that :class:`ndevio.nImage` reads back the same layers:

=====================  ==================================================
napari layer           OME-Zarr
=====================  ==================================================
image layers           channels of the root image, stacked on a ``c`` axis
name                   ``omero.channels[].label``
colormap               ``omero.channels[].color`` (the colormap's end colour)
contrast_limits        ``omero.channels[].window.start`` / ``.end``
visible                ``omero.channels[].active``
scale, units, axes     ``multiscales[].axes`` and ``coordinateTransformations``
multiscale data        one dataset per level
labels layers          ``labels/<name>``, an image-label with the image's
                       levels; a direct colormap becomes ``image-label.colors``
=====================  ==================================================

When only labels layers are saved there is no image to put them under, so
they are stacked as the root image and an ``ndevio`` attribute records
that they are labels.

Large files are written chunk by chunk through dask; for converting very
large files without napari layers in between, bioio-conversion
(https://github.com/bioio-devs/bioio-conversion) writes the same layout.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._save_normalize import normalize_layer_data

if TYPE_CHECKING:
    from ._save_model import LayerSaveSpec

NGFF_VERSION = '0.5'
LABELS_GROUP = 'labels'

# The root attribute that records layer types NGFF has no word for.
NDEVIO_ATTR = 'ndevio'
NDEVIO_ATTR_VERSION = 1

# Used for an image layer that says nothing about its colour.
_FALLBACK_COLORS = ('FF00FF', '00FF00', '00FFFF', 'FFFF00', 'FF0000', '0000FF')


def write_layers_to_ome_zarr(
    path: str,
    layer_data: list[tuple[Any, dict[str, Any], str]],
) -> list[str]:
    """Write napari image and labels layers to one OME-Zarr store.

    Parameters
    ----------
    path : str
        Where to write the store (``.ome.zarr``).  Replaced if it exists.
    layer_data : list of (data, attributes, layer_type)
        napari's FullLayerData tuples, as napari hands them to a writer or
        as ``Layer.as_layer_data_tuple()`` returns them.  Image layers must
        share axes, scale, units and number of levels; labels layers must
        have the image's axes and level-0 shape.

    Returns
    -------
    list of str
        The path written, as napari's writer protocol expects.

    """
    specs = normalize_layer_data(layer_data)
    _write_specs(str(path), specs)
    return [str(path)]


def write_file_to_ome_zarr(
    source: str | Path,
    destination: str | Path,
    *,
    scene: int | str = 0,
    layer_type: str | None = None,
) -> list[str]:
    """Convert one scene of a file that ndevio can read to OME-Zarr.

    The file is read as napari layers would be (channel names, colours,
    the labels of an OME-Zarr source) and written with
    :func:`write_layers_to_ome_zarr`, so the result opens as the same
    layers.  Multiscale sources keep their number of levels.
    """
    from .nimage import nImage

    image = nImage(source)
    image.set_scene(scene)
    layer_data = [
        (data, {**attrs, 'name': _short_name(attrs['name'])}, kind)
        for data, attrs, kind in image.get_layer_data_tuples(
            layer_type=layer_type
        )
    ]
    return write_layers_to_ome_zarr(str(destination), layer_data)


def write_files_to_ome_zarr(
    sources: Iterable[str | Path],
    destination_directory: str | Path,
    *,
    scene: int | str = 0,
    layer_type: str | None = None,
) -> list[str]:
    """Convert several files to ``<stem>.ome.zarr`` in one folder."""
    destination_root = Path(destination_directory)
    destination_root.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for source in sources:
        source_path = Path(source)
        destination = (
            destination_root / f'{_output_stem(source_path)}.ome.zarr'
        )
        written.extend(
            write_file_to_ome_zarr(
                source_path, destination, scene=scene, layer_type=layer_type
            )
        )
    return written


# --- the store ------------------------------------------------------------


def _write_specs(path: str, specs: list[LayerSaveSpec]) -> None:
    import zarr

    images = [s for s in specs if s.layer_type == 'image']
    labels = [s for s in specs if s.layer_type == 'labels']
    root_specs, label_specs = (images, labels) if images else (labels, [])

    level_shapes = _write_root_image(path, root_specs, Path(path).stem)

    root = zarr.open_group(path, mode='r+')
    if not images:
        root.attrs[NDEVIO_ATTR] = {
            'version': NDEVIO_ATTR_VERSION,
            'layer_types': [s.layer_type for s in root_specs],
        }
    if label_specs:
        group = root.require_group(LABELS_GROUP)
        group.attrs['ome'] = {
            'version': NGFF_VERSION,
            'labels': [s.name for s in label_specs],
        }
        spatial_shapes = [shape[1:] for shape in level_shapes]
        for spec in label_specs:
            _write_label(
                f'{path}/{LABELS_GROUP}/{spec.name}', spec, spatial_shapes
            )


def _write_root_image(
    path: str, specs: list[LayerSaveSpec], image_name: str
) -> list[tuple[int, ...]]:
    """Stack *specs* on a ``c`` axis and write them; return the level shapes."""
    import dask.array as da
    from bioio_ome_zarr.writers import Channel, OMEZarrWriter

    first = specs[0]
    stacked = da.stack([_lazy(s.data_levels[0]) for s in specs], axis=0)
    level_shapes = [
        (len(specs), *(int(d) for d in np.shape(level)))
        for level in first.data_levels
    ]
    channels = [
        Channel(
            label=spec.name,
            color=spec.color or _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)],
            active=spec.visible,
            window=_window(spec, stacked.dtype),
        )
        for i, spec in enumerate(specs)
    ]
    writer = OMEZarrWriter(
        store=path,
        level_shapes=level_shapes,
        dtype=stacked.dtype,
        zarr_format=3,
        image_name=image_name,
        channels=channels,
        axes_names=['c', *(a.lower() for a in first.axis_labels)],
        axes_types=['channel', *(_axis_type(a) for a in first.axis_labels)],
        axes_units=[None, *first.units],
        physical_pixel_size=[1.0, *first.scale],
    )
    writer.write_full_volume(stacked)
    return level_shapes


def _write_label(
    path: str,
    spec: LayerSaveSpec,
    level_shapes: list[tuple[int, ...]],
) -> None:
    """Write one image-label with the image's levels."""
    import zarr
    from bioio_ome_zarr.writers import OMEZarrWriter

    data = _lazy(spec.data_levels[0])
    writer = OMEZarrWriter(
        store=path,
        level_shapes=level_shapes,
        dtype=data.dtype,
        zarr_format=3,
        image_name=spec.name,
        axes_names=[a.lower() for a in spec.axis_labels],
        axes_types=[_axis_type(a) for a in spec.axis_labels],
        axes_units=list(spec.units),
        physical_pixel_size=list(spec.scale),
    )
    writer.write_full_volume(data)

    image_label: dict[str, Any] = {
        'version': NGFF_VERSION,
        'source': {'image': '../../'},
    }
    if spec.label_colors:
        image_label['colors'] = [
            {'label-value': value, 'rgba': list(rgba)}
            for value, rgba in sorted(spec.label_colors.items())
        ]
    group = zarr.open_group(path, mode='r+')
    ome = dict(group.attrs.get('ome', {}))
    ome.pop('omero', None)  # a label image has no rendering channels
    ome['image-label'] = image_label
    group.attrs['ome'] = ome


def _window(spec: LayerSaveSpec, dtype: np.dtype) -> dict[str, float]:
    """OMERO window: the dtype's range, and the layer's contrast limits."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low, high = float(info.min), float(info.max)
    else:
        low, high = spec.contrast_limits or (0.0, 1.0)
    start, end = spec.contrast_limits or (low, high)
    return {'min': low, 'max': high, 'start': start, 'end': end}


def _lazy(level: Any) -> Any:
    """A dask view of one level, so it is written chunk by chunk."""
    import dask.array as da

    if isinstance(level, da.Array):
        return level
    if not hasattr(level, 'shape'):
        level = np.asarray(level)
    return da.from_array(level, chunks='auto')


def _short_name(layer_name: str) -> str:
    """``'DAPI :: cells.ome'`` -> ``'DAPI'``: the channel, not the file."""
    return layer_name.split(' :: ')[0]


def _output_stem(source: Path) -> str:
    """Return a clean output filename stem, stripping known suffixes."""
    name = source.name
    for suffix in ('.ome.zarr', '.zarr', '.ome.tiff', '.ome.tif'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return source.stem


def _axis_type(axis_name: str) -> str:
    return 'time' if axis_name.upper() == 'T' else 'space'
