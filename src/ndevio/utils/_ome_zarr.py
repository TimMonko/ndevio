"""Read napari display metadata and labels from OME-Zarr (NGFF) stores.

bioio's OME-Zarr reader returns the image, its scale, units and channel
names.  Two things it leaves to the caller are read here:

- the OMERO rendering hints (``omero.channels[]``: colour, window,
  active), which napari calls ``colormap``, ``contrast_limits`` and
  ``visible``;
- the ``labels/`` group of the NGFF image-label spec, which holds label
  images that belong to the image.

Nothing here imports napari.  Both NGFF 0.5 (metadata under an ``ome`` key,
Zarr v3) and 0.4 (metadata at the top level, Zarr v2) are read.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

LABELS_GROUP = 'labels'

_HEX_COLOUR = re.compile(r'^#?([0-9a-fA-F]{6})$')


def ngff_attrs(attrs: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the NGFF metadata of a group's attributes.

    NGFF 0.5 nests everything under ``ome``; 0.4 and earlier put it at the
    top level.
    """
    if not isinstance(attrs, Mapping):
        return {}
    ome = attrs.get('ome')
    return ome if isinstance(ome, Mapping) else attrs


def channel_display(
    attrs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    """Napari layer kwargs for each channel, from ``omero.channels``.

    Parameters
    ----------
    attrs : Mapping or None
        The image group's attributes (``zarr.Group.attrs``, or
        ``GroupMetadata.attributes``).

    Returns
    -------
    tuple of dict
        One dict per listed channel, holding only what the store states:
        ``colormap`` (``'#rrggbb'``), ``contrast_limits`` (``(start, end)``)
        and ``visible``.  Empty when there is no ``omero`` block.  When
        every channel has the same colour (writers fill in white when they
        are told nothing), colours are left out, so the reader's default
        colour cycle applies.

    """
    omero = ngff_attrs(attrs).get('omero')
    channels = omero.get('channels') if isinstance(omero, Mapping) else None
    if not isinstance(channels, list):
        return ()

    display = [_one_channel(ch) for ch in channels]
    colours = {d.get('colormap') for d in display}
    if len(display) > 1 and len(colours) == 1:
        for d in display:
            d.pop('colormap', None)
    return tuple(display)


def _one_channel(channel: Any) -> dict[str, Any]:
    if not isinstance(channel, Mapping):
        return {}
    out: dict[str, Any] = {}

    colour = channel.get('color')
    match = _HEX_COLOUR.match(colour) if isinstance(colour, str) else None
    if match:
        out['colormap'] = '#' + match.group(1).lower()

    limits = _window_limits(channel.get('window'))
    if limits is not None:
        out['contrast_limits'] = limits

    active = channel.get('active')
    if isinstance(active, bool):
        out['visible'] = active
    return out


def _window_limits(window: Any) -> tuple[float, float] | None:
    """``(start, end)`` of an OMERO window, if it narrows the range.

    A window whose start and end span its whole ``min``-``max`` chose
    nothing (writers fill it in with the dtype range, or bioio-ome-zarr's
    0-255 default), so it gives None and napari picks contrast limits from
    the data.
    """
    if not isinstance(window, Mapping):
        return None
    try:
        start, end = float(window['start']), float(window['end'])
    except (KeyError, TypeError, ValueError):
        return None
    if not start < end:
        return None
    try:
        low, high = float(window['min']), float(window['max'])
    except (KeyError, TypeError, ValueError):
        return start, end
    if start <= low and end >= high:
        return None
    return start, end


def channel_layer_types(attrs: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Layer types ndevio recorded for the root image's channels.

    NGFF has no way to say "these channels are labels" when there is no
    image to put a ``labels/`` group under, so ndevio's writer records it
    in an ``ndevio`` attribute.  Anything else gives an empty tuple.
    """
    if not isinstance(attrs, Mapping):
        return ()
    ours = attrs.get('ndevio')
    types = ours.get('layer_types') if isinstance(ours, Mapping) else None
    if not isinstance(types, list) or not all(
        t in ('image', 'labels') for t in types
    ):
        return ()
    return tuple(types)


@dataclass
class LabelImage:
    """One image-label from a store's ``labels/`` group.

    ``levels`` are the arrays as the store holds them (``zarr.Array``),
    level 0 first, so nothing is read until napari asks for a slice.
    """

    name: str
    levels: list[Any]
    axis_labels: tuple[str, ...]
    scale: tuple[float, ...]
    units: tuple[str | None, ...]
    colors: dict[int | None, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


def read_labels(store: Any) -> list[LabelImage]:
    """Open every image-label listed in the store's ``labels/`` group.

    Parameters
    ----------
    store : str, Path or zarr store
        The image's root: a local path, an fsspec URL, or a store.

    Returns
    -------
    list of LabelImage
        In the order the group lists them.  A store with no ``labels/``
        group gives an empty list.  A listed label that cannot be opened is
        logged and skipped.

    """
    import zarr

    try:
        group = zarr.open_group(store, path=LABELS_GROUP, mode='r')
    except (FileNotFoundError, ValueError) as e:
        # zarr's GroupNotFoundError and NodeTypeValidationError are both.
        logger.debug('No labels group in %s: %s', store, e)
        return []

    names = ngff_attrs(group.attrs.asdict()).get('labels')
    if not isinstance(names, list):
        return []

    found: list[LabelImage] = []
    for name in names:
        try:
            found.append(_read_label(group, str(name)))
        except Exception as e:  # noqa: BLE001 - skip one bad label, keep others
            logger.warning('Skipping label %r: %s', name, e)
    return found


def _read_label(labels: Any, name: str) -> LabelImage:
    node = labels[name]
    meta = ngff_attrs(node.attrs.asdict())
    multiscale = meta['multiscales'][0]
    axes = multiscale['axes']
    datasets = multiscale['datasets']

    levels = [node[d['path']] for d in datasets]
    scale = _level_scale(datasets[0], len(axes))
    axis_labels = tuple(_axis_name(ax).upper() for ax in axes)
    units = tuple(
        ax.get('unit') if isinstance(ax, Mapping) else None for ax in axes
    )

    # napari draws Labels without a channel axis; drop a size-1 c axis.
    keep = [
        i
        for i, ax in enumerate(axes)
        if not (_axis_name(ax).lower() == 'c' and levels[0].shape[i] == 1)
    ]
    if len(keep) != len(axes):
        index = tuple(
            slice(None) if i in keep else 0 for i in range(len(axes))
        )
        levels = [LazyView(level, index) for level in levels]
        axis_labels = tuple(axis_labels[i] for i in keep)
        scale = tuple(scale[i] for i in keep)
        units = tuple(units[i] for i in keep)

    return LabelImage(
        name=name,
        levels=levels,
        axis_labels=axis_labels,
        scale=scale,
        units=units,
        colors=_label_colors(meta.get('image-label')),
    )


def _axis_name(axis: Any) -> str:
    return str(axis['name'] if isinstance(axis, Mapping) else axis)


def _level_scale(dataset: Mapping[str, Any], ndim: int) -> tuple[float, ...]:
    for transform in dataset.get('coordinateTransformations', []):
        if transform.get('type') == 'scale':
            return tuple(float(s) for s in transform['scale'])
    return (1.0,) * ndim


def _label_colors(
    image_label: Any,
) -> dict[int | None, tuple[float, float, float, float]]:
    """Label value to RGBA (0-1) from ``image-label.colors``."""
    if not isinstance(image_label, Mapping):
        return {}
    colors: dict[int | None, tuple[float, float, float, float]] = {}
    for entry in image_label.get('colors', []) or []:
        try:
            value = int(entry['label-value'])
            r, g, b, a = (float(c) / 255 for c in entry['rgba'])
        except (KeyError, TypeError, ValueError):
            continue
        colors[value] = (r, g, b, a)
    return colors


class LazyView:
    """A view of an array with some axes fixed at an index, read on access.

    Holds "channel 2 of this ``zarr.Array``, without its size-1 ``t``
    axis" without reading anything, which is what napari needs from layer
    data: ``shape``, ``dtype``, ``ndim`` and ``__getitem__``.  Indexing
    returns what the underlying array returns (numpy, for zarr).

    Parameters
    ----------
    array : array-like
        Anything with ``shape``, ``dtype`` and numpy-style ``__getitem__``.
    index : tuple of int or slice(None), optional
        One entry per axis of *array*: an int fixes that axis (it is not
        part of the view), ``slice(None)`` keeps it.  Default: keep all.

    """

    def __init__(
        self, array: Any, index: tuple[int | slice, ...] | None = None
    ) -> None:
        if index is None:
            index = (slice(None),) * len(array.shape)
        if len(index) != len(array.shape):
            raise ValueError(
                f'index has {len(index)} entries for {len(array.shape)} axes'
            )
        self._array = array
        self._index = tuple(index)
        self._kept = [i for i, ix in enumerate(index) if isinstance(ix, slice)]
        self.dtype = array.dtype
        self.shape = tuple(int(array.shape[i]) for i in self._kept)
        self.ndim = len(self.shape)

    @property
    def size(self) -> int:
        """Number of elements, without reading them."""
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def nbytes(self) -> int:
        """Bytes the view would take in memory, without reading it."""
        return self.size * self.dtype.itemsize

    def take(self, axis: int, i: int) -> LazyView:
        """The view with one more axis fixed at *i*, still unread."""
        if not -self.ndim <= axis < self.ndim:
            raise IndexError(f'axis {axis} out of range for {self.ndim} dims')
        if not -self.shape[axis] <= i < self.shape[axis]:
            raise IndexError(f'index {i} out of range for axis {axis}')
        index = list(self._index)
        index[self._kept[axis]] = i % self.shape[axis]
        return LazyView(self._array, tuple(index))

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, tuple):
            key = (key,)
        at = next((n for n, k in enumerate(key) if k is Ellipsis), None)
        if at is not None:
            fill = (slice(None),) * (self.ndim - len(key) + 1)
            key = key[:at] + fill + key[at + 1 :]
        if len(key) > self.ndim:
            raise IndexError(
                f'too many indices: {len(key)} for {self.ndim} dims'
            )
        key = key + (slice(None),) * (self.ndim - len(key))
        full = list(self._index)
        for axis, k in zip(self._kept, key, strict=True):
            full[axis] = k
        return self._array[tuple(full)]

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        import numpy as np

        return np.asarray(self[...], dtype=dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f'LazyView(shape={self.shape}, dtype={self.dtype}, '
            f'of {type(self._array).__name__})'
        )


def take_channel(array: Any, axis: int, i: int) -> Any:
    """Channel *i* along *axis*, without reading it if *array* is lazy.

    A :class:`LazyView` stays a view.  Anything else (numpy, dask) is
    indexed as before: numpy reads it, dask stays lazy by itself.
    """
    if isinstance(array, LazyView):
        return array.take(axis, i)
    index: list[int | slice] = [slice(None)] * array.ndim
    index[axis] = i
    return array[tuple(index)]


def level_views(store: Any, dims: Sequence[str]) -> list[LazyView] | None:
    """The store's resolution levels as views with exactly *dims*.

    Parameters
    ----------
    store : str, Path or zarr store
        The image's root.
    dims : sequence of str
        The dims the layer data must have, in order (``nImage``'s
        squeezed reference dims, e.g. ``('C', 'Z', 'Y', 'X')``).

    Returns
    -------
    list of LazyView or None
        One view per dataset, level 0 first.  None when the store cannot
        give *dims* by dropping size-1 axes alone (a transposed store, an
        axis *dims* names that the store lacks, more than one multiscale):
        the caller then keeps its own (dask) path.

    """
    import zarr

    try:
        root = zarr.open_group(store, mode='r')
    except (FileNotFoundError, ValueError):
        return None
    multiscales = ngff_attrs(root.attrs.asdict()).get('multiscales')
    if not isinstance(multiscales, list) or len(multiscales) != 1:
        return None
    try:
        names = [_axis_name(ax).upper() for ax in multiscales[0]['axes']]
        paths = [d['path'] for d in multiscales[0]['datasets']]
    except (KeyError, TypeError):
        return None
    if [n for n in names if n in dims] != list(dims):
        return None

    views: list[LazyView] = []
    for path in paths:
        array = root[path]
        if len(array.shape) != len(names):
            return None
        index: list[int | slice] = []
        for name, size in zip(names, array.shape, strict=True):
            if name in dims:
                index.append(slice(None))
            elif size == 1:
                index.append(0)
            else:
                return None
        views.append(LazyView(array, tuple(index)))
    return views


__all__ = [
    'LABELS_GROUP',
    'LabelImage',
    'LazyView',
    'channel_display',
    'channel_layer_types',
    'level_views',
    'ngff_attrs',
    'read_labels',
    'take_channel',
]
