"""Reading napari display metadata and labels from OME-Zarr stores.

The stores are written here with bioio-ome-zarr's own writer plus plain
zarr for the ``labels/`` group, so the tests describe what a store written
by any NGFF-0.5 tool should open as.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip('zarr')
writers = pytest.importorskip('bioio_ome_zarr.writers')

UNIT = 'micrometer'


def _write_image(path: Path, *, channels: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, (2, 4, 32, 32), dtype=np.uint16)
    kwargs = {}
    if channels:
        kwargs['channels'] = [
            writers.Channel(
                label='DAPI',
                color='0000FF',
                window={'min': 0, 'max': 4000, 'start': 100, 'end': 3000},
            ),
            writers.Channel(
                label='GFP',
                color='00FF00',
                window={'min': 0, 'max': 4000, 'start': 200, 'end': 2500},
                active=False,
            ),
        ]
    writer = writers.OMEZarrWriter(
        str(path),
        level_shapes=[(2, 4, 32, 32), (2, 4, 16, 16)],
        dtype=data.dtype,
        axes_names=['c', 'z', 'y', 'x'],
        axes_types=['channel', 'space', 'space', 'space'],
        axes_units=[None, UNIT, UNIT, UNIT],
        physical_pixel_size=[1.0, 2.0, 0.5, 0.5],
        **kwargs,
    )
    writer.write_full_volume(data)
    return data


def _write_label(path: Path, name: str) -> np.ndarray:
    """Add an NGFF 0.5 image-label to the store at *path*."""
    root = zarr.open_group(str(path), mode='r+')
    labels = root.require_group('labels')
    names = list(labels.attrs.get('ome', {}).get('labels', []))
    labels.attrs['ome'] = {'version': '0.5', 'labels': [*names, name]}
    lab = np.zeros((4, 32, 32), dtype=np.uint32)
    lab[:, 4:10, 4:10] = 1
    lab[:, 20:30, 20:30] = 2
    group = labels.require_group(name)
    group.create_array('0', data=lab, chunks=(1, 32, 32))
    group.create_array('1', data=lab[:, ::2, ::2], chunks=(1, 16, 16))
    axes = [{'name': ax, 'type': 'space', 'unit': UNIT} for ax in 'zyx']
    group.attrs['ome'] = {
        'version': '0.5',
        'multiscales': [
            {
                'name': name,
                'axes': axes,
                'datasets': [
                    {
                        'path': '0',
                        'coordinateTransformations': [
                            {'type': 'scale', 'scale': [2.0, 0.5, 0.5]}
                        ],
                    },
                    {
                        'path': '1',
                        'coordinateTransformations': [
                            {'type': 'scale', 'scale': [2.0, 1.0, 1.0]}
                        ],
                    },
                ],
            }
        ],
        'image-label': {
            'version': '0.5',
            'colors': [
                {'label-value': 1, 'rgba': [255, 0, 0, 255]},
                {'label-value': 2, 'rgba': [0, 255, 0, 255]},
            ],
            'source': {'image': '../../'},
        },
    }
    return lab


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / 'cells.ome.zarr'
    _write_image(path)
    return path


# --- the pure helpers ------------------------------------------------------


class TestChannelDisplay:
    def test_reads_colour_window_and_visibility_from_omero(self):
        from ndevio.utils._ome_zarr import channel_display

        attrs = {
            'ome': {
                'version': '0.5',
                'omero': {
                    'channels': [
                        {
                            'color': '0000FF',
                            'window': {'start': 1, 'end': 9},
                            'active': True,
                        },
                        {'color': 'ff8000', 'active': False},
                    ]
                },
            }
        }
        assert channel_display(attrs) == (
            {
                'colormap': '#0000ff',
                'contrast_limits': (1.0, 9.0),
                'visible': True,
            },
            {'colormap': '#ff8000', 'visible': False},
        )

    def test_ngff_04_keeps_omero_at_the_root(self):
        from ndevio.utils._ome_zarr import channel_display

        attrs = {'omero': {'channels': [{'color': 'FF0000'}]}}
        assert channel_display(attrs) == ({'colormap': '#ff0000'},)

    @pytest.mark.parametrize(
        'attrs',
        [
            None,
            {},
            {'ome': {}},
            {'omero': {'channels': 'nonsense'}},
            {'omero': {'channels': [{'color': 'not-a-colour'}]}},
            {'omero': {'channels': [{'window': {'start': 5, 'end': 5}}]}},
            # A window over its whole range chose nothing.
            {
                'omero': {
                    'channels': [
                        {
                            'window': {
                                'min': 0,
                                'max': 65535,
                                'start': 0,
                                'end': 65535,
                            }
                        }
                    ]
                }
            },
        ],
    )
    def test_missing_or_malformed_gives_nothing_to_override(self, attrs):
        from ndevio.utils._ome_zarr import channel_display

        assert all(d == {} for d in channel_display(attrs))


class TestLazyView:
    def test_drops_fixed_axes_and_reads_on_index(self):
        from ndevio.utils._ome_zarr import LazyView

        base = np.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
        view = LazyView(base, (0, slice(None), slice(None), slice(None)))
        assert view.shape == (2, 3, 4)
        assert view.ndim == 3
        assert view.nbytes == base.nbytes
        np.testing.assert_array_equal(view[1], base[0, 1])
        np.testing.assert_array_equal(view[..., 2], base[0, :, :, 2])
        np.testing.assert_array_equal(np.asarray(view), base[0])

    def test_take_composes_without_reading(self):
        from ndevio.utils._ome_zarr import LazyView

        class Counting:
            def __init__(self, a):
                self.a, self.reads = a, 0
                self.shape, self.dtype = a.shape, a.dtype

            def __getitem__(self, key):
                self.reads += 1
                return self.a[key]

        base = Counting(np.arange(24).reshape(2, 3, 4))
        channel = LazyView(base).take(0, 1)
        assert base.reads == 0
        assert channel.shape == (3, 4)
        np.testing.assert_array_equal(channel[2], base.a[1, 2])
        assert base.reads == 1

    def test_take_channel_leaves_numpy_and_dask_alone(self):
        da = pytest.importorskip('dask.array')
        from ndevio.utils._ome_zarr import take_channel

        a = np.arange(24).reshape(2, 3, 4)
        np.testing.assert_array_equal(take_channel(a, 0, 1), a[1])
        lazy = take_channel(da.from_array(a), 1, 2)
        assert isinstance(lazy, da.Array)
        np.testing.assert_array_equal(lazy.compute(), a[:, 2])


class TestLevelViews:
    def test_store_levels_with_singletons_dropped(self, tmp_path):
        from ndevio.utils._ome_zarr import level_views

        path = tmp_path / 'a.ome.zarr'
        data = _write_image(path)
        views = level_views(str(path), ('C', 'Z', 'Y', 'X'))
        assert [v.shape for v in views] == [(2, 4, 32, 32), (2, 4, 16, 16)]
        np.testing.assert_array_equal(np.asarray(views[0]), data)

    def test_a_transposed_store_gives_none(self, tmp_path):
        from ndevio.utils._ome_zarr import level_views

        path = tmp_path / 't.zarr'
        root = zarr.open_group(str(path), mode='w')
        root.create_array('0', data=np.zeros((8, 4, 2), dtype=np.uint8))
        root.attrs['ome'] = {
            'version': '0.5',
            'multiscales': [
                {
                    'axes': [{'name': n} for n in ('x', 'y', 'c')],
                    'datasets': [{'path': '0'}],
                }
            ],
        }
        assert level_views(str(path), ('C', 'Y', 'X')) is None

    def test_not_a_store_gives_none(self, tmp_path):
        from ndevio.utils._ome_zarr import level_views

        assert level_views(str(tmp_path / 'missing.zarr'), ('Y', 'X')) is None


# --- through nImage ---------------------------------------------------------


class TestDaskFree:
    """A multiscale OME-Zarr store gives napari zarr arrays, not dask."""

    def test_channel_layers_are_unread_views(self, store):
        from ndevio import nImage
        from ndevio.utils._ome_zarr import LazyView

        layers = nImage(store).get_layer_data_tuples()
        for data, _, _ in layers:
            assert all(isinstance(level, LazyView) for level in data)
        assert [lvl.shape for lvl in layers[0][0]] == [
            (4, 32, 32),
            (4, 16, 16),
        ]

    def test_values_match_the_dask_path(self, store):
        from ndevio import nImage

        img = nImage(store)
        views = img.get_layer_data_tuples()[1][0]
        dask_level0 = img.xarray_dask_data.isel(T=0, C=1).data
        np.testing.assert_array_equal(views[0][2], dask_level0[2].compute())

    def test_napari_slices_the_views(self, store):
        layers = pytest.importorskip('napari.layers')
        from ndevio import nImage

        data, kwargs, kind = nImage(store).get_layer_data_tuples()[0]
        layer = layers.Layer.create(data, kwargs, kind)
        assert layer.multiscale
        plane = np.asarray(layer.data[0][1])
        assert plane.shape == (32, 32)

    def test_single_level_small_store_stays_in_memory(self, tmp_path):
        from ndevio import nImage
        from ndevio.utils._ome_zarr import LazyView

        path = tmp_path / 'one.ome.zarr'
        writers.OMEZarrWriter(
            str(path),
            level_shapes=[(2, 16, 16)],
            dtype=np.uint8,
            axes_names=['c', 'y', 'x'],
            axes_types=['channel', 'space', 'space'],
        ).write_full_volume(np.ones((2, 16, 16), dtype=np.uint8))
        data = nImage(path).get_layer_data_tuples()[0][0]
        level = data[0] if isinstance(data, list) else data
        assert isinstance(level, np.ndarray)
        assert not isinstance(level, LazyView)


# --- through nImage ---------------------------------------------------------


class TestReadDisplay:
    def test_channel_names_units_and_scale(self, store):
        from ndevio import nImage

        img = nImage(store)
        assert list(img.channel_names) == ['DAPI', 'GFP']
        assert img.layer_scale == (2.0, 0.5, 0.5)
        assert [str(u) for u in img.layer_units] == [UNIT] * 3

    def test_layers_take_colour_window_and_visibility(self, store):
        from ndevio import nImage

        layers = nImage(store).get_layer_data_tuples()
        image_layers = [kw for _, kw, kind in layers if kind == 'image']
        dapi, gfp = image_layers
        assert dapi['colormap'] == '#0000ff'
        assert dapi['contrast_limits'] == (100.0, 3000.0)
        assert 'visible' not in dapi or dapi['visible']
        assert gfp['colormap'] == '#00ff00'
        assert gfp['contrast_limits'] == (200.0, 2500.0)
        assert gfp['visible'] is False

    def test_channel_kwargs_still_win(self, store):
        from ndevio import nImage

        layers = nImage(store).get_layer_data_tuples(
            channel_kwargs={'DAPI': {'colormap': 'gray'}}
        )
        assert layers[0][1]['colormap'] == 'gray'

    def test_store_without_omero_keeps_default_cycle(self, tmp_path):
        from ndevio import nImage

        path = tmp_path / 'plain.ome.zarr'
        _write_image(path, channels=False)
        img = nImage(path)
        layers = img.get_layer_data_tuples()
        assert [kw['colormap'] for _, kw, _ in layers] == ['magenta', 'green']
        # The reader gives one channel name for two channels here; every
        # layer still gets a name of its own.
        assert len(img.layer_names) == 2
        assert len({kw['name'] for _, kw, _ in layers}) == 2


class TestReadLabels:
    def test_labels_group_opens_as_labels_layers(self, store):
        from ndevio import nImage

        expected = _write_label(store, 'nuclei')
        layers = nImage(store).get_layer_data_tuples()
        kinds = [kind for _, _, kind in layers]
        assert kinds == ['image', 'image', 'labels']

        data, kwargs, _ = layers[-1]
        assert kwargs['name'].startswith('nuclei')
        assert kwargs['scale'] == (2.0, 0.5, 0.5)
        assert kwargs['axis_labels'] == ('Z', 'Y', 'X')
        assert [str(u) for u in kwargs['units']] == [UNIT] * 3
        # Multiscale, level 0 first, and read straight from zarr.
        assert isinstance(data, list)
        assert len(data) == 2
        np.testing.assert_array_equal(np.asarray(data[0]), expected)

    def test_labels_colours_from_image_label(self, store):
        from ndevio import nImage

        _write_label(store, 'nuclei')
        kwargs = nImage(store).get_layer_data_tuples()[-1][1]
        colors = kwargs['colormap']
        assert colors[1] == pytest.approx((1.0, 0.0, 0.0, 1.0))
        assert colors[2] == pytest.approx((0.0, 1.0, 0.0, 1.0))
        assert None in colors

    def test_several_labels_in_listed_order(self, store):
        from ndevio import nImage

        _write_label(store, 'nuclei')
        _write_label(store, 'cells')
        names = [
            kw['name']
            for _, kw, kind in nImage(store).get_layer_data_tuples()
            if kind == 'labels'
        ]
        assert [n.split(' :: ')[0] for n in names] == ['nuclei', 'cells']

    def test_include_labels_false_skips_them(self, store):
        from ndevio import nImage

        _write_label(store, 'nuclei')
        layers = nImage(store).get_layer_data_tuples(include_labels=False)
        assert [kind for _, _, kind in layers] == ['image', 'image']

    def test_listed_label_that_is_missing_is_skipped(self, store, caplog):
        from ndevio import nImage

        _write_label(store, 'nuclei')
        group = zarr.open_group(str(store / 'labels'), mode='r+')
        group.attrs['ome'] = {'version': '0.5', 'labels': ['nuclei', 'gone']}
        layers = nImage(store).get_layer_data_tuples()
        assert [kind for _, _, kind in layers] == ['image', 'image', 'labels']
        assert 'gone' in caplog.text

    def test_non_zarr_image_has_no_labels(self, resources_dir):
        from ndevio import nImage

        path = resources_dir / 'cells3d2ch_legacy.tiff'
        if not path.exists():
            pytest.skip('resource missing')
        layers = nImage(path).get_layer_data_tuples()
        assert all(kind != 'labels' for _, _, kind in layers) or all(
            'labels' in kw['name'] for _, kw, _ in layers
        )
