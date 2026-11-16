"""Writing napari layers to OME-Zarr, and reading them back as the same layers.

Every test that writes also reads the store back with ``nImage``: the
round trip is the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip('zarr')

from ndevio import (  # noqa: E402
    nImage,
    write_file_to_ome_zarr,
    write_files_to_ome_zarr,
    write_layers_to_ome_zarr,
)
from ndevio._save_normalize import SaveNormalizationError  # noqa: E402

UM = 'micrometer'


def _image(data, name, **attrs):
    base = {
        'name': name,
        'scale': (2.0, 0.5, 0.5),
        'axis_labels': ('Z', 'Y', 'X'),
        'units': (UM, UM, UM),
    }
    return (data, {**base, **attrs}, 'image')


def _labels(data, name, **attrs):
    data_, attrs_, _ = _image(data, name, **attrs)
    return (data_, attrs_, 'labels')


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, (4, 32, 32), dtype=np.uint16)


@pytest.fixture
def mask():
    lab = np.zeros((4, 32, 32), dtype=np.uint32)
    lab[:, 4:10, 4:10] = 1
    lab[:, 20:30, 20:30] = 2
    return lab


def _level0(data):
    """Level 0 of layer data, which nImage gives as a list of levels."""
    return np.asarray(data[0] if isinstance(data, list) else data)


def _kinds(path):
    return [kind for _, _, kind in nImage(path).get_layer_data_tuples()]


class TestStoreLayout:
    def test_is_ngff_05_on_zarr_v3(self, tmp_path, stack):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(str(path), [_image(stack, 'DAPI')])
        meta = json.loads((path / 'zarr.json').read_text())
        assert meta['zarr_format'] == 3
        assert meta['attributes']['ome']['version'] == '0.5'

    def test_labels_group_follows_the_image_label_spec(
        self, tmp_path, stack, mask
    ):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(
            str(path), [_image(stack, 'DAPI'), _labels(mask, 'nuclei')]
        )
        root = zarr.open_group(str(path), mode='r')
        assert root['labels'].attrs['ome'] == {
            'version': '0.5',
            'labels': ['nuclei'],
        }
        ome = root['labels/nuclei'].attrs['ome']
        assert ome['image-label']['source'] == {'image': '../../'}
        assert 'multiscales' in ome
        assert 'omero' not in ome
        assert 'ndevio' not in root.attrs

    def test_overwrites_an_existing_store(self, tmp_path, stack, mask):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(
            str(path), [_image(stack, 'DAPI'), _labels(mask, 'nuclei')]
        )
        write_layers_to_ome_zarr(str(path), [_image(stack, 'GFP')])
        assert _kinds(path) == ['image']
        assert not (path / 'labels').exists()


class TestImageRoundTrip:
    def test_names_scale_units_and_data(self, tmp_path, stack):
        path = tmp_path / 'a.ome.zarr'
        other = stack[::-1].copy()
        write_layers_to_ome_zarr(
            str(path), [_image(stack, 'DAPI'), _image(other, 'GFP')]
        )
        img = nImage(path)
        assert list(img.channel_names) == ['DAPI', 'GFP']
        assert img.layer_scale == (2.0, 0.5, 0.5)
        assert [str(u) for u in img.layer_units] == [UM] * 3
        layers = img.get_layer_data_tuples()
        np.testing.assert_array_equal(_level0(layers[0][0]), stack)
        np.testing.assert_array_equal(_level0(layers[1][0]), other)

    def test_colour_contrast_and_visibility(self, tmp_path, stack):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(
            str(path),
            [
                _image(
                    stack,
                    'DAPI',
                    colormap='blue',
                    contrast_limits=(100, 3000),
                ),
                _image(stack, 'GFP', colormap='#ff8000', visible=False),
            ],
        )
        dapi, gfp = (kw for _, kw, _ in nImage(path).get_layer_data_tuples())
        assert dapi['colormap'] == '#0000ff'
        assert dapi['contrast_limits'] == (100.0, 3000.0)
        assert gfp['colormap'] == '#ff8000'
        assert 'contrast_limits' not in gfp  # the full uint16 range
        assert gfp['visible'] is False

    def test_multiscale_keeps_its_levels(self, tmp_path, stack):
        path = tmp_path / 'a.ome.zarr'
        levels = [stack, stack[:, ::2, ::2], stack[:, ::4, ::4]]
        write_layers_to_ome_zarr(str(path), [_image(levels, 'DAPI')])
        data = nImage(path).get_layer_data_tuples()[0][0]
        assert [tuple(level.shape) for level in data] == [
            (4, 32, 32),
            (4, 16, 16),
            (4, 8, 8),
        ]

    def test_rgb_splits_to_red_green_blue_channels(self, tmp_path):
        path = tmp_path / 'rgb.ome.zarr'
        rgb = np.zeros((10, 12, 3), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 100, 150, 200
        write_layers_to_ome_zarr(
            str(path),
            [
                (
                    rgb,
                    {
                        'name': 'photo',
                        'scale': (0.5, 0.25),
                        'axis_labels': ('Y', 'X'),
                        'rgb': True,
                    },
                    'image',
                )
            ],
        )
        img = nImage(path)
        assert list(img.channel_names) == ['photo_R', 'photo_G', 'photo_B']
        colours = [kw['colormap'] for _, kw, _ in img.get_layer_data_tuples()]
        assert colours == ['#ff0000', '#00ff00', '#0000ff']

    def test_napari_placeholder_axes_and_pixel_units(self, tmp_path, stack):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(
            str(path),
            [
                (
                    stack,
                    {
                        'name': 'raw',
                        'axis_labels': ('-3', '-2', '-1'),
                        'units': ('pixel', 'pixel', 'pixel'),
                    },
                    'image',
                )
            ],
        )
        img = nImage(path)
        assert img.layer_axis_labels == ('Z', 'Y', 'X')
        assert img.layer_units == (None, None, None)


class TestLabelsRoundTrip:
    def test_image_and_labels_reopen_as_image_and_labels(
        self, tmp_path, stack, mask
    ):
        path = tmp_path / 'a.ome.zarr'
        write_layers_to_ome_zarr(
            str(path), [_image(stack, 'DAPI'), _labels(mask, 'nuclei')]
        )
        layers = nImage(path).get_layer_data_tuples()
        assert [kind for _, _, kind in layers] == ['image', 'labels']
        data, kwargs, _ = layers[1]
        assert kwargs['name'].startswith('nuclei')
        assert kwargs['scale'] == (2.0, 0.5, 0.5)
        np.testing.assert_array_equal(np.asarray(data), mask)

    def test_labels_get_the_image_pyramid(self, tmp_path, stack, mask):
        path = tmp_path / 'a.ome.zarr'
        levels = [stack, stack[:, ::2, ::2]]
        write_layers_to_ome_zarr(
            str(path), [_image(levels, 'DAPI'), _labels(mask, 'nuclei')]
        )
        data = nImage(path).get_layer_data_tuples()[-1][0]
        assert [tuple(level.shape) for level in data] == [
            (4, 32, 32),
            (4, 16, 16),
        ]
        # Nearest-neighbour: the coarse level holds only real label values.
        assert set(np.unique(np.asarray(data[1]))) <= {0, 1, 2}

    def test_direct_label_colours(self, tmp_path, stack, mask):
        path = tmp_path / 'a.ome.zarr'
        colormap = {None: (0.5, 0.5, 0.5, 1), 1: (1, 0, 0, 1), 2: (0, 0, 1, 1)}
        write_layers_to_ome_zarr(
            str(path),
            [
                _image(stack, 'DAPI'),
                _labels(mask, 'nuclei', colormap=colormap),
            ],
        )
        colours = nImage(path).get_layer_data_tuples()[-1][1]['colormap']
        assert colours[1] == pytest.approx((1, 0, 0, 1))
        assert colours[2] == pytest.approx((0, 0, 1, 1))

    def test_labels_alone_reopen_as_labels(self, tmp_path, mask):
        path = tmp_path / 'labels.ome.zarr'
        write_layers_to_ome_zarr(str(path), [_labels(mask, 'manual')])
        assert _kinds(path) == ['labels']

    def test_caller_override_beats_the_recorded_type(self, tmp_path, mask):
        path = tmp_path / 'labels.ome.zarr'
        write_layers_to_ome_zarr(str(path), [_labels(mask, 'manual')])
        tuples = nImage(path).get_layer_data_tuples(layer_type='image')
        assert [kind for _, _, kind in tuples] == ['image']

    def test_labels_must_share_the_image_grid(self, tmp_path, stack, mask):
        with pytest.raises(
            SaveNormalizationError, match='same axes and shape'
        ):
            write_layers_to_ome_zarr(
                str(tmp_path / 'a.ome.zarr'),
                [_image(stack, 'DAPI'), _labels(mask[:, :16], 'nuclei')],
            )


class TestNapariLayers:
    """Layers as napari hands them to a writer: ``as_layer_data_tuple()``."""

    def test_napari_layer_state_round_trips(self, tmp_path, stack, mask):
        layers_mod = pytest.importorskip('napari.layers')
        image = layers_mod.Image(
            stack,
            name='DAPI',
            colormap='magenta',
            contrast_limits=(10, 2000),
            scale=(2.0, 0.5, 0.5),
            units=UM,
            axis_labels=('Z', 'Y', 'X'),
        )
        labels = layers_mod.Labels(
            mask,
            name='nuclei',
            colormap={None: (0.5, 0.5, 0.5, 1), 1: (0, 1, 0, 1)},
            scale=(2.0, 0.5, 0.5),
            units=UM,
            axis_labels=('Z', 'Y', 'X'),
        )
        path = tmp_path / 'napari.ome.zarr'
        write_layers_to_ome_zarr(
            str(path),
            [image.as_layer_data_tuple(), labels.as_layer_data_tuple()],
        )

        back = [
            layers_mod.Layer.create(*t)
            for t in nImage(path).get_layer_data_tuples()
        ]
        new_image, new_labels = back
        assert isinstance(new_labels, layers_mod.Labels)
        assert new_image.colormap.name == 'magenta'
        assert tuple(new_image.contrast_limits) == (10, 2000)
        assert tuple(new_image.scale) == (2.0, 0.5, 0.5)
        assert [str(u) for u in new_labels.units] == [UM] * 3
        green = new_labels.colormap.map(np.array([1]))[0]
        assert green == pytest.approx((0, 1, 0, 1))
        np.testing.assert_array_equal(np.asarray(new_labels.data), mask)


def test_viewer_save_and_reopen(make_napari_viewer, tmp_path, stack, mask):
    """Save from a viewer with napari's own save, open it again."""
    viewer = make_napari_viewer()
    viewer.add_image(
        stack, name='DAPI', colormap='cyan', contrast_limits=(50, 1500)
    )
    painted = viewer.add_labels(mask, name='nuclei')

    path = tmp_path / 'session.ome.zarr'
    written = viewer.layers.save(str(path), plugin='ndevio')
    assert written == [str(path)]

    viewer.layers.clear()
    viewer.open(str(path), plugin='ndevio')
    assert [type(layer).__name__ for layer in viewer.layers] == [
        'Image',
        'Labels',
    ]
    image, labels = viewer.layers
    assert image.colormap.name == 'cyan'
    assert tuple(image.contrast_limits) == (50, 1500)
    assert labels.name.startswith('nuclei')
    np.testing.assert_array_equal(np.asarray(labels.data), painted.data)


class TestFiles:
    def test_tiff_converts_with_names_and_scale(self, resources_dir, tmp_path):
        source = resources_dir / 'cells3d2ch_legacy.tiff'
        destination = tmp_path / 'converted.ome.zarr'
        assert write_file_to_ome_zarr(source, destination) == [
            str(destination)
        ]
        before, after = nImage(source), nImage(destination)
        assert list(after.channel_names) == list(before.channel_names)
        assert after.layer_scale == before.layer_scale
        assert after.reference_xarray.shape == before.reference_xarray.shape

    def test_ome_zarr_source_keeps_its_labels(self, tmp_path, stack, mask):
        source = tmp_path / 'source.ome.zarr'
        write_layers_to_ome_zarr(
            str(source), [_image(stack, 'DAPI'), _labels(mask, 'nuclei')]
        )
        destination = tmp_path / 'copy.ome.zarr'
        write_file_to_ome_zarr(source, destination)
        assert _kinds(destination) == ['image', 'labels']
        names = [
            kw['name'].split(' :: ')[0]
            for _, kw, _ in nImage(destination).get_layer_data_tuples()
        ]
        assert names == ['DAPI', 'nuclei']

    def test_batch(self, resources_dir, tmp_path):
        sources = [
            resources_dir / 'cells3d2ch_legacy.tiff',
            resources_dir / 'dimension_handling_zyx_V3.zarr',
        ]
        out = tmp_path / 'out'
        written = write_files_to_ome_zarr(sources, out)
        assert written == [
            str(out / 'cells3d2ch_legacy.ome.zarr'),
            str(out / 'dimension_handling_zyx_V3.ome.zarr'),
        ]
        source_zarr = nImage(sources[1])
        saved_zarr = nImage(Path(written[1]))
        assert len(saved_zarr.resolution_levels) == len(
            source_zarr.resolution_levels
        )
