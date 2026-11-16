from typing import TYPE_CHECKING

try:  # noqa: D104
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

if TYPE_CHECKING:
    from ._writer_ome_zarr import (
        write_file_to_ome_zarr as write_file_to_ome_zarr,
    )
    from ._writer_ome_zarr import (
        write_files_to_ome_zarr as write_files_to_ome_zarr,
    )
    from ._writer_ome_zarr import (
        write_layers_to_ome_zarr as write_layers_to_ome_zarr,
    )
    from .nimage import nImage as nImage
    from .utils import helpers as helpers

_WRITERS = frozenset(
    {
        'write_file_to_ome_zarr',
        'write_files_to_ome_zarr',
        'write_layers_to_ome_zarr',
    }
)


def __getattr__(name: str) -> object:
    """Lazily import heavy submodules to speed up package import."""
    if name == 'nImage':
        from .nimage import nImage

        return nImage
    if name == 'helpers':
        from .utils import helpers

        return helpers
    if name in _WRITERS:
        from . import _writer_ome_zarr

        return getattr(_writer_ome_zarr, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    '__version__',
    'helpers',
    'nImage',
    'write_file_to_ome_zarr',
    'write_files_to_ome_zarr',
    'write_layers_to_ome_zarr',
]
