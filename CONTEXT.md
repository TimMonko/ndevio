# ndevio — Domain Glossary (CONTEXT.md)

A shared vocabulary for ndevio. Used by architecture reviews and design
discussions to keep terms consistent. Add or sharpen terms here as they
crystallize.

## Core concepts

- **nImage** — the primary module: a `BioImage` subclass that produces
  napari-ready layer data. It is both a *loader* (reader selection, scenes,
  resolution levels, dask/memory decisions) and the entry point for layer
  translation.
- **BioImage** — the input to the layer metadata translator. Typed as
  `BioImage` (an `nImage` in production); the translator only touches its
  `scale`, `dimension_properties`, `metadata`, and `ome_metadata` accessors,
  so tests pass a lightweight stand-in.
- **Layer metadata translator** — the deep module in
  `utils/_layer_metadata.py` that resolves physical metadata (axis labels,
  scale, units, metadata dict) from a `BioImage`. Owns all OME-Zarr
  spec-version quirks and fallbacks; the `nImage.layer_*` properties are thin
  delegations that call it directly.
- **LayerMeta / `LayerMetadata`** — the typed result of the translator:
  `axis_labels`, `scale`, `units`, `metadata`, aligned 1:1 with the (squeezed)
  layer data dims.
- **Layer data tuple** — the `(data, metadata, layer_type)` triple napari
  consumes via `Layer.create`. Built by `build_layer_tuple`.
- **Scene** — a named series inside a file (`img.scenes`). Switching scenes
  (`nImage.set_scene`) invalidates the per-scene layer-data caches so the next
  access reloads.
- **Squeezed dims** — the dimension names of the layer data after singleton
  dims are removed; `Channel`/`Samples` are excluded from the exposed axis
  labels (channels become separate layers; RGB is one layer).
- **OME-Zarr compatibility patches** — load-time adapters in
  `bioio_plugins/_compatibility.py` that mutate old `bioio_ome_zarr` reader
  internals so downstream metadata access works. Distinct from the
  translator's access-time fallbacks.
- **Scene action** — how a file's scenes are opened: `SceneAction.OPEN_FIRST`,
  `OPEN_ALL`, or `OPEN_WIDGET`. Decided purely by `resolve_scene_action()` in
  `_napari_reader.py` (single scene or `open_first` wins; then `open_all`;
  else the widget); the `scene_handling` setting maps to the two flags via
  `resolve_scene_flags()`.

## Architecture vocabulary

Use the terms from the `/codebase-design` skill for anything architectural:
**module**, **interface**, **implementation**, **depth**, **seam**, **adapter**,
**leverage**, **locality**. Do not substitute "component", "service", "API",
or "boundary".

## Key decisions (see ADRs when present)

- The metadata translator lives in `utils/_layer_metadata.py` (metadata only —
  data loading stays on `nImage`).
- `_compatibility.py` stays a load-time adapter; the translator owns the
  residual fallbacks.
- `nImage.set_scene` is the single place that invalidates per-scene caches.
- The reader scene policy is pure: `napari_reader_function` decides via
  `resolve_scene_action()` then acts; UI-opening (scene widget, plugin
  installer) lives behind the widget adapters, so the npe2 reader hook is
  testable without a viewer.
