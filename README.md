# noodles

some comfyui nodes for the purposes of committing war crimes with video models

also just some misc utility nodes

## Quickstart

1. Probably don't use this. It doesn't even qualify as a "WIP".
1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- A list of features

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd noodles
uv pip install -e '.[dev]'
pre-commit install
```

This assumes you have `uv`. If you do not have `uv`, you should get `uv`. You can also try just
`pip install -e '.[dev]'` and that should work probably.

## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. They are violently incomplete
and probably not worth running. I may vibe some up later once I'm sorta happy with how any of this works.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes
