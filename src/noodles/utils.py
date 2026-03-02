import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from comfy_api.latest import ComfyAPISync
from folder_paths import get_input_directory, get_output_directory
from PIL import Image
from ulid import ULID

api = ComfyAPISync()


def get_input_dir_path() -> Path:
    return Path(get_input_directory())


def get_output_dir_path() -> Path:
    return Path(get_output_directory())


def get_folders_in_outdir(depth: int = 2) -> list[str]:
    output_root = get_output_dir_path()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_root}")
    if depth == 1:
        # short-circuit
        return sorted([f.name for f in output_root.iterdir() if f.is_dir()])

    folders = set()
    for f in output_root.rglob("*"):
        if not f.is_dir():
            continue
        try:
            subpath = f.relative_to(output_root)
            if len(subpath.parts) <= depth:
                folders.add(str(subpath))
        except ValueError:
            continue  # skip folders that can't be relativized for some reason
    return sorted(folders, key=lambda s: s.lower())


def prune_dict(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def parse_ulid(
    value: ULID | str | int | bytes | None,
    field_name: str | None = None,
    *,
    optional: bool = False,
) -> ULID | None:
    field_name = f"ULID field '{field_name}'" if field_name else "ULID value"

    match value:
        case ULID():
            return value
        case str() if value.strip():  # non-empty string
            value = value.strip().upper()
            # Strip single-character prefix if present to make extracting from filenames easier
            if len(value) == 27 and value[0].isalpha():
                value = value[1:]
            return ULID.parse(value)
        case None | str():  # None or empty string
            if optional:
                return None
            raise ValueError(f"Got None or empty string for {field_name}")
        case _:
            try:
                return ULID.parse(value)
            except Exception as e:
                raise ValueError(f"Could not parse value '{value!r}' of type '{type(value)}' for {field_name}") from e


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def ndimage_to_webp(arr: np.ndarray, quality: int = 80, method: int = 3) -> BytesIO:
    pil_image = Image.fromarray(arr)
    buf = BytesIO()
    pil_image.save(buf, format="WebP", lossless=True, quality=quality, method=method)
    return buf.getbuffer()


@torch.inference_mode()
def compress_image_tensor_webp(
    images: torch.Tensor,  # [N, H, W, C]
    return_padded: bool = True,
    report_progress: bool = False,
) -> torch.Tensor:
    if images.ndim == 3:
        images = images.unsqueeze(0)  # add batch dimension for easier processing

    if images.dtype == torch.float32 and (images.min() <= 0.0 or images.max() >= 1.0):
        images = images.mul(255.0).clamp_(0, 255)
    elif images.dtype != torch.uint8:
        raise ValueError(f"Unsupported image tensor dtype: {images.dtype}")

    # force is equivalent to .detach().cpu().resolve_conj().resolve_neg().numpy()
    ndimages: np.ndarray = images.numpy(force=True).astype(np.uint8)  # (N, H, W, C)

    webp_tensors = []
    for img in ndimages:
        webp_buf = ndimage_to_webp(img, quality=0, method=0)
        webp_tensors.append(torch.frombuffer(webp_buf, dtype=torch.uint8))
        if report_progress:
            api.execution.set_progress(value=len(webp_tensors), max_value=len(ndimages))

    compressed_tensor = torch.nested.nested_tensor(webp_tensors, dtype=torch.uint8, layout=torch.jagged)
    if return_padded:
        compressed_tensor = compressed_tensor.to_padded_tensor(0)
    return compressed_tensor.contiguous().to(images.device)


@torch.inference_mode()
def decompress_image_tensor_webp(
    images: torch.Tensor,
    size: tuple[int, int],  # (W, H as PIL)
    as_float: bool = False,
    report_progress: bool = False,
) -> torch.Tensor:
    # decompress a zero-padded [N, max_len] uint8 tensor of WebP bytes back into a [N, H, W, C] image tensor
    if images.ndim != 2 or images.dtype != torch.uint8:
        raise ValueError("Expected a 2D uint8 tensor of WebP bytes")

    n_images = images.shape[0]
    image_tensor = torch.empty((n_images, *size[::-1], 3), dtype=torch.uint8)  # (N, H, W, C)
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        for idx, img in enumerate(images):
            with BytesIO(img.numpy(force=True).tobytes()) as buf:
                with Image.open(buf) as img:
                    image_tensor[idx] = torch.from_numpy(np.asarray(img, dtype=np.uint8))
            if report_progress:
                api.execution.set_progress(value=idx + 1, max_value=n_images)
    if as_float:
        image_tensor = image_tensor.to(torch.float32).div_(255.0)

    return image_tensor.contiguous().clone().to(images.device)
