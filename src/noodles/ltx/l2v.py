import json
import warnings
from datetime import UTC, datetime
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from comfy.sd import VAE
from comfy.utils import load_torch_file, save_torch_file
from comfy_api.latest import ComfyAPI, LatentInput, io
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator
from pydantic.types import AwareDatetime
from ulid import ULID

from ..utils import (
    ValidateAnyMixin,
    compress_image_tensor_webp,
    decompress_image_tensor_webp,
    get_folders_in_outdir,
    get_output_dir_path,
    parse_ulid,
)
from .common import (
    SEGMENT_METADATA_KEY,
    BootstrapMode,
    ComfyULID,
    MaskParams,
    MaskStrategy,
    get_mask_decay_curve,
)
from .io import BootstrapModeIO, MaskParamsIO, MaskStrategyIO
from .paths import find_segment_file, get_next_segment_iteration, list_segment_files

_API = ComfyAPI()


class LTXLat2VidSegmentData(BaseModel, ValidateAnyMixin):
    video_id: ULID = Field(default_factory=ULID)
    parent_id: ULID | None = None
    segment_id: ULID = Field(default_factory=ULID)

    segment_idx: int = Field(default=0, min=0, description="Index of the segment within the video.")
    iteration: int = Field(default=0, min=0, description="Iteration number for this segment.")

    fps: float = Field(24.0, gt=0, description="Framerate for this segment.")
    start_frame: int = Field(
        ...,
        min=0,
        description="Global frame index for the start of this segment. Used for video assembly and overlap calculations.",
    )
    n_frames: int = Field(
        ...,
        min=1,
        description="Number of good frames in the frame tensor for this segment. Includes overlap at the start, "
        + "and the bootstrap frame for segment 0. start_frame + n_frames - overlap_frames = next segment's start_frame.",
    )
    n_frames_batch: int = Field(
        ...,
        min=1,
        description="Number of frames generated for this segment, including overlap and bootstrap frame. Used to make future segments the same length.",
    )

    n_latents: int = Field(
        ...,
        min=2,
        description="Number of latents stored in the latent tensor, inc. bootstrap, excl. discarded terminal (if enabled)",
    )
    overlap_k: int = Field(
        6,
        ge=2,
        le=512,
        description="Number of latents at the end of the segment that will be overlapped with the next segment.",
    )
    keep_bootstrap: bool = Field(
        False,
        description="if True, the first (bootstrap) frame in the frames tensor will be kept during video assembly. "
        + "Generally you only want to keep the bootstrap frame from segment 0.",
    )
    drop_last_latent: bool = Field(
        True,
        description="Whether to drop the last latent before saving. The last latent is usually low-quality and best discarded.",
    )

    bootstrap_mode: BootstrapMode = BootstrapMode.DummyLatent
    mask_params: MaskParams = Field(default_factory=MaskParams)
    mask_strat: MaskStrategy | None = Field(
        None,
        exclude=True,
        deprecated="Use mask_params.strategy field instead",
    )

    video_name: str = Field(default="untitled", description="Video name for organization purposes.")
    subfolder: str = Field(..., description="Subfolder that this segment was saved in originally.")
    width_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Width of video in pixels")
    height_px: int = Field(..., ge=256, le=4096, multiple_of=32, description="Height of video in pixels")

    prompt: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)
    extra_pnginfo: SerializeAsAny[dict[str, Any]] = Field(default_factory=dict)

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    model_config: ConfigDict = ConfigDict(
        extra="ignore",
    )

    @model_validator(mode="after")
    def update_mask_strategy(self) -> Any:
        # prevents the validator emitting a deprecation warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            # move mask strategy from deprecated top-level field to mask_params if it's set and mask_params.strategy is still default
            if self.mask_strat is not None and self.mask_params.strategy == MaskStrategy.NoStrategy:
                self.mask_params.strategy = self.mask_strat
        return self


@io.comfytype(io_type="LAT2VID_SEGMENT")
class LTXLat2VidSegmentIO(io.ComfyTypeIO):
    if TYPE_CHECKING:
        Type = LTXLat2VidSegmentData

    class Input(io.Input):
        pass

    class Output(io.Output):
        pass


class LTXLat2VidSegmentChainData(BaseModel, ValidateAnyMixin):
    video_id: ULID
    video_name: str
    video_folder: str
    head_segment_id: ULID
    tail_segment_id: ULID
    segment_ids: list[ULID]
    segment_paths: list[str]
    segment_count: int = Field(..., ge=1)

    model_config: ConfigDict = ConfigDict(
        extra="ignore",
    )


@io.comfytype(io_type="LAT2VID_SEGMENT_CHAIN")
class LTXLat2VidSegmentChainIO(io.ComfyTypeIO):
    if TYPE_CHECKING:
        Type = LTXLat2VidSegmentChainData

    class Input(io.Input):
        pass

    class Output(io.Output):
        pass


def _segment_data_from_headers(raw_metadata: dict[str, str]) -> tuple[LTXLat2VidSegmentData, str]:
    metadata_json = raw_metadata.get(SEGMENT_METADATA_KEY)
    if not metadata_json:
        raise ValueError(f"No '{SEGMENT_METADATA_KEY}' metadata found in checkpoint")

    segment_data = LTXLat2VidSegmentData.model_validate_json(metadata_json)

    update = {}
    for key in ["prompt", "extra_pnginfo"]:
        if key in raw_metadata and raw_metadata[key]:
            update[key] = raw_metadata[key]
    if update:
        segment_data = segment_data.model_copy(update=update)

    return segment_data, segment_data.model_dump_json()


def _resolve_segment_path(segment_path: PathLike[str]) -> Path:
    segment_path = Path(segment_path).expanduser()
    if not segment_path.is_absolute():
        segment_path = get_output_dir_path() / segment_path
    resolved_path = segment_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Segment file does not exist: {resolved_path}")
    return resolved_path


def _read_safetensors_metadata(segment_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(segment_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    if not isinstance(metadata, dict):
        return {}

    out = {}
    for key, value in metadata.items():
        if value is None:
            continue
        out[str(key)] = value if isinstance(value, str) else json.dumps(value)
    return out


def load_segment_file(
    segment_path: PathLike[str],
) -> tuple[dict[str, torch.Tensor], LTXLat2VidSegmentData, str, Path]:
    resolved_path = _resolve_segment_path(segment_path)
    state_dict, raw_metadata = load_torch_file(str(resolved_path), return_metadata=True)
    segment_data, metadata_json = _segment_data_from_headers(raw_metadata)
    return state_dict, segment_data, metadata_json, resolved_path


def load_segment_metadata(
    segment_path: PathLike[str],
) -> tuple[LTXLat2VidSegmentData, str, Path]:
    resolved_path = _resolve_segment_path(segment_path)
    raw_metadata = _read_safetensors_metadata(resolved_path)
    segment_data, metadata_json = _segment_data_from_headers(raw_metadata)
    return segment_data, metadata_json, resolved_path


def get_video_folder_from_segment_metadata(metadata: LTXLat2VidSegmentData) -> str:
    video_dirname = f"{metadata.video_name}_{metadata.video_id}"
    folder_prefix = str(metadata.subfolder or "").replace(video_dirname, "").rstrip("\\/")
    return f"{folder_prefix}/{video_dirname}" if folder_prefix else video_dirname


def resolve_segment_chain(
    video_folder: str,
    tail_segment_id: ULID | str | None = None,
    *,
    max_depth: int = 4096,
) -> LTXLat2VidSegmentChainData:
    segment_files = list_segment_files(video_folder, return_tuple=True)
    if not segment_files:
        raise FileNotFoundError(f"No segment files found in folder: {video_folder}")

    all_paths: list[Path] = [path for _, _, path, _ in segment_files]
    suffix_index: dict[str, list[Path]] = {}
    for _, _, path, id_suffix in segment_files:
        if id_suffix:
            suffix_index.setdefault(id_suffix.lower(), []).append(path)

    metadata_cache: dict[Path, LTXLat2VidSegmentData] = {}
    id_to_path: dict[str, Path] = {}

    def load_meta(path: Path) -> LTXLat2VidSegmentData:
        if path in metadata_cache:
            return metadata_cache[path]

        metadata, _, _ = load_segment_metadata(path)
        metadata_cache[path] = metadata
        id_to_path[str(metadata.segment_id)] = path
        return metadata

    def find_by_segment_id(segment_id: ULID) -> Path:
        segment_id_str = str(segment_id)
        if segment_id_str in id_to_path:
            return id_to_path[segment_id_str]

        suffix = segment_id_str[-6:].lower()
        candidates = suffix_index.get(suffix, [])
        if not candidates:
            raise FileNotFoundError(f"Could not find segment_id={segment_id_str} in folder: {video_folder}")
        if len(candidates) == 1:
            return candidates[0]

        # Rare ULID suffix collisions: disambiguate by checking full metadata ID.
        for path in candidates:
            meta = load_meta(path)
            if str(meta.segment_id) == segment_id_str:
                return path

        raise FileNotFoundError(f"Could not disambiguate segment_id={segment_id_str} in folder: {video_folder} (suffix collision)")

    target_tail_id = parse_ulid(tail_segment_id, "tail_segment_id", optional=True)
    if target_tail_id is not None:
        tail_path = find_by_segment_id(target_tail_id)
        tail_meta = load_meta(tail_path)
    else:
        tail_path = all_paths[-1]
        tail_meta = load_meta(tail_path)

    chain_paths_rev: list[Path] = []
    chain_ids_rev: list[ULID] = []
    seen_ids: set[str] = set()
    expected_video_id = str(tail_meta.video_id)

    current_path = tail_path
    current_meta = tail_meta
    for _ in range(max_depth):
        current_segment_id = str(current_meta.segment_id)
        if current_segment_id in seen_ids:
            raise ValueError(f"Cycle detected in segment parent chain at segment_id={current_segment_id}")
        seen_ids.add(current_segment_id)

        if str(current_meta.video_id) != expected_video_id:
            raise ValueError(
                f"Cross-video parent chain detected for segment_id={current_segment_id}: "
                + f"expected video_id={expected_video_id}, got {current_meta.video_id}"
            )

        chain_paths_rev.append(current_path)
        chain_ids_rev.append(current_meta.segment_id)

        parent_id = current_meta.parent_id
        if parent_id is None:
            break
        parent_id_str = str(parent_id)
        # Segment-0 often uses video_id as parent_id sentinel. Treat this as a valid root marker.
        if parent_id_str == expected_video_id:
            break
        # Self-parenting is effectively a root/cycle sentinel in some legacy metadata.
        if parent_id_str == current_segment_id:
            break

        try:
            current_path = find_by_segment_id(parent_id)
        except FileNotFoundError:
            # If we're already at the earliest known segment, stop instead of failing hard.
            if int(current_meta.segment_idx) <= 0:
                break
            raise
        current_meta = load_meta(current_path)
    else:
        raise ValueError(f"Exceeded max_depth={max_depth} while resolving segment chain in folder: {video_folder}")

    chain_paths = [str(path) for path in reversed(chain_paths_rev)]
    chain_ids = list(reversed(chain_ids_rev))

    return LTXLat2VidSegmentChainData(
        video_id=tail_meta.video_id,
        video_name=tail_meta.video_name,
        video_folder=video_folder,
        head_segment_id=chain_ids[0],
        tail_segment_id=chain_ids[-1],
        segment_ids=chain_ids,
        segment_paths=chain_paths,
        segment_count=len(chain_paths),
    )


def _compute_overlap_strengths(
    total_k: int,
    bootstrap_mode: BootstrapMode,
    bootstrap_strength: float | None = None,
    mask_params: MaskParams = ...,
) -> list[float]:
    if total_k < 2:
        raise ValueError("total_k must be at least 2")

    match bootstrap_mode:
        case BootstrapMode.SegmentZero:
            raise ValueError("SegmentZero bootstrap mode is invalid for continuation segments")
        case BootstrapMode.RawLatent:
            bootstrap_strength = 1.0
        case BootstrapMode.DummyLatent:
            # For Dummy Latent continuation, bootstrap latent should be freely diffused (strength 0.0)
            bootstrap_strength = 0.0
        case BootstrapMode.VAERoundtrip:
            # For VAE Roundtrip, the bootstrap latent is a VAE roundtrip of the last frame and may be somewhat noisy/artifacted,
            # so we give it a moderate strength to allow some correction while still being anchored to the source.
            bootstrap_strength = bootstrap_strength if bootstrap_strength is not None else 0.5

    # and overlap latents should be masked according to selected strategy.
    overlap_strengths = get_mask_decay_curve(
        mask_strat=mask_params.strategy,
        total_k=total_k - 1,
        hard_mask_k=mask_params.hard_mask_k,
        w_max=mask_params.w_max,
        w_min=mask_params.w_min,
        decay_sigma=mask_params.decay_sigma,
    )
    return [bootstrap_strength, *overlap_strengths]


def _deterministic_seed(
    prev_segment_data: LTXLat2VidSegmentData,
    seed: int,
    overlap_k: int,
) -> int:
    seed_key = f"{prev_segment_data.video_id}|{prev_segment_data.segment_id}|{prev_segment_data.segment_idx}|{seed}|{overlap_k}"
    digest = sha256(seed_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class LTXLat2VidSegmentSaveNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidSegmentSaveNood",
            display_name="LTX-L2V Segment Save",
            category="noodles/ltx",
            is_experimental=True,
            inputs=[
                io.Image.Input(
                    "images",
                    display_name="frames",
                    optional=False,
                    tooltip="Frames for this segment, including overlap. Overlap will be trimmed before saving.",
                ),
                io.String.Input(
                    "folder_prefix",
                    display_name="Folder Prefix",
                    multiline=False,
                    default="ltx/",
                    tooltip="Folder prefix for the saved segment data. A subdirectory will be created for each video with the format {video_name}_{video_id}. Segments will be stored as safetensors files within the video directory.",
                ),
                io.Latent.Input("latent", tooltip="The latents for this segment"),
                ComfyULID.Input(
                    "video_id",
                    optional=True,
                    default=None,
                    force_input=True,
                    tooltip="Optional video ID. If not provided, a new one will be generated.",
                ),
                ComfyULID.Input(
                    "parent_id",
                    optional=True,
                    default=None,
                    force_input=True,
                    tooltip="Optional parent segment ID for hierarchical video structuring",
                ),
                io.String.Input(
                    "video_name",
                    display_name="Video Name",
                    default="untitled",
                    tooltip="Optional video name for organizational purposes. Not used for processing.",
                ),
                io.Int.Input(
                    "segment_idx",
                    display_name="Segment Index",
                    default=0,
                    min=0,
                    tooltip="Index of the segment within the video. Used for ordering segments during video assembly.",
                ),
                io.Int.Input(
                    "start_frame",
                    display_name="Start Frame",
                    default=0,
                    min=0,
                    tooltip="The starting frame index for this segment. Used for video assembly and overlap calculations.",
                ),
                io.Int.Input(
                    "n_frames_batch",
                    display_name="Batch Frames",
                    default=1,
                    min=1,
                    tooltip="The number of frames generated for this run (including overlap and bootstrap)",
                ),
                io.Int.Input(
                    "overlap_k",
                    display_name="Overlapped Latents",
                    default=6,
                    min=2,
                    tooltip="Number of latents at the end of the segment that will be overlapped with the next segment.",
                ),
                io.Boolean.Input(
                    "drop_last_latent",
                    display_name="Last Latent",
                    default=True,
                    label_on="Drop",
                    label_off="Keep",
                    tooltip="Whether to drop the last latent before saving. The last latent is usually low-quality and best discarded.",
                ),
                BootstrapModeIO.Input("bootstrap_mode", default=BootstrapMode.DummyLatent),
                MaskParamsIO.Input("mask_params"),
            ],
            outputs=[
                ComfyULID.Output(display_name="video_id", tooltip="ULID for the video that this segment belongs to."),
                ComfyULID.Output(display_name="segment_id", tooltip="This segment's ULID, unique per iteration"),
                io.String.Output(
                    display_name="segment_path",
                    tooltip="Full path to the saved segment file, including filename. Useful for debugging and chaining nodes that need to read the file.",
                ),
                io.String.Output(
                    display_name="video_prefix",
                    tooltip="File prefix for the saved segment. Feed to video save node if saving decoded video.",
                ),
                io.String.Output(
                    display_name="frames_prefix",
                    tooltip="Prefix to use if saving frames for this segment. Feed to frame save node if saving frames.",
                ),
                io.Image.Output(
                    display_name="frames",
                    tooltip="Frames for this segment, excluding overlap, for visualization purposes. Not used for processing.",
                ),
                LTXLat2VidSegmentIO.Output(
                    display_name="metadata", tooltip="Segment metadata as a structured object for use in downstream nodes."
                ),
                io.String.Output(display_name="metadata_json", tooltip="Full segment metadata as JSON string for debugging purposes."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        folder_prefix: str,
        latent: LatentInput,
        video_id: ULID | str | None,
        parent_id: ULID | str | None,
        video_name: str,
        segment_idx: int,
        start_frame: int,
        n_frames_batch: int,
        overlap_k: int,
        drop_last_latent: bool,
        bootstrap_mode: BootstrapMode,
        mask_params: MaskParams,
    ) -> io.NodeOutput:
        mask_params = MaskParams.model_validate_any(mask_params, strict=False)
        bootstrap_mode = BootstrapMode(bootstrap_mode)

        video_ulid = parse_ulid(video_id, "video_id", optional=True) or ULID()
        parent_ulid = parse_ulid(parent_id, "parent_id", optional=True)
        segment_id = ULID()

        # get output root
        output_root = get_output_dir_path()

        # build name of segment save folder
        video_dirname = f"{video_name}_{video_ulid}"
        # strip out the video_dirname from the folder_prefix (if present), trim slashes for consistency
        folder_prefix = folder_prefix.replace(video_dirname, "").rstrip("\\/")
        # then reassemble to ensure consistent formatting (no double slashes, etc)
        video_folder = f"{folder_prefix}/{video_dirname}"

        # build the segment filename prefix (minus iteration and ID suffix)
        filename_prefix = f"{video_name}.v{str(video_ulid)[:10]}.s{segment_idx:03d}"
        # Get the next iteration number for this segment index to avoid overwriting existing files.
        counter = get_next_segment_iteration(output_root / video_folder / (filename_prefix + ".safetensors"))
        # assemble the final segment filename
        seg_filename = f"{filename_prefix}_i{counter:03d}.{str(segment_id)[-6:]}"

        # build full segment path and ensure parent directories exist
        segment_path = output_root / video_folder / (seg_filename + ".safetensors")
        segment_path.parent.mkdir(parents=True, exist_ok=True)

        # tail end of the aux file (frames/video) output prefixes
        aux_file_suffix = f"s{segment_idx:03d}_i{counter:03d}.{str(segment_id)[-4:]}"

        # build the output prefixes for the other save nodes
        frames_prefix = f"{video_folder}/{seg_filename}/frames.{aux_file_suffix}/frame"
        output_root.joinpath(frames_prefix).parent.mkdir(parents=True, exist_ok=True)
        video_prefix = f"{video_folder}/{seg_filename}/video.{aux_file_suffix}"
        output_root.joinpath(video_prefix).parent.mkdir(parents=True, exist_ok=True)

        samples: torch.Tensor = latent["samples"].cpu().contiguous()
        if samples.ndim != 5:
            raise ValueError(f"Expected latent['samples'] with shape [B,C,T,H,W], got {samples.shape!r}")

        if drop_last_latent:
            samples = samples[:, :, :-1, :, :]
            images = images[:-8]

        bootstrap_frame = images[:1]
        save_images = images if segment_idx == 0 else images[1:]

        width_px = int(save_images.shape[2])
        height_px = int(save_images.shape[1])
        n_latents = int(samples.shape[2])
        n_frames = int(save_images.shape[0])

        metadata = LTXLat2VidSegmentData(
            video_id=video_ulid,
            parent_id=parent_ulid,
            segment_id=segment_id,
            segment_idx=segment_idx,
            iteration=counter,
            start_frame=start_frame,
            n_frames=n_frames,
            n_frames_batch=n_frames_batch,
            n_latents=n_latents,
            overlap_k=overlap_k,
            keep_bootstrap=segment_idx == 0,
            drop_last_latent=drop_last_latent,
            bootstrap_mode=bootstrap_mode,
            mask_params=mask_params,
            video_name=video_name,
            subfolder=folder_prefix,
            width_px=width_px,
            height_px=height_px,
            prompt=cls.hidden.prompt,
            extra_pnginfo=cls.hidden.extra_pnginfo,
        )

        compressed_frames = compress_image_tensor_webp(save_images, report_progress=True)
        compressed_bootstrap = compress_image_tensor_webp(bootstrap_frame)

        output = {
            "latent": samples.cpu().contiguous(),
            "bootstrap_latent": samples[:, :, :1, :, :].cpu().contiguous(),
            "compressed_frames": compressed_frames.cpu().contiguous(),
            "compressed_bootstrap": compressed_bootstrap.cpu().contiguous(),
            "ltx_l2v_format_v0": torch.tensor([]),
        }

        save_torch_file(
            output,
            segment_path,
            metadata={
                SEGMENT_METADATA_KEY: metadata.model_dump_json(exclude={"prompt", "extra_pnginfo"}),
                "prompt": json.dumps(cls.hidden.prompt),
                "extra_pnginfo": json.dumps(cls.hidden.extra_pnginfo),
            },
        )

        # make segment_path relative to outdir before outputting
        segment_path = segment_path.relative_to(get_output_dir_path())

        return io.NodeOutput(
            str(video_ulid),
            str(segment_id),
            str(segment_path),
            video_prefix,
            frames_prefix,
            save_images,
            metadata,
            metadata.model_dump_json(exclude={"prompt", "extra_pnginfo"}, indent=2),
        )


class LTXLat2VidSegmentLoadNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXLat2VidSegmentLoadNood",
            display_name="LTX-L2V Segment Loader",
            category="noodles/ltx",
            inputs=[
                io.Combo.Input(
                    "video_folder",
                    options=get_folders_in_outdir(depth=2),
                    tooltip="Subfolder to load segments from. Can be connected to save node's 'subfolder' output.",
                ),
                io.Int.Input(
                    "segment_idx",
                    default=0,
                    min=0,
                    tooltip="Segment index to load.",
                ),
                io.Int.Input(
                    "iteration",
                    default=0,
                    min=0,
                    tooltip="Iteration index to load. 0 loads the latest iteration for the segment.",
                ),
                io.Boolean.Input(
                    "metadata_only",
                    default=False,
                    label_on="Metadata Only",
                    label_off="Full Segment",
                    tooltip="Whether to load only the metadata, or also load the state dict for this segment.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Image.Output(display_name="frames"),
                LTXLat2VidSegmentIO.Output(display_name="metadata"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        video_folder: str,
        segment_idx: int,
        iteration: int,
        metadata_only: bool,
    ) -> io.NodeOutput:
        segment_path = find_segment_file(video_folder, segment_idx, iteration)
        if metadata_only:
            metadata, _, _ = load_segment_metadata(segment_path)
        else:
            state_dict, metadata, _, _ = load_segment_file(segment_path)
            if "latent" not in state_dict:
                raise ValueError("Segment file is missing required 'latent' tensor")
            if "frames" not in state_dict and "compressed_frames" not in state_dict:
                raise ValueError("Segment file does not contain a frame tensor")

        if not metadata_only:
            samples = state_dict["latent"].cpu().contiguous()
            if "frames" in state_dict:
                frames = state_dict["frames"].cpu().contiguous()
            elif "compressed_frames" in state_dict:
                frames = decompress_image_tensor_webp(
                    state_dict["compressed_frames"].cpu().contiguous(),
                    (metadata.width_px, metadata.height_px),
                    as_float=True,
                )
            else:
                raise ValueError("Segment file does not contain a frame tensor")
        else:
            samples = torch.empty(0)
            frames = torch.empty(0)

        latent: dict[str, torch.Tensor] = {"samples": samples}
        return io.NodeOutput(
            latent,
            frames,
            metadata,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        video_folder: str,
        segment_idx: int,
        iteration: int,
        metadata_only: bool,
    ) -> str:
        segment_path = find_segment_file(video_folder, segment_idx, iteration).resolve()
        segment_stat = segment_path.stat()

        fingerprint = sha256()
        fingerprint.update(str(metadata_only).encode("utf-8"))
        fingerprint.update(str(segment_path).encode("utf-8"))
        fingerprint.update(str(segment_stat.st_size).encode("utf-8"))
        fingerprint.update(str(segment_stat.st_mtime_ns).encode("utf-8"))
        return fingerprint.hexdigest()


class LTXLat2VidResolveSegmentChainNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidResolveSegmentChainNood",
            display_name="LTX-L2V Resolve Segment Chain",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentIO.Input(
                    "metadata",
                    display_name="metadata",
                    tooltip="Segment metadata from LTX-L2V Segment Loader. This segment is used as the chain tail.",
                ),
            ],
            outputs=[
                LTXLat2VidSegmentChainIO.Output(display_name="chain"),
                io.String.Output(
                    display_name="segment_paths_json",
                    tooltip="JSON list of resolved segment paths in playback order (start -> end).",
                ),
                ComfyULID.Output(display_name="video_id"),
                ComfyULID.Output(display_name="head_id"),
            ],
        )

    @classmethod
    def execute(
        cls,
        metadata: LTXLat2VidSegmentData | dict | str,
    ) -> io.NodeOutput:
        metadata = LTXLat2VidSegmentData.model_validate_any(metadata)
        video_folder = get_video_folder_from_segment_metadata(metadata)
        chain = resolve_segment_chain(
            video_folder=video_folder,
            tail_segment_id=metadata.segment_id,
        )

        return io.NodeOutput(
            chain,
            json.dumps(chain.segment_paths, indent=2),
            chain.video_id,
            chain.head_segment_id,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        metadata: LTXLat2VidSegmentData | dict | str,
    ) -> str:
        metadata = LTXLat2VidSegmentData.model_validate_any(metadata)
        video_folder = get_video_folder_from_segment_metadata(metadata)

        fingerprint = sha256()
        fingerprint.update(metadata.model_dump_json(exclude={"prompt", "extra_pnginfo"}).encode("utf-8"))

        output_root = get_output_dir_path()
        for _, _, segment_path, _ in list_segment_files(video_folder, return_tuple=True):
            abs_path = output_root / segment_path
            stats = abs_path.stat()
            fingerprint.update(str(segment_path).encode("utf-8"))
            fingerprint.update(str(stats.st_size).encode("utf-8"))
            fingerprint.update(str(stats.st_mtime_ns).encode("utf-8"))

        return fingerprint.hexdigest()


class LTXLat2VidAssembleSegmentChainNood(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXLat2VidAssembleSegmentChainNood",
            display_name="LTX-L2V Assemble Segment Chain",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentChainIO.Input(
                    "chain",
                    tooltip="Resolved segment chain from the chain resolver node.",
                ),
                io.Boolean.Input(
                    "strict",
                    default=True,
                    label_on="Strict",
                    label_off="Loose",
                    tooltip="Validate parent linkage, ordering, fps, and dimensions while assembling.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="frames"),
                io.Int.Output(display_name="start_frame"),
                io.Int.Output(display_name="n_frames"),
                io.Float.Output(display_name="fps"),
                io.String.Output(
                    display_name="filename_prefix",
                    tooltip="Filename prefix for saving the assembled clip with frame/video combiner nodes.",
                ),
                io.String.Output(
                    display_name="segment_paths_json",
                    tooltip="JSON list of source segment paths that were assembled.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        chain: LTXLat2VidSegmentChainData | dict | str,
        strict: bool,
    ) -> io.NodeOutput:
        chain = LTXLat2VidSegmentChainData.model_validate_any(chain)
        if not chain.segment_paths:
            raise ValueError("Segment chain is empty")

        expected_video_id = str(chain.video_id)
        fps: float | None = None
        min_start: int | None = None
        max_end = 0
        prev_metadata: LTXLat2VidSegmentData | None = None
        segment_metadata: list[tuple[str, LTXLat2VidSegmentData, int, int]] = []
        n_segments = len(chain.segment_paths)
        total_steps = max(1, n_segments * 2)

        # pass 1: metadata validation and output frame bounds
        for idx, segment_path in enumerate(chain.segment_paths):
            segment_data, _, _ = load_segment_metadata(segment_path)

            if str(segment_data.video_id) != expected_video_id:
                raise ValueError(f"Segment {segment_path} has video_id={segment_data.video_id}, expected {expected_video_id}")

            if idx < len(chain.segment_ids):
                expected_segment_id = str(chain.segment_ids[idx])
                if str(segment_data.segment_id) != expected_segment_id:
                    raise ValueError(f"Segment {segment_path} has segment_id={segment_data.segment_id}, expected {expected_segment_id}")

            if fps is None:
                fps = float(segment_data.fps)
            elif strict and abs(float(segment_data.fps) - fps) > 1e-6:
                raise ValueError(f"FPS mismatch in chain at {segment_path}: got {segment_data.fps}, expected {fps}")

            if strict and prev_metadata is not None:
                if segment_data.parent_id != prev_metadata.segment_id:
                    raise ValueError(
                        f"Chain linkage mismatch at {segment_path}: parent_id={segment_data.parent_id} "
                        + f"but previous segment_id={prev_metadata.segment_id}"
                    )
                if segment_data.start_frame < prev_metadata.start_frame:
                    raise ValueError(
                        f"Non-monotonic start_frame in chain at {segment_path}: "
                        + f"{segment_data.start_frame} < {prev_metadata.start_frame}"
                    )

            trim_lead_frames = 8 if int(segment_data.segment_idx) > 0 and int(segment_data.mask_params.hard_mask_k) >= 1 else 0
            total_segment_frames = int(segment_data.n_frames)
            if trim_lead_frames >= total_segment_frames:
                raise ValueError(f"Segment {segment_path} has n_frames={total_segment_frames}, cannot trim {trim_lead_frames} lead frames")

            # Trim first overlapped latent block (8 frames) for continuation segments when hard mask is active.
            # Keep segment end fixed by shifting the write start forward.
            start_frame = int(segment_data.start_frame) + trim_lead_frames
            n_frames = total_segment_frames - trim_lead_frames
            end_frame = start_frame + n_frames

            min_start = start_frame if min_start is None else min(min_start, start_frame)
            max_end = max(max_end, end_frame)
            segment_metadata.append((segment_path, segment_data, start_frame, n_frames))
            prev_metadata = segment_data
            await _API.execution.set_progress(value=idx + 1, max_value=total_steps)

        if min_start is None or fps is None:
            raise ValueError("No segment metadata was loaded from chain")

        out_start = min_start
        out_n_frames = max_end - out_start
        expected_shape: tuple[int, int, int] | None = None
        full_frames: torch.Tensor | None = None

        # pass 2: frame loading and assembly
        for idx, (segment_path, metadata, start_frame, n_frames) in enumerate(segment_metadata):
            state_dict, segment_data, _, _ = load_segment_file(segment_path)

            if "frames" in state_dict:
                frames = state_dict["frames"].cpu().contiguous()
            elif "compressed_frames" in state_dict:
                frames = decompress_image_tensor_webp(
                    state_dict["compressed_frames"].cpu().contiguous(),
                    (segment_data.width_px, segment_data.height_px),
                    as_float=True,
                )
            else:
                raise ValueError(f"Segment file {segment_path} is missing frame tensors ('frames' or 'compressed_frames')")

            total_segment_frames = int(metadata.n_frames)
            if frames.shape[0] < total_segment_frames:
                raise ValueError(
                    f"Segment {segment_path} has fewer decoded frames ({frames.shape[0]}) than metadata.n_frames ({total_segment_frames})"
                )
            if frames.shape[0] > total_segment_frames:
                frames = frames[:total_segment_frames]

            trim_lead_frames = total_segment_frames - n_frames
            if trim_lead_frames > 0:
                frames = frames[trim_lead_frames:]

            if expected_shape is None:
                expected_shape = tuple(frames.shape[1:])
                full_frames = torch.zeros((out_n_frames, *expected_shape), dtype=frames.dtype, device=frames.device)
            elif tuple(frames.shape[1:]) != expected_shape:
                raise ValueError(f"Frame shape mismatch while assembling chain: got {tuple(frames.shape[1:])}, expected {expected_shape}")

            if strict and str(segment_data.segment_id) != str(metadata.segment_id):
                raise ValueError(
                    f"Segment metadata changed between resolve and load for {segment_path}: "
                    + f"loaded {segment_data.segment_id}, expected {metadata.segment_id}"
                )

            end_frame = start_frame + n_frames
            dst_start = start_frame - out_start
            dst_end = end_frame - out_start
            full_frames[dst_start:dst_end] = frames[: dst_end - dst_start]

            del state_dict
            await _API.execution.set_progress(value=n_segments + idx + 1, max_value=total_steps)

        if full_frames is None:
            raise ValueError("No frames were loaded from chain")

        output_root = get_output_dir_path()
        video_folder = chain.video_folder.rstrip("\\/")
        assembled_tail = str(chain.tail_segment_id)[-6:]
        filename_tail = f"assembled/{chain.video_name}.v{str(chain.video_id)[:10]}.full.{assembled_tail}"
        filename_prefix = f"{video_folder}/{filename_tail}" if video_folder else filename_tail
        output_root.joinpath(filename_prefix).parent.mkdir(parents=True, exist_ok=True)

        return io.NodeOutput(
            full_frames.contiguous(),
            out_start,
            out_n_frames,
            fps,
            filename_prefix,
            json.dumps(chain.segment_paths, indent=2),
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        chain: LTXLat2VidSegmentChainData | dict | str,
        strict: bool,
    ) -> str:
        chain = LTXLat2VidSegmentChainData.model_validate_any(chain)
        fingerprint = sha256()
        fingerprint.update(str(chain.video_id).encode("utf-8"))
        fingerprint.update(str(strict).encode("utf-8"))

        output_root = get_output_dir_path()
        for segment_path in chain.segment_paths:
            abs_path = Path(segment_path)
            if not abs_path.is_absolute():
                abs_path = output_root / abs_path
            if not abs_path.exists():
                fingerprint.update(f"missing:{segment_path}".encode())
                continue
            stats = abs_path.stat()
            fingerprint.update(str(segment_path).encode("utf-8"))
            fingerprint.update(str(stats.st_size).encode("utf-8"))
            fingerprint.update(str(stats.st_mtime_ns).encode("utf-8"))

        return fingerprint.hexdigest()


class LTXLat2VidAssembleLatentChainNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidAssembleLatentChainNood",
            display_name="LTX-L2V Assemble Latent Chain",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentChainIO.Input(
                    "chain",
                    tooltip="Resolved segment chain from the chain resolver node.",
                ),
                io.Boolean.Input(
                    "strict",
                    default=True,
                    label_on="Strict",
                    label_off="Loose",
                    tooltip="Validate parent linkage, ordering, fps, dimensions, and latent alignment while assembling.",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="start_frame"),
                io.Int.Output(display_name="n_frames"),
                io.Int.Output(display_name="n_latents"),
                io.Float.Output(display_name="fps"),
                io.String.Output(
                    display_name="filename_prefix",
                    tooltip="Filename prefix for saving the assembled latent clip.",
                ),
                io.String.Output(
                    display_name="segment_paths_json",
                    tooltip="JSON list of source segment paths that were assembled.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        chain: LTXLat2VidSegmentChainData | dict | str,
        strict: bool,
    ) -> io.NodeOutput:
        chain = LTXLat2VidSegmentChainData.model_validate_any(chain)
        if not chain.segment_paths:
            raise ValueError("Segment chain is empty")

        expected_video_id = str(chain.video_id)
        fps: float | None = None
        min_start: int | None = None
        max_end = 0
        prev_metadata: LTXLat2VidSegmentData | None = None
        segment_metadata: list[tuple[str, LTXLat2VidSegmentData, int, int, int]] = []
        n_segments = len(chain.segment_paths)
        total_steps = max(1, n_segments * 2)

        # pass 1: metadata validation and output frame bounds
        for idx, segment_path in enumerate(chain.segment_paths):
            segment_data, _, _ = load_segment_metadata(segment_path)

            if str(segment_data.video_id) != expected_video_id:
                raise ValueError(f"Segment {segment_path} has video_id={segment_data.video_id}, expected {expected_video_id}")

            if idx < len(chain.segment_ids):
                expected_segment_id = str(chain.segment_ids[idx])
                if str(segment_data.segment_id) != expected_segment_id:
                    raise ValueError(f"Segment {segment_path} has segment_id={segment_data.segment_id}, expected {expected_segment_id}")

            if fps is None:
                fps = float(segment_data.fps)
            elif strict and abs(float(segment_data.fps) - fps) > 1e-6:
                raise ValueError(f"FPS mismatch in chain at {segment_path}: got {segment_data.fps}, expected {fps}")

            if strict and prev_metadata is not None:
                if segment_data.parent_id != prev_metadata.segment_id:
                    raise ValueError(
                        f"Chain linkage mismatch at {segment_path}: parent_id={segment_data.parent_id} "
                        + f"but previous segment_id={prev_metadata.segment_id}"
                    )
                if segment_data.start_frame < prev_metadata.start_frame:
                    raise ValueError(
                        f"Non-monotonic start_frame in chain at {segment_path}: "
                        + f"{segment_data.start_frame} < {prev_metadata.start_frame}"
                    )

            trim_lead_frames = 8 if int(segment_data.segment_idx) > 0 and int(segment_data.mask_params.hard_mask_k) >= 1 else 0
            total_segment_frames = int(segment_data.n_frames)
            if trim_lead_frames >= total_segment_frames:
                raise ValueError(f"Segment {segment_path} has n_frames={total_segment_frames}, cannot trim {trim_lead_frames} lead frames")

            start_frame = int(segment_data.start_frame) + trim_lead_frames
            n_frames = total_segment_frames - trim_lead_frames
            end_frame = start_frame + n_frames

            min_start = start_frame if min_start is None else min(min_start, start_frame)
            max_end = max(max_end, end_frame)
            segment_metadata.append((segment_path, segment_data, start_frame, n_frames, trim_lead_frames))
            prev_metadata = segment_data
            await _API.execution.set_progress(value=idx + 1, max_value=total_steps)

        if min_start is None or fps is None:
            raise ValueError("No segment metadata was loaded from chain")

        out_start = min_start
        out_n_frames = max_end - out_start
        if out_n_frames < 1:
            raise ValueError(f"Invalid assembled frame count: {out_n_frames}")
        if out_n_frames > 1 and ((out_n_frames - 1) % 8 != 0):
            raise ValueError(f"Assembled frame count {out_n_frames} is not compatible with packed LTX latent format (expected 1 + 8*k)")
        out_n_latents = 1 + ((out_n_frames - 1) // 8) if out_n_frames > 1 else 1

        full_samples: torch.Tensor | None = None
        expected_sample_shape: tuple[int, int, int, int] | None = None
        bootstrap_written = False

        # pass 2: latent loading and assembly
        for idx, (segment_path, metadata, _, _, trim_lead_frames) in enumerate(segment_metadata):
            state_dict, segment_data, _, _ = load_segment_file(segment_path)
            if "latent" not in state_dict:
                raise ValueError(f"Segment file {segment_path} is missing required 'latent' tensor")

            samples = state_dict["latent"].cpu().contiguous()
            if samples.ndim != 5:
                raise ValueError(f"Expected latent tensor with shape [B,C,T,H,W], got {samples.shape!r} in {segment_path}")

            if strict and str(segment_data.segment_id) != str(metadata.segment_id):
                raise ValueError(
                    f"Segment metadata changed between resolve and load for {segment_path}: "
                    + f"loaded {segment_data.segment_id}, expected {metadata.segment_id}"
                )

            sample_shape = (int(samples.shape[0]), int(samples.shape[1]), int(samples.shape[3]), int(samples.shape[4]))
            if expected_sample_shape is None:
                expected_sample_shape = sample_shape
                full_samples = torch.zeros(
                    (sample_shape[0], sample_shape[1], out_n_latents, sample_shape[2], sample_shape[3]),
                    dtype=samples.dtype,
                    device=samples.device,
                )
            elif sample_shape != expected_sample_shape:
                raise ValueError(f"Latent shape mismatch while assembling chain: got {sample_shape}, expected {expected_sample_shape}")

            if metadata.keep_bootstrap and trim_lead_frames == 0:
                bootstrap_frame_start = int(metadata.start_frame)
                if strict and bootstrap_frame_start != out_start:
                    raise ValueError(
                        f"Bootstrap latent in {segment_path} starts at frame {bootstrap_frame_start}, expected assembled start {out_start}"
                    )
                if bootstrap_frame_start == out_start:
                    full_samples[:, :, 0:1, :, :] = samples[:, :, 0:1, :, :]
                    bootstrap_written = True

            first_block_idx = 1 + (trim_lead_frames // 8)
            n_latents = int(samples.shape[2])
            if first_block_idx >= n_latents:
                raise ValueError(f"Segment {segment_path} has only {n_latents} latents, cannot start from latent index {first_block_idx}")

            for src_latent_idx in range(first_block_idx, n_latents):
                if metadata.keep_bootstrap:
                    block_start_frame = int(metadata.start_frame) + 1 + (8 * (src_latent_idx - 1))
                else:
                    block_start_frame = int(metadata.start_frame) + (8 * (src_latent_idx - 1))

                if block_start_frame < out_start + 1:
                    continue

                delta = block_start_frame - (out_start + 1)
                if strict and (delta % 8) != 0:
                    raise ValueError(
                        f"Latent block alignment mismatch in {segment_path}: block_start_frame={block_start_frame}, out_start={out_start}"
                    )
                if (delta % 8) != 0:
                    continue

                dst_latent_idx = 1 + (delta // 8)
                if dst_latent_idx < 1:
                    continue
                if dst_latent_idx >= out_n_latents:
                    continue

                full_samples[:, :, dst_latent_idx : dst_latent_idx + 1, :, :] = samples[:, :, src_latent_idx : src_latent_idx + 1, :, :]

            del state_dict
            await _API.execution.set_progress(value=n_segments + idx + 1, max_value=total_steps)

        if full_samples is None:
            raise ValueError("No latents were loaded from chain")
        if not bootstrap_written:
            raise ValueError("Assembled latent chain is missing bootstrap latent at output start frame")

        output_root = get_output_dir_path()
        video_folder = chain.video_folder.rstrip("\\/")
        assembled_tail = str(chain.tail_segment_id)[-6:]
        filename_tail = f"assembled-latent/{chain.video_name}.v{str(chain.video_id)[:10]}.full.{assembled_tail}"
        filename_prefix = f"{video_folder}/{filename_tail}" if video_folder else filename_tail
        output_root.joinpath(filename_prefix).parent.mkdir(parents=True, exist_ok=True)

        return io.NodeOutput(
            {"samples": full_samples.contiguous()},
            out_start,
            out_n_frames,
            out_n_latents,
            fps,
            filename_prefix,
            json.dumps(chain.segment_paths, indent=2),
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        chain: LTXLat2VidSegmentChainData | dict | str,
        strict: bool,
    ) -> str:
        chain = LTXLat2VidSegmentChainData.model_validate_any(chain)
        fingerprint = sha256()
        fingerprint.update(str(chain.video_id).encode())
        fingerprint.update(str(strict).encode())

        output_root = get_output_dir_path()
        for segment_path in chain.segment_paths:
            abs_path = Path(segment_path)
            if not abs_path.is_absolute():
                abs_path = output_root / abs_path
            if not abs_path.exists():
                fingerprint.update(f"missing:{segment_path}".encode())
                continue
            stats = abs_path.stat()
            fingerprint.update(str(segment_path).encode())
            fingerprint.update(str(stats.st_size).encode())
            fingerprint.update(str(stats.st_mtime_ns).encode())

        return fingerprint.hexdigest()


class LTXLat2VidInplaceNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidInplaceNood",
            display_name="LTX-L2V Inplace",
            category="noodles/ltx",
            inputs=[
                io.Vae.Input("vae", optional=True),
                io.Int.Input(
                    "noise_seed",
                    display_name="noise seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="Seed for the deterministic seed generator.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="Empty latent tensor for this segment.",
                ),
                io.Latent.Input(
                    "prev_latent",
                    tooltip="Latent tensor from previous segment.",
                ),
                LTXLat2VidSegmentIO.Input(
                    "prev_metadata",
                    display_name="metadata",
                    tooltip="Metadata from the previous segment.",
                ),
                io.Int.Input(
                    "overlap_k",
                    default=6,
                    min=2,
                    max=512,
                    tooltip="Number of overlapped latents to carry into the next segment.",
                ),
                BootstrapModeIO.Input("bootstrap_mode", default=BootstrapMode.DummyLatent),
                io.Float.Input(
                    "bootstrap_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    display_name="bootstrap strength",
                    tooltip="Strength of the bootstrap frame when doing inplace I2V. Should be 1.0 most of the time.",
                    advanced=True,
                ),
                MaskParamsIO.Input("mask_params"),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="noise_seed"),
                LTXLat2VidSegmentIO.Output(display_name="metadata"),
                BootstrapModeIO.Output(display_name="bootstrap_mode"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae: VAE | None,
        noise_seed: int,
        latent: LatentInput,
        prev_latent: LatentInput,
        prev_metadata: LTXLat2VidSegmentData | str,
        overlap_k: int,
        bootstrap_mode: BootstrapMode,
        bootstrap_strength: float,
        mask_params: MaskParams | dict | None,
    ) -> io.NodeOutput:
        prev_segment_data = LTXLat2VidSegmentData.model_validate_any(prev_metadata)

        prev_samples: torch.Tensor = prev_latent["samples"]
        next_samples: torch.Tensor = latent["samples"]

        if prev_samples.ndim != 5 or next_samples.ndim != 5:
            raise ValueError(
                f"Expected both prev and next latent samples to have shape [B,C,T,H,W], got {prev_samples.shape!r} and {next_samples.shape!r}"
            )

        if overlap_k < 2:
            raise ValueError("overlap_k must be at least 2")
        if next_samples.shape[2] < overlap_k:
            raise ValueError(f"next_latent has {next_samples.shape[2]} latents but overlap_k is {overlap_k}")

        if prev_samples.shape[2] < overlap_k:
            raise ValueError(f"prev_latent has {prev_samples.shape[2]} usable latents but overlap_k is {overlap_k}")

        overlap_chunk = prev_samples[:, :, -overlap_k:, :, :].clone()
        source_latent = overlap_chunk[:, :, 0:1, :, :]
        carried_latents = overlap_chunk[:, :, 1:, :, :]

        if bootstrap_mode == BootstrapMode.SegmentZero:
            raise ValueError("BootstrapMode.SegmentZero is only valid for the first segment and cannot be used for continuation")

        new_seed = _deterministic_seed(prev_segment_data, seed=noise_seed, overlap_k=overlap_k)

        if bootstrap_mode == BootstrapMode.RawLatent:
            bootstrap_latent = source_latent
        elif bootstrap_mode == BootstrapMode.DummyLatent:
            bootstrap_latent = next_samples[:, :, :1, :, :].clone()
        elif bootstrap_mode == BootstrapMode.VAERoundtrip:
            if vae is None:
                raise ValueError("VAE is required for BootstrapMode.VAERoundtrip")
            decoded = vae.decode(source_latent)
            if decoded.ndim == 4:
                # Expected image decode shape [N, H, W, C]
                last_frame = decoded[-1:, :, :, :3]
            else:
                raise ValueError(f"Unexpected decoded latent shape from VAE: {decoded.shape!r}")

            encoded = vae.encode(last_frame)
            if encoded.ndim == 4:
                encoded = encoded.unsqueeze(2)
            if encoded.ndim != 5:
                raise ValueError(f"Unexpected encoded latent shape from VAE: {encoded.shape!r}")

            bootstrap_latent = encoded[:, :, :1, :, :]
        else:
            raise ValueError(f"Unknown bootstrap mode: {bootstrap_mode}")

        next_samples[:, :, 0:1, :, :] = bootstrap_latent
        next_samples[:, :, 1:overlap_k, :, :] = carried_latents[:, :, : overlap_k - 1, :, :]

        strengths = _compute_overlap_strengths(
            total_k=overlap_k,
            bootstrap_mode=bootstrap_mode,
            bootstrap_strength=bootstrap_strength,
            mask_params=mask_params,
        )

        noise_mask = torch.ones(
            (next_samples.shape[0], 1, next_samples.shape[2], 1, 1),
            dtype=next_samples.dtype,
            device=next_samples.device,
        )
        strengths_tensor = torch.tensor(strengths, dtype=next_samples.dtype, device=next_samples.device)

        noise_mask[:, :, :overlap_k, :, :] = 1.0 - strengths_tensor.view(1, 1, overlap_k, 1, 1)

        next_segment_idx = int(prev_segment_data.segment_idx) + 1
        next_start_frame = int(prev_segment_data.start_frame) + int(prev_segment_data.n_frames) - (8 * (overlap_k - 1))

        next_metadata = prev_segment_data.model_copy(
            update={
                "parent_id": prev_segment_data.segment_id,
                "segment_id": ULID(),
                "segment_idx": next_segment_idx,
                "start_frame": next_start_frame,
                "overlap_k": overlap_k,
                "bootstrap_mode": bootstrap_mode,
                "mask_params": mask_params,
            }
        )

        out_latent: dict[str, torch.Tensor] = {
            "samples": next_samples,
            "noise_mask": noise_mask,
        }
        return io.NodeOutput(
            out_latent,
            new_seed,
            next_metadata,
            bootstrap_mode,
        )


class LTXLat2VidPrepNextDataNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidPrepNextDataNood",
            display_name="LTX-L2V Prepare Segment Data",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentIO.Input(
                    id="metadata",
                    display_name="metadata",
                    tooltip="Previous segment metadata to unpack",
                ),
                MaskParamsIO.Input(
                    "mask_params",
                    optional=True,
                    default=None,
                    tooltip="Optional mask parameters to override those in the metadata for the next segment",
                ),
            ],
            outputs=[
                MaskParamsIO.Output(display_name="mask_params"),
                io.Int.Output(display_name="overlap_k"),
                BootstrapModeIO.Output(display_name="bootstrap_mode"),
                io.Int.Output(display_name="width_px"),
                io.Int.Output(display_name="height_px"),
                io.Int.Output(display_name="n_frames_batch"),
                io.Int.Output(display_name="next_start_frame"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(
        cls,
        metadata: LTXLat2VidSegmentData,
        mask_params: MaskParams | dict | None,
    ) -> io.NodeOutput:
        # override mask_params in metadata if new ones are provided, otherwise keep existing
        if mask_params is not None:
            metadata.mask_params = mask_params

        # work out the last frame index of the current segment so we can calculate the next segment's start frame
        last_end_frame = metadata.start_frame + metadata.n_frames
        # calculate how many frames of overlap, accounting for the first overlapped latent only actually counting for one frame of overlap
        n_overlap_frames = (metadata.overlap_k * 8) - 7
        # the next segment should start right after the non-overlapped frames of the previous segment
        next_start_frame = last_end_frame - n_overlap_frames

        return io.NodeOutput(
            metadata.mask_params,
            metadata.overlap_k,
            metadata.bootstrap_mode,
            metadata.width_px,
            metadata.height_px,
            metadata.n_frames_batch,
            next_start_frame,
            metadata.fps,
        )


class LTXLat2VidPrepSaveDataNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLat2VidPrepSaveDataNood",
            display_name="LTX-L2V Prepare Save Data",
            category="noodles/ltx",
            inputs=[
                LTXLat2VidSegmentIO.Input(
                    id="metadata",
                    display_name="metadata",
                    tooltip="Previous segment metadata to unpack",
                ),
            ],
            outputs=[
                ComfyULID.Output(display_name="video_id"),
                ComfyULID.Output(display_name="parent_id"),
                io.String.Output(display_name="folder_prefix"),
                io.String.Output(display_name="video_name"),
                io.Int.Output(display_name="next_segment_idx"),
                io.Int.Output(display_name="next_start_frame"),
                io.Int.Output(display_name="n_frames_batch"),
                io.Int.Output(display_name="overlap_k"),
                io.Float.Output(display_name="fps"),
            ],
        )

    @classmethod
    def execute(cls, metadata: LTXLat2VidSegmentData) -> io.NodeOutput:
        # work out the last frame index of the current segment so we can calculate the next segment's start frame
        last_end_frame = metadata.start_frame + metadata.n_frames
        # calculate how many frames of overlap, accounting for the first overlapped latent only actually counting for one frame of overlap
        n_overlap_frames = (metadata.overlap_k * 8) - 7
        # the next segment should start right after the non-overlapped frames of the previous segment
        next_start_frame = last_end_frame - n_overlap_frames

        # increment segment idx for the next segment
        next_segment_idx = metadata.segment_idx + 1

        # build name of segment save folder
        video_dirname = f"{metadata.video_name}_{metadata.video_id}"
        # strip out the video_dirname from the folder_prefix (if present), trim slashes for consistency
        folder_prefix = metadata.subfolder.split(f"/{video_dirname}", 1)[0].rstrip("\\/")

        return io.NodeOutput(
            metadata.video_id,
            metadata.segment_id,
            folder_prefix,
            metadata.video_name,
            next_segment_idx,
            next_start_frame,
            metadata.n_frames_batch,
            metadata.overlap_k,
            metadata.fps,
        )


class LTXMaskParamsNood(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXMaskParamsNood",
            display_name="LTX L2V Mask Params",
            category="noodles/ltx",
            inputs=[
                MaskStrategyIO.Input(
                    "strategy",
                    default=MaskStrategy.NoStrategy,
                ),
                io.Int.Input(
                    "hard_mask_k",
                    default=1,
                    min=1,
                    max=32,
                    tooltip="Number of overlapped latents to hard-mask at strength 1.0 before decay.",
                ),
                io.Float.Input(
                    "w_max",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    display_name="w_max",
                    tooltip="Maximum strength for the final latent in the overlap window.",
                ),
                io.Float.Input(
                    "w_min",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    display_name="w_min",
                    tooltip="Minimum strength for the final latent in the overlap window.",
                ),
                io.Float.Input(
                    "decay_sigma",
                    default=0.4,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="Sigma for half-Gaussian decay curve.",
                ),
            ],
            outputs=[
                MaskParamsIO.Output(display_name="mask_params"),
            ],
        )

    @classmethod
    def execute(
        cls,
        strategy: MaskStrategy,
        hard_mask_k: int,
        w_max: float,
        w_min: float,
        decay_sigma: float,
    ) -> dict:
        mask_params = MaskParams(
            strategy=strategy,
            hard_mask_k=hard_mask_k,
            w_max=w_max,
            w_min=w_min,
            decay_sigma=decay_sigma,
        )

        return io.NodeOutput(mask_params)
