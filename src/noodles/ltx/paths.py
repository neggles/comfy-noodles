import re
from os import PathLike
from pathlib import Path

from ulid import ULID

from ..utils import get_output_dir_path

# segment format: <whatever>.s{segment_idx}_i{iteration}.{optional_id}.safetensors
SEGMENT_ITER_RE = re.compile(r"\.s(?P<segment>\d+)(?:_i(?P<iteration>\d+))?(?:\.(?P<id>\w{4,6}))?(?:\.safetensors)?$", re.I)
# video folder format: {video_name}_v{video_ulid} (sometimes the `v` is missing)
VIDEO_FOLDER_RE = re.compile(r"^(?P<video_name>.+)_v?(?P<video_ulid>\w{26})$", re.I)


def parse_video_folder_name(folder_name: str) -> tuple[str, str] | None:
    """Parse video name and ULID from a folder name. Returns (video_name, video_ulid) or None if not a match."""
    if match := VIDEO_FOLDER_RE.match(folder_name):
        return match.group("video_name"), match.group("video_ulid")
    return None


def list_video_folders(
    prefix: str = "",
    recursive: bool = False,
    return_tuple: bool = False,
) -> list[Path] | list[tuple[str, str, Path]]:
    """List subfolders in the output directory that match the video folder format."""
    output_root = get_output_dir_path()
    search_root = output_root / prefix if prefix else output_root

    if not search_root.exists() or not search_root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {search_root}")

    candidates = search_root.rglob("*") if recursive else search_root.iterdir()
    video_folders = set()
    for path in candidates:
        if path.is_dir() and (parsed := parse_video_folder_name(path.name)):
            try:
                video_name, video_ulid = parsed
                subpath = path.relative_to(output_root)
                video_folders.add((video_name, video_ulid, subpath))

            except ValueError:
                continue  # skip folders that can't be relativized for some reason

    video_folders = sorted(video_folders, key=lambda x: x[0].lower())

    return video_folders if return_tuple else [item[2] for item in video_folders]


def get_video_folder_by_id(id: str | ULID, prefix: str = "") -> Path:
    """Find the folder path for a given video by ULID"""
    output_root = get_output_dir_path()
    search_root = output_root / prefix if prefix else output_root
    if not search_root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {search_root}")

    video_ulid = str(id)
    for path in search_root.rglob("*"):
        if path.is_dir() and (parsed := parse_video_folder_name(path.name)):
            _, folder_ulid = parsed
            if folder_ulid == video_ulid:
                return path.relative_to(output_root)

    raise FileNotFoundError(f"No folder found for video ULID: {video_ulid}")


def parse_segment_name(path: Path) -> tuple[int, int | None, str | None] | None:
    """Parse segment and iteration numbers from a filename, if present.
    Returns (segment, iteration, optional_id | None) or (None, None, None) if not found.
    """
    # ensure path is a Path object
    path = Path(path)
    # attempt to parse segment, iteration, and optional id from the filename
    if match := SEGMENT_ITER_RE.search(path.name):
        s_num = match.group("segment")
        i_num = match.group("iteration")
        return (
            int(s_num) if s_num is not None else -1,
            int(i_num) if i_num is not None else -1,
            match.group("id"),
        )
    return None


def list_segment_files(video_folder: str, return_tuple: bool = False) -> list[Path] | list[tuple[int, int, Path, str | None]]:
    """List segment files for a given video folder, sorted by segment index and iteration."""
    output_root = get_output_dir_path()
    folder_path = output_root / video_folder
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Subfolder does not exist: {folder_path}")

    segments = []
    for path in folder_path.glob("*.safetensors"):
        path = path.relative_to(output_root)
        seg_idx, seg_iter, seg_id = parse_segment_name(path)
        if seg_idx is not None and seg_iter is not None:
            segments.append((seg_idx, seg_iter, path, seg_id))

    # sort by segment index, then iteration
    segments.sort(key=lambda item: (item[0], item[1]))
    return segments if return_tuple else [item[2] for item in segments]


def find_segment_file(video_folder: str, segment_idx: int, iteration: int, segment_id: str | None = None) -> Path:
    output_root = get_output_dir_path()
    folder_path = output_root / video_folder
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Subfolder does not exist: {folder_path}")

    candidates: list[tuple[int, Path]] = []
    for seg_idx, seg_iter, path, seg_id in list_segment_files(video_folder, return_tuple=True):
        if seg_idx == segment_idx:
            if seg_id is None or segment_id is None or seg_id.endswith(segment_id):
                candidates.append((seg_iter, path))

    if not candidates:
        raise FileNotFoundError(f"No segment files found for segment_idx={segment_idx} in {folder_path}")

    candidates.sort(key=lambda item: item[0])
    # negative iteration and no ID means we want the latest iteration, so return the last one
    if iteration <= 0 and not segment_id:
        return candidates[-1][1]

    for seg_iter, path in candidates:
        if seg_iter == iteration:
            return path

    available = ", ".join(str(seg_iter) for seg_iter, _ in candidates)
    raise FileNotFoundError(
        f"No segment file found for segment_idx={segment_idx}, iteration={iteration}. Available iterations: {available}"
    )


def get_segment_idx_iter(path: Path) -> tuple[int, int] | None:
    """Convenience function to extract just the segment index and iteration from a segment filename."""
    segment, iteration, _ = parse_segment_name(path)
    if segment is not None and iteration is not None:
        return segment, iteration
    return None


def get_next_segment_iteration(filepath: PathLike) -> int:
    """Get the next iteration number for a segment file's name."""
    filepath = Path(filepath)
    folder = filepath.parent
    if not folder.is_dir():
        return 0

    if parsed := parse_segment_name(filepath):
        segment_idx, _, _ = parsed
    else:
        raise ValueError(f"Filepath does not match expected segment format: {filepath}")

    max_iter = -1
    for path in folder.glob(f"*.s{segment_idx:03d}_i*.safetensors"):
        if parsed := parse_segment_name(path):
            _, iteration, _ = parsed
            if iteration is not None and iteration > max_iter:
                max_iter = iteration

    return max_iter + 1
