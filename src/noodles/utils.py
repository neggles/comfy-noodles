import re
from os import PathLike
from pathlib import Path

from folder_paths import get_output_directory
from ulid import ULID

re_idx_pattern = re.compile(r"_[svi](\d{3,5})_", re.I)


def get_output_dir_path() -> Path:
    return Path(get_output_directory())


def get_next_file_idx(filepath: PathLike) -> int:
    filepath = Path(filepath)
    folder = filepath.parent
    if not folder.exists():
        return 1
    max_idx = 0
    for f in folder.iterdir():
        if f.is_file() and f.stem.startswith(filepath.stem):
            if match := re_idx_pattern.search(f.stem):
                idx = int(match.group(1))
                max_idx = max(max_idx, idx)
    return max_idx + 1


def prune_dict(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def parse_ulid(
    value: ULID | str | int | bytes | None, field_name: str | None = None, *, optional: bool = False
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
                raise ValueError(
                    f"Could not parse value '{value!r}' of type '{type(value)}' for {field_name}"
                ) from e
