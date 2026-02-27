from pathlib import Path

from folder_paths import get_output_directory
from ulid import ULID


def prune_dict(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def get_output_dir_path() -> Path:
    return Path(get_output_directory())


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
