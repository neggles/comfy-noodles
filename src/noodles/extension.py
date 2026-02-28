from comfy_api.latest import ComfyExtension, io

from .ltx import (
    LTXImg2VidInplaceNood,
    LTXLat2VidPrepareSegmentNood,
    LTXLat2VidSegmentLoadNood,
    LTXLat2VidSegmentSaveNood,
    LTXULIDFromStrNood,
    LTXULIDPreviewNood,
)
from .misc import StringIntAddNood


class NoodlesExtension(ComfyExtension):
    async def on_load(self) -> None:
        pass

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXImg2VidInplaceNood,
            LTXLat2VidSegmentLoadNood,
            LTXLat2VidSegmentSaveNood,
            LTXLat2VidPrepareSegmentNood,
            LTXULIDPreviewNood,
            LTXULIDFromStrNood,
            StringIntAddNood,
        ]


__all__ = [
    "NoodlesExtension",
]
