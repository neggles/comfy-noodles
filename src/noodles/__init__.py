from comfy_api.latest import ComfyExtension, io

# need to import to register routes, even if not used directly
from . import routes  # noqa: F401
from .ltx.common import (
    ULIDFromStrNood,
    ULIDPreviewNood,
)
from .ltx.i2v import LTXImg2VidInplaceNood
from .ltx.l2v import (
    LTXLat2VidGetNextSegmentDataNood,
    LTXLat2VidGetNextSegmentSaveDataNood,
    LTXLat2VidInplaceNood,
    LTXLat2VidSegmentLoadNood,
    LTXLat2VidSegmentSaveNood,
    LTXMaskParamsNood,
)
from .misc import (
    AudioPreviewMelSpectrogramNood,
    StringIntAddNood,
)

_NODE_LIST = [
    AudioPreviewMelSpectrogramNood,
    LTXImg2VidInplaceNood,
    LTXLat2VidGetNextSegmentDataNood,
    LTXLat2VidGetNextSegmentSaveDataNood,
    LTXLat2VidInplaceNood,
    LTXLat2VidSegmentLoadNood,
    LTXLat2VidSegmentSaveNood,
    LTXMaskParamsNood,
    StringIntAddNood,
    ULIDFromStrNood,
    ULIDPreviewNood,
]


class NoodlesExtension(ComfyExtension):
    async def on_load(self) -> None:
        pass

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _NODE_LIST.copy()

    def get_node_list_sync(self) -> list[type[io.ComfyNode]]:
        return _NODE_LIST.copy()


async def comfy_entrypoint() -> NoodlesExtension:
    return NoodlesExtension()
