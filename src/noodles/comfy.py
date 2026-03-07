import warnings

from comfy_api.latest import ComfyAPI, ComfyExtension, io

# need to import to register routes, even if not used directly
from .ltx.common import (
    ULIDFromStrNood,
    ULIDPreviewNood,
)
from .ltx.i2v import LTXImg2VidInplaceNood
from .ltx.l2v import (
    LTXLat2VidAssembleLatentChainNood,
    LTXLat2VidAssembleSegmentChainNood,
    LTXLat2VidInplaceNood,
    LTXLat2VidPrepNextDataNood,
    LTXLat2VidPrepSaveDataNood,
    LTXLat2VidResolveSegmentChainNood,
    LTXLat2VidSegmentLoadNood,
    LTXLat2VidSegmentSaveNood,
    LTXMaskParamsNood,
)
from .misc import (
    AudioPreviewMelSpectrogramNood,
    StringIntAddNood,
    VideoGenParamsNood,
)

_NODE_LIST = [
    AudioPreviewMelSpectrogramNood,
    LTXLat2VidAssembleLatentChainNood,
    LTXLat2VidAssembleSegmentChainNood,
    LTXImg2VidInplaceNood,
    LTXLat2VidInplaceNood,
    LTXLat2VidPrepNextDataNood,
    LTXLat2VidPrepSaveDataNood,
    LTXLat2VidResolveSegmentChainNood,
    LTXLat2VidSegmentLoadNood,
    LTXLat2VidSegmentSaveNood,
    LTXMaskParamsNood,
    StringIntAddNood,
    ULIDFromStrNood,
    ULIDPreviewNood,
    VideoGenParamsNood,
]

try:
    api = ComfyAPI()
except Exception:
    api = None  # type: ignore[assignment]


async def remap_old_node_ids():
    if not api:
        return

    # fix when I got the prep-save and prep-next nodes mixed up and they ended up with the wrong IDs (oops)
    await api.node_replacement.register(
        io.NodeReplace(old_node_id="LTXLat2VidGetNextSegmentDataNood", new_node_id="LTXLat2VidPrepSaveDataNood")
    )
    await api.node_replacement.register(
        io.NodeReplace(old_node_id="LTXLat2VidGetNextSegmentSaveDataNood", new_node_id="LTXLat2VidPrepNextDataNood")
    )


class NoodlesExtension(ComfyExtension):
    async def on_load(self) -> None:
        if not api:
            warnings.warn("Failed to initialize ComfyAPI, making the extension a no-op.", RuntimeWarning, stacklevel=2)
            return
        from . import routes  # noqa: F401

        await remap_old_node_ids()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _NODE_LIST.copy()

    def get_node_list_sync(self) -> list[type[io.ComfyNode]]:
        return _NODE_LIST.copy()


async def comfy_entrypoint() -> NoodlesExtension:
    return NoodlesExtension()
