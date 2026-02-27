from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noodles.extension import NoodlesExtension
else:
    from .src.noodles.extension import NoodlesExtension

__author__ = """Andi Powers-Holmes"""
__email__ = "aholmes@omnom.net"
__version__ = "0.1.0"


async def comfy_entrypoint() -> NoodlesExtension:
    return NoodlesExtension()


WEB_DIRECTORY = "./web"

__all__ = [
    "WEB_DIRECTORY",
    "NoodlesExtension",
    "comfy_entrypoint",
]
