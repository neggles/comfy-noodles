from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noodles.comfy import comfy_entrypoint
else:
    from .src.noodles.comfy import comfy_entrypoint

__author__ = """Andi Powers-Holmes"""
__email__ = "aholmes@omnom.net"
__version__ = "0.1.0"


WEB_DIRECTORY = "./web"

__all__ = [
    "WEB_DIRECTORY",
    "NoodlesExtension",
    "comfy_entrypoint",
]
