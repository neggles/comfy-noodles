from aiohttp import web
from server import PromptServer
from ulid import ULID

from .ltx.paths import get_video_folder_by_id, list_segment_files, list_video_folders

API_BASE = "/noodles"

routes = PromptServer.instance.routes


@routes.get(f"{API_BASE}/ltx/videos")
async def list_videos(request: web.Request) -> web.Response:
    subfolder = request.query.get("subfolder", "").strip("/")
    videos = []
    for name, ulid, folder in list_video_folders(subfolder, recursive=True, return_tuple=True):
        videos.append(
            {
                "name": str(name),
                "id": str(ulid),
                "folder": folder.as_posix(),
            }
        )
    return web.json_response(videos)


@routes.get(API_BASE + r"/ltx/videos/{video_id:v?\w{26}}")
async def get_video_by_id(request: web.Request) -> web.Response:
    video_id = request.match_info["video_id"]
    try:
        video_ulid = ULID.from_str(video_id)
    except ValueError as e:
        raise web.HTTPBadRequest(text=f"Invalid video ID: {video_id}") from e
    try:
        video_folder = get_video_folder_by_id(video_ulid)
    except FileNotFoundError as e:
        raise web.HTTPNotFound(text=f"Video not found: {video_id}") from e

    segments = []
    for index, iteration, path, id_suffix in list_segment_files(video_folder, return_tuple=True):
        segments.append(
            {
                "index": index,
                "iteration": iteration,
                "filename": path.relative_to(video_folder).as_posix(),
                "id_suffix": id_suffix,
            }
        )

    return web.json_response(
        {
            "id": video_id,
            "name": video_folder.name.rsplit("_", 1)[0],
            "folder": video_folder.as_posix(),
            "segments": segments,
        }
    )


@routes.get(API_BASE + r"/ltx/videos/{video_id:v?\w{26}}/segments")
async def list_video_segments(request: web.Request) -> web.Response:
    video_id = request.match_info["video_id"]
    try:
        video_ulid = ULID.from_str(video_id)
    except ValueError as e:
        raise web.HTTPBadRequest(text=f"Invalid video ID: {video_id}") from e
    try:
        video_folder = get_video_folder_by_id(video_ulid)
    except FileNotFoundError as e:
        raise web.HTTPNotFound(text=f"Video not found: {video_id}") from e

    segments = []
    for index, iteration, path, id_suffix in list_segment_files(video_folder, return_tuple=True):
        segments.append(
            {
                "index": index,
                "iteration": iteration,
                "filename": path.relative_to(video_folder).as_posix(),
                "id_suffix": id_suffix,
            }
        )

    return web.json_response(segments)
