from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import yt_dlp  # type: ignore


router = APIRouter()

class VideoRequest(BaseModel):
    query: str
    max_results: int = 5

@router.post("/recommend")
def recommend_videos(data: VideoRequest):
    search_url = f"ytsearch{data.max_results}:{data.query}"

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'force_generic_extractor': True
    }

    videos: List[Dict] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: # type: ignore
        info = ydl.extract_info(search_url, download=False)
        for entry in info.get("entries", []):
            videos.append({
                "title": entry.get("title"),
                "url": entry.get("url"),
                "duration": entry.get("duration"),
                "channel": entry.get("uploader"),
                "views": entry.get("view_count")
            })

    return {"videos": videos}
