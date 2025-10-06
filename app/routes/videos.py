from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from youtube_search import YoutubeSearch # type: ignore

router = APIRouter()

class VideoRequest(BaseModel):
    query: str
    max_results: int = 5

@router.post("/recommend")
def recommend_videos(data: VideoRequest):
    results = YoutubeSearch(data.query, max_results=data.max_results).to_dict()
    
    videos: List[Dict] = []
    for video in results:
        videos.append({
            "title": video.get("title"),
            "url": f"https://www.youtube.com{video.get('url_suffix')}",
            "duration": video.get("duration"),
            "channel": video.get("channel"),
            "views": video.get("views")
        })
    
    return {"videos": videos}
