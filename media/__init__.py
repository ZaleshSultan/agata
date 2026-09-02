"""Media automation package: providers, search, ranking, and playback."""

from media.base import ActionResult, MediaProvider, MediaResult
from media.ranking import rank_media_results
from media.soundcloud import SoundCloudProvider
from media.youtube import YouTubeProvider

__all__ = [
    "ActionResult",
    "MediaProvider",
    "MediaResult",
    "rank_media_results",
    "YouTubeProvider",
    "SoundCloudProvider",
]
