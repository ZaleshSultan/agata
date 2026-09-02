"""Independent public-source investigation framework."""
from osint.engine import OSINTEngine, get_engine
async def investigate(target: str, target_type: str = "auto", image: bytes | None = None, depth: str = "standard", options: dict | None = None) -> dict:
    return await get_engine().investigate(target, target_type, image, depth, options)
__all__ = ["OSINTEngine", "investigate", "get_engine"]
