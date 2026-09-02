"""Small, public username discovery set; deliberately not mass enumeration."""

from __future__ import annotations

from time import perf_counter
import httpx
import config
from osint.parsers import parse_public_page

from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence
from osint.collectors.base import CollectorSpec


class UsernameCollector:
    spec = CollectorSpec("username_discovery", (EntityType.USERNAME,), priority=10)
    _PLATFORMS = {
        "GitHub": "https://github.com/{username}",
        "Reddit": "https://www.reddit.com/user/{username}",
        "X": "https://x.com/{username}",
        "YouTube": "https://www.youtube.com/@{username}",
    }

    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        items, evidence, errors, requests = [], [], [], 0
        try:
            async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS, follow_redirects=True, headers={"User-Agent": "AgataOSINT/1.0 (+public-source research)"}) as client:
                for platform, template in self._PLATFORMS.items():
                    context.budget.consume_request(); requests += 1
                    url = template.format(username=subject.value)
                    try:
                        response = await client.get(url)
                        if not response.is_success or len(response.text) < 100: continue
                        metadata = parse_public_page(response.text, str(response.url))
                        # A 200 response alone is never evidence of a real profile.
                        title, bio = str(metadata["title"]), str(metadata["bio"])
                        confidence = 0.35 + (0.2 if subject.value.casefold() in (title + bio).casefold() else 0)
                        item = Entity(EntityType.SOCIAL, str(response.url), label=platform, attributes={"platform": platform, "username": subject.value, "display_name": title[:200], "bio": bio[:1000], "avatar_url": metadata["avatar_url"], "external_links": metadata["external_links"], "profile_confidence": confidence})
                        proof = Evidence(f"Public {platform} candidate profile page responded and was parsed.", item.value, excerpt=title[:300], source=self.spec.name)
                        item.evidence_ids.append(proof.id); items.append(item); evidence.append(proof)
                    except Exception as exc: errors.append(f"{platform}: {exc}")
        except Exception as exc: errors.append(f"Username discovery failed: {exc}")
        return CollectorOutcome(self.spec.name, not errors or bool(items), items, evidence, errors=errors, requests=requests, elapsed_ms=int((perf_counter() - started) * 1000))
