"""Public GitHub profile collector using its documented unauthenticated API."""

from __future__ import annotations

from time import perf_counter

import httpx

import config
from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence, Observation


class GitHubCollector:
    spec = CollectorSpec("github", (EntityType.USERNAME,), priority=20)

    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        url, cache_key = f"https://api.github.com/users/{subject.value}", f"github:{subject.value}"
        cached = context.cache.get(cache_key)
        if cached is not None: return cached
        try:
            context.budget.consume_request()
            async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS, headers={"Accept": "application/vnd.github+json"}) as client:
                response = await client.get(url)
            if response.status_code == 404:
                outcome = CollectorOutcome(self.spec.name, True, elapsed_ms=int((perf_counter()-started)*1000))
            elif response.is_success:
                data = response.json()
                profile = Entity(EntityType.SOCIAL, data["html_url"], label="GitHub", attributes={
                    "platform": "GitHub", "username": data.get("login", subject.value), "display_name": data.get("name") or "",
                    "bio": data.get("bio") or "", "website": data.get("blog") or "", "organization": data.get("company") or "", "public_email": data.get("email") or "", "created_at": data.get("created_at") or "", "updated_at": data.get("updated_at") or "",
                })
                proof = Evidence("Public GitHub profile returned by GitHub API.", profile.value, source=self.spec.name)
                profile.evidence_ids.append(proof.id)
                observation = Observation(profile.id, f"GitHub profile {data.get('login', subject.value)} was publicly returned.", evidence_ids=[proof.id], source=self.spec.name)
                entities, proofs, observations = [profile], [proof], [observation]
                for field, kind in (("email", EntityType.EMAIL), ("blog", EntityType.WEBSITE), ("company", EntityType.ORGANIZATION)):
                    value = str(data.get(field) or "").strip()
                    if value:
                        child = Entity(kind, value, label=value, attributes={"publicly_exposed_by": "GitHub"}); child.evidence_ids.append(proof.id); entities.append(child)
                outcome = CollectorOutcome(self.spec.name, True, entities, proofs, observations, requests=1, elapsed_ms=int((perf_counter()-started)*1000))
            else:
                outcome = CollectorOutcome(self.spec.name, False, errors=[f"GitHub returned HTTP {response.status_code}"], requests=1, elapsed_ms=int((perf_counter()-started)*1000))
        except Exception as exc:
            outcome = CollectorOutcome(self.spec.name, False, errors=[f"GitHub collector failed: {exc}"], elapsed_ms=int((perf_counter()-started)*1000))
        context.cache.set(cache_key, outcome); return outcome
