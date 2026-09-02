"""Passive domain normalization; active DNS/WHOIS can be registered later."""

from __future__ import annotations

import asyncio
from time import perf_counter
from urllib.parse import urlsplit
import httpx
import config
from osint.parsers import parse_public_page

from osint.collectors.base import CollectorSpec
from osint.context import InvestigationContext
from osint.models import CollectorOutcome, Entity, EntityType, Evidence


class DomainCollector:
    spec = CollectorSpec("domain", (EntityType.DOMAIN, EntityType.WEBSITE), priority=20)
    async def collect(self, context: InvestigationContext, entity_id: str) -> CollectorOutcome:
        started, subject = perf_counter(), context.entities[entity_id]
        domain = urlsplit(subject.value if "://" in subject.value else f"https://{subject.value}").netloc or subject.value
        items, evidence, errors, requests = [Entity(EntityType.DOMAIN, domain, label=domain)], [], [], 0
        proof = Evidence("Domain derived from supplied public URL or domain.", source=self.spec.name); items[0].evidence_ids.append(proof.id); evidence.append(proof)
        try:
            context.budget.consume_request(); requests += 1
            addresses = await asyncio.get_running_loop().getaddrinfo(domain, None, type=0)
            values = sorted({address[4][0] for address in addresses})
            items[0].attributes["addresses"] = values
            evidence.append(Evidence("Public DNS address resolution completed.", source= self.spec.name, excerpt=", ".join(values)))
        except Exception as exc: errors.append(f"DNS: {exc}")
        # RDAP is a public registration endpoint. Its response is stored only as
        # minimal public metadata; no probing or active scan is performed.
        try:
            context.budget.consume_request(); requests += 1
            async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS) as client: response = await client.get(f"https://rdap.org/domain/{domain}")
            if response.is_success:
                data = response.json(); items[0].attributes["rdap"] = {"handle": data.get("handle", ""), "status": data.get("status", []), "events": data.get("events", [])[:5]}
                evidence.append(Evidence("Public RDAP record returned.", f"https://rdap.org/domain/{domain}", source=self.spec.name))
            else: errors.append(f"RDAP HTTP {response.status_code}")
        except Exception as exc: errors.append(f"RDAP: {exc}")
        if context.options.get("fetch_website", True):
            try:
                context.budget.consume_request(pages=1); requests += 1
                async with httpx.AsyncClient(timeout=config.OSINT_REQUEST_TIMEOUT_SECONDS, follow_redirects=True) as client: response = await client.get(f"https://{domain}")
                if response.is_success:
                    parsed = parse_public_page(response.text, str(response.url)); website = Entity(EntityType.WEBSITE, str(response.url), label=str(parsed["title"])[:200], attributes=parsed)
                    site_evidence = Evidence("Public website was fetched and parsed.", website.value, excerpt=str(parsed["title"])[:300], source=self.spec.name)
                    website.evidence_ids.append(site_evidence.id); items.append(website); evidence.append(site_evidence)
            except Exception as exc: errors.append(f"Website: {exc}")
        return CollectorOutcome(self.spec.name, True, items, evidence, errors=errors, requests=requests, elapsed_ms=int((perf_counter()-started)*1000))
