"""
Fuzzy Command & Target Resolver using length-adaptive Levenshtein distance.

Provides robust matching for speech-to-text variations, typos, grammatical cases,
and phonetic inaccuracies without excessive false positives.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuzzyMatchResult:
    query: str
    matched_candidate: str
    target_value: Any
    score: float
    distance: int
    normalized_distance: float


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the standard Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    len_s1, len_s2 = len(s1), len(s2)
    if len_s1 > len_s2:
        s1, s2 = s2, s1
        len_s1, len_s2 = len_s2, len_s1

    current_row = list(range(len_s1 + 1))
    for i, c2 in enumerate(s2):
        new_row = [i + 1] * (len_s1 + 1)
        for j, c1 in enumerate(s1):
            insertions = current_row[j + 1] + 1
            deletions = new_row[j] + 1
            substitutions = current_row[j] + (0 if c1 == c2 else 1)
            new_row[j + 1] = min(insertions, deletions, substitutions)
        current_row = new_row

    return current_row[len_s1]


def calculate_similarity(s1: str, s2: str) -> tuple[float, int, float]:
    """
    Calculate similarity score and distances.

    Returns:
        (score, raw_distance, normalized_distance)
        where score in [0.0, 1.0] (1.0 = exact match).
    """
    s1_clean = " ".join(s1.casefold().split())
    s2_clean = " ".join(s2.casefold().split())

    if s1_clean == s2_clean:
        return 1.0, 0, 0.0

    max_len = max(len(s1_clean), len(s2_clean))
    if max_len == 0:
        return 1.0, 0, 0.0

    dist = levenshtein_distance(s1_clean, s2_clean)
    norm_dist = dist / max_len
    score = max(0.0, 1.0 - norm_dist)
    return score, dist, norm_dist


def is_fuzzy_match(
    query: str,
    target: str,
    threshold: float | None = None,
    max_distance: int | None = None,
) -> tuple[bool, float, int, float]:
    """
    Check if query matches target using length-adaptive thresholding.

    Threshold rules:
    - Length <= 3: exact match only (distance 0)
    - Length 4..5: max distance 1, score >= 0.75
    - Length 6..8: max distance 2, score >= 0.70
    - Length >= 9: max distance 3, score >= 0.65

    Returns:
        (is_match, score, distance, normalized_distance)
    """
    q_clean = " ".join(query.casefold().split())
    t_clean = " ".join(target.casefold().split())

    if not q_clean or not t_clean:
        return False, 0.0, max(len(q_clean), len(t_clean)), 1.0

    if q_clean == t_clean:
        return True, 1.0, 0, 0.0

    score, dist, norm_dist = calculate_similarity(q_clean, t_clean)
    max_len = max(len(q_clean), len(t_clean))
    min_len = min(len(q_clean), len(t_clean))

    # Length-adaptive defaults if threshold not specified
    if threshold is not None:
        effective_threshold = threshold
        allowed_dist = max_distance if max_distance is not None else max(1, int(max_len * (1.0 - threshold)))
    else:
        if min_len <= 3:
            effective_threshold = 1.0
            allowed_dist = 0
        elif max_len <= 5:
            effective_threshold = 0.75
            allowed_dist = 1
        elif max_len <= 8:
            effective_threshold = 0.70
            allowed_dist = 2
        else:
            effective_threshold = 0.65
            allowed_dist = 3

    if max_distance is not None:
        allowed_dist = min(allowed_dist, max_distance)

    matched = (score >= effective_threshold) and (dist <= allowed_dist)
    return matched, score, dist, norm_dist


def fuzzy_resolve(
    query: str,
    candidates: Mapping[str, Any] | Sequence[str] | Iterable[str],
    threshold: float | None = None,
    max_distance: int | None = None,
) -> FuzzyMatchResult | None:
    """
    Match query against a collection of candidates.

    Args:
        query: User input string.
        candidates: Dict of {candidate_str: target_value} or list of candidate strings.
        threshold: Optional custom similarity threshold (0.0..1.0).
        max_distance: Optional hard ceiling on edit distance.

    Returns:
        FuzzyMatchResult for the highest-scoring candidate, or None if no match.
    """
    if not query or not query.strip():
        return None

    best_match: FuzzyMatchResult | None = None
    best_score: float = -1.0
    best_dist: int = 999

    if isinstance(candidates, Mapping):
        items = candidates.items()
    else:
        items = ((c, c) for c in candidates)

    for cand_str, target_val in items:
        matched, score, dist, norm_dist = is_fuzzy_match(
            query, cand_str, threshold=threshold, max_distance=max_distance
        )
        if matched:
            # Prefer higher score, then lower distance, then shorter candidate
            if (
                score > best_score
                or (score == best_score and dist < best_dist)
                or (score == best_score and dist == best_dist and len(cand_str) < (len(best_match.matched_candidate) if best_match else 999))
            ):
                best_score = score
                best_dist = dist
                best_match = FuzzyMatchResult(
                    query=query,
                    matched_candidate=cand_str,
                    target_value=target_val,
                    score=score,
                    distance=dist,
                    normalized_distance=norm_dist,
                )

    return best_match
