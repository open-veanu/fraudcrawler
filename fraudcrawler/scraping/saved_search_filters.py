from __future__ import annotations

from typing import Iterable, List

from fraudcrawler.scraping.saved_search_models import (
    SavedSearchCandidate,
    SavedSearchPatternFilterResult,
)


def _normalize_tokens(tokens: Iterable[str]) -> List[str]:
    values = []
    seen = set()
    for token in tokens:
        txt = str(token).strip()
        if not txt:
            continue
        low = txt.lower()
        if low in seen:
            continue
        seen.add(low)
        values.append(low)
    return values


def apply_candidate_url_pattern_filters(
    candidates: List[SavedSearchCandidate],
    include_substrings: Iterable[str] | None = None,
    exclude_substrings: Iterable[str] | None = None,
) -> SavedSearchPatternFilterResult:
    include_tokens = _normalize_tokens(include_substrings or [])
    exclude_tokens = _normalize_tokens(exclude_substrings or [])

    result = SavedSearchPatternFilterResult(
        filteredCandidates=[],
        includeCount=len(include_tokens),
        excludeCount=len(exclude_tokens),
    )
    if not include_tokens and not exclude_tokens:
        result.filtered_candidates = candidates
        return result

    kept: List[SavedSearchCandidate] = []
    for candidate in candidates:
        lowered_url = candidate.url.lower()
        if include_tokens and not all(tok in lowered_url for tok in include_tokens):
            result.dropped_by_missing_include_substring += 1
            result.dropped_by_missing_include_all_match += 1
            if not result.first_dropped_by_missing_include_substring:
                result.first_dropped_by_missing_include_substring = candidate.url
            continue
        if exclude_tokens and any(tok in lowered_url for tok in exclude_tokens):
            result.dropped_by_exclude_substring += 1
            if not result.first_dropped_by_exclude_substring:
                result.first_dropped_by_exclude_substring = candidate.url
            continue
        kept.append(candidate)

    result.filtered_candidates = kept
    return result
