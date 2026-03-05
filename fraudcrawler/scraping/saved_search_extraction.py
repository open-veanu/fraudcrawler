from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from fraudcrawler.scraping.saved_search_models import (
    SavedSearchCandidate,
    SavedSearchExtractionResult,
    SavedSearchRenderedNetworkCapture,
    SavedSearchRenderedProductListItem,
)

MAX_IMAGE_URLS_PER_CANDIDATE = 5
IMAGE_FIELD_NAME_PATTERN = re.compile(
    r"(image|thumbnail|thumb|photo|picture|gallery|media)", re.IGNORECASE
)
PRICE_FIELD_NAME_PATTERN = re.compile(
    r"(price|amount|saleprice|formattedprice|currentprice|offerprice)", re.IGNORECASE
)
DESCRIPTION_FIELD_NAME_PATTERN = re.compile(
    r"(description|fulldescription|shortdescription|summary|teaser|subtitle|details)",
    re.IGNORECASE,
)
IMAGE_ATTR_REGEX = re.compile(
    r"\b(?:src|data-src|data-srcset|srcset)\s*=\s*([\"'])(.*?)\1", re.IGNORECASE
)


def _dedupe_image_urls(
    urls: List[str], max_items: int = MAX_IMAGE_URLS_PER_CANDIDATE
) -> List[str]:
    deduped = []
    seen = set()
    for value in urls:
        txt = str(value or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        deduped.append(txt)
        if len(deduped) >= max_items:
            break
    return deduped


def _parse_srcset_candidates(srcset: str) -> List[str]:
    values = []
    for segment in srcset.split(","):
        token = segment.strip().split()[0].strip() if segment.strip() else ""
        if token:
            values.append(token)
    return values


def _normalize_image_url(
    source_url: str,
    raw_url: str,
    normalize_url: Callable[[str, str], Optional[str]],
) -> Optional[str]:
    token = str(raw_url or "").strip()
    if not token:
        return None
    if token.startswith(("data:", "blob:", "javascript:", "#")):
        return None
    normalized = normalize_url(source_url, token)
    if not normalized:
        return None
    if not normalized.startswith(("http://", "https://")):
        return None
    return normalized


def _extract_image_urls_from_html_context(
    anchor_html: str,
    surrounding_html: str,
    source_url: str,
    normalize_url: Callable[[str, str], Optional[str]],
) -> List[str]:
    context = f"{anchor_html} {surrounding_html}"
    raw_candidates: List[str] = []
    for match in IMAGE_ATTR_REGEX.finditer(context):
        value = (match.group(2) or "").strip()
        if not value:
            continue
        if "srcset" in match.group(0).lower():
            raw_candidates.extend(_parse_srcset_candidates(value))
        else:
            raw_candidates.append(value)
    normalized = [
        _normalize_image_url(source_url, entry, normalize_url)
        for entry in raw_candidates
    ]
    return _dedupe_image_urls([entry for entry in normalized if entry])


def _collect_string_leaves(value: Any, bucket: List[str]) -> None:
    if isinstance(value, str):
        bucket.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_string_leaves(item, bucket)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _collect_string_leaves(nested, bucket)


def _collect_potential_urls_from_text(text: str) -> List[str]:
    return [
        val.strip().rstrip("),.;")
        for val in re.findall(r"https?://[^\s\"'<>\\]+|/[A-Za-z0-9][^\s\"'<>\\]+", text)
        if val.strip()
    ]


def _collect_potential_urls_from_body(body_text: str) -> List[str]:
    direct_matches = _collect_potential_urls_from_text(body_text.replace("\\/", "/"))
    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError:
        return direct_matches
    leaves: List[str] = []
    _collect_string_leaves(parsed, leaves)
    from_leaves = []
    for leaf in leaves:
        from_leaves.extend(_collect_potential_urls_from_text(leaf.replace("\\/", "/")))
    return direct_matches + from_leaves


def _normalize_description(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = re.sub(r"<[^>]*>", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 20:
        return None
    return text[:4000]


def _normalize_price(value: Any) -> Optional[str]:
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return str(value)
    if not isinstance(value, str):
        return None
    txt = re.sub(r"\s+", " ", value).strip()
    if not re.search(r"\d", txt):
        return None
    return txt[:128]


def _extract_field_mappings_from_structured_body(
    body: Any,
    source_url: str,
    normalize_url: Callable[[str, str], Optional[str]],
) -> Dict[str, Dict[str, Any]]:
    mappings: Dict[str, Dict[str, Any]] = {}

    def add_mapping(
        candidate_url: str,
        images: List[str],
        price: Optional[str],
        description: Optional[str],
    ) -> None:
        prev = mappings.get(
            candidate_url, {"imageUrls": [], "price": None, "description": None}
        )
        merged_images = _dedupe_image_urls(list(prev["imageUrls"]) + images)
        mappings[candidate_url] = {
            "imageUrls": merged_images,
            "price": price if price is not None else prev["price"],
            "description": description
            if description is not None
            else prev["description"],
        }

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return

        candidate_urls: List[str] = []
        image_values: List[Any] = []
        price_values: List[Any] = []
        description_values: List[Any] = []
        for key, value in node.items():
            key_lower = str(key).lower()
            if isinstance(value, str) and (
                key_lower in {"url", "href"}
                or key_lower.endswith("url")
                or key_lower.endswith("_url")
            ):
                normalized = normalize_url(source_url, value)
                if normalized:
                    candidate_urls.append(normalized)
            if IMAGE_FIELD_NAME_PATTERN.search(key_lower):
                image_values.append(value)
            if PRICE_FIELD_NAME_PATTERN.search(key_lower):
                price_values.append(value)
            if DESCRIPTION_FIELD_NAME_PATTERN.search(key_lower):
                description_values.append(value)

        images: List[str] = []
        for raw in image_values:
            if isinstance(raw, str):
                for token in _collect_potential_urls_from_text(raw):
                    normalized = _normalize_image_url(source_url, token, normalize_url)
                    if normalized:
                        images.append(normalized)
            elif isinstance(raw, list):
                for element in raw:
                    if isinstance(element, str):
                        normalized = _normalize_image_url(
                            source_url, element, normalize_url
                        )
                        if normalized:
                            images.append(normalized)
        normalized_images = _dedupe_image_urls(images)
        price = next(
            (v for v in (_normalize_price(v) for v in price_values) if v), None
        )
        description = next(
            (v for v in (_normalize_description(v) for v in description_values) if v),
            None,
        )
        for candidate_url in candidate_urls:
            add_mapping(candidate_url, normalized_images, price, description)

        for value in node.values():
            visit(value)

    visit(body)
    return mappings


def extract_candidate_offers(
    html: str,
    source_url: str,
    max_items: int,
    normalize_url: Callable[[str, str], Optional[str]],
) -> SavedSearchExtractionResult:
    anchor_regex = re.compile(
        r"<a\b[^>]*href=([\"'])(.*?)\1[^>]*>([\s\S]*?)</a>", re.IGNORECASE
    )
    source_host = ""
    host_match = re.match(r"^https?://([^/:?#]+)", source_url)
    if host_match:
        source_host = host_match.group(1).lower()

    seen = set()
    candidates: List[SavedSearchCandidate] = []
    candidates_before = 0
    for match in anchor_regex.finditer(html):
        href = (match.group(2) or "").strip()
        inner_html = match.group(3) or ""
        text_title = re.sub(r"<[^>]*>", " ", inner_html)
        text_title = re.sub(r"\s+", " ", text_title).strip()
        if not href or len(text_title) < 6:
            continue
        if href.startswith(("#", "mailto:", "javascript:")):
            continue
        normalized = normalize_url(source_url, href)
        if not normalized:
            continue
        host_match = re.match(r"^https?://([^/:?#]+)", normalized)
        host = host_match.group(1).lower() if host_match else ""
        if source_host and host != source_host:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates_before += 1
        context_start = max(0, match.start() - 320)
        context_end = min(len(html), match.end() + 320)
        surrounding = html[context_start:context_end]
        image_urls = _extract_image_urls_from_html_context(
            match.group(0), surrounding, source_url, normalize_url
        )
        candidates.append(
            SavedSearchCandidate(url=normalized, title=text_title, imageUrls=image_urls)
        )
        if len(candidates) >= max_items:
            break

    return SavedSearchExtractionResult(
        candidates=candidates,
        strategy="generic",
        candidatesBefore=candidates_before,
        candidatesAfter=len(candidates),
        topRejectReason="no-product-candidates" if len(candidates) == 0 else None,
    )


def extract_candidates_from_rendered_product_list(
    items: List[SavedSearchRenderedProductListItem],
    source_url: str,
    max_items: int,
    normalize_url: Callable[[str, str], Optional[str]],
) -> List[SavedSearchCandidate]:
    if not items:
        return []
    results: List[SavedSearchCandidate] = []
    seen = set()
    for item in items:
        normalized = normalize_url(source_url, item.url or "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        image_candidates = []
        if item.main_image:
            image_candidates.append(item.main_image)
        image_candidates.extend(item.images or [])
        image_urls = _dedupe_image_urls(
            [
                entry
                for entry in (
                    _normalize_image_url(source_url, raw, normalize_url)
                    for raw in image_candidates
                )
                if entry
            ]
        )
        results.append(
            SavedSearchCandidate(
                url=normalized,
                title=(item.name or "").strip()
                or "Recovered from rendered product list",
                imageUrls=image_urls,
                price=_normalize_price(item.price),
                description=_normalize_description(item.description),
            )
        )
        if len(results) >= max_items:
            break
    return results


def extract_candidate_urls_from_render_captures(
    captures: List[SavedSearchRenderedNetworkCapture],
    source_url: str,
    max_items: int,
    normalize_url: Callable[[str, str], Optional[str]],
) -> List[SavedSearchCandidate]:
    if not captures:
        return []
    scored: Dict[str, int] = {}
    fields: Dict[str, Dict[str, Any]] = {}
    source_host_match = re.match(r"^https?://([^/:?#]+)", source_url)
    source_host = source_host_match.group(1).lower() if source_host_match else ""

    def keep_url(raw_url: str, base_url: str) -> None:
        normalized = normalize_url(base_url, raw_url)
        if not normalized:
            return
        if source_host:
            normalized_host_match = re.match(r"^https?://([^/:?#]+)", normalized)
            normalized_host = (
                normalized_host_match.group(1).lower() if normalized_host_match else ""
            )
            if normalized_host != source_host:
                return
        score = 4
        scored[normalized] = max(score, scored.get(normalized, 0))

    for capture in captures:
        if capture.url:
            keep_url(capture.url, source_url)
        body_text = (capture.body_text or "").strip()
        if not body_text:
            continue
        for entry in _collect_potential_urls_from_body(body_text):
            keep_url(entry, source_url)
        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError:
            continue
        mapping = _extract_field_mappings_from_structured_body(
            parsed, source_url, normalize_url
        )
        for candidate_url, candidate_fields in mapping.items():
            keep_url(candidate_url, source_url)
            current = fields.get(
                candidate_url, {"imageUrls": [], "price": None, "description": None}
            )
            fields[candidate_url] = {
                "imageUrls": _dedupe_image_urls(
                    list(current["imageUrls"]) + list(candidate_fields["imageUrls"])
                ),
                "price": candidate_fields["price"] or current["price"],
                "description": candidate_fields["description"]
                or current["description"],
            }

    urls = [
        url for url, _ in sorted(scored.items(), key=lambda item: item[1], reverse=True)
    ]
    urls = urls[:max_items]
    results = []
    for url in urls:
        payload = fields.get(url, {"imageUrls": [], "price": None, "description": None})
        results.append(
            SavedSearchCandidate(
                url=url,
                title="Recovered from rendered network response",
                imageUrls=payload["imageUrls"],
                price=payload["price"],
                description=payload["description"],
            )
        )
    return results
