from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def strip_code_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_payload(value: str) -> Any | None:
    candidate = strip_code_fences(value)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = normalize_text(text)
    return any(keyword in lowered for keyword in keywords)


def word_overlap_score(text: str, keywords: Iterable[str]) -> float:
    lowered = normalize_text(text)
    keywords = list(keywords)
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return min(1.0, hits / len(keywords))


def normalize_email(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def normalize_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return digits


def normalize_date(value: str) -> str:
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y", "%m/%d/%Y"]
    for pattern in formats:
        try:
            return datetime.strptime(value.strip(), pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value.strip()


def canonicalize_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "id": str(row.get("id", "")).strip().zfill(3),
        "name": " ".join(part.capitalize() for part in str(row.get("name", "")).strip().split()),
        "email": normalize_email(str(row.get("email", ""))),
        "signup_date": normalize_date(str(row.get("signup_date", ""))),
        "phone": normalize_phone(str(row.get("phone", ""))),
    }


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get("id", "")), str(row.get("name", ""))))


def compare_rows(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]]) -> tuple[float, Dict[str, float]]:
    total_fields = len(expected) * 5
    matched_fields = 0
    field_breakdown = {"id": 0.0, "name": 0.0, "email": 0.0, "signup_date": 0.0, "phone": 0.0}

    actual_by_id = {str(row.get("id", "")).strip().zfill(3): canonicalize_row(row) for row in actual if isinstance(row, dict)}
    for expected_row in expected:
        canonical_expected = canonicalize_row(expected_row)
        canonical_actual = actual_by_id.get(canonical_expected["id"])
        if not canonical_actual:
            continue
        for field in field_breakdown:
            if canonical_actual.get(field) == canonical_expected.get(field):
                matched_fields += 1
                field_breakdown[field] += 1.0

    score = matched_fields / total_fields if total_fields else 0.0
    for field in field_breakdown:
        field_breakdown[field] = field_breakdown[field] / max(len(expected), 1)
    return score, field_breakdown
