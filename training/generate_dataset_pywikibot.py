#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pywikibot
from pywikibot.data.api import Request

REVERT_TAGS = {"mw-rollback", "mw-undo", "mw-manual-revert", "mw-reverted"}
REVERT_WORDS = ("revert", "annulation", "rv ", "rvv", "rollback", "restauration")
CONTEXT_PATTERNS = [
    re.compile(r"https?://|www\.", flags=re.IGNORECASE),
    re.compile(r"\b(?:test|lol|mdr|ptdr|xd+|haha+)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:fake|faux|hoax|spam)\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:idiot|stupide|nul|debile|imbecile)\b", flags=re.IGNORECASE),
    re.compile(r"(.)\1{5,}"),
    re.compile(r"\b\d{6,}\b"),
    re.compile(r"[A-Z]{8,}"),
    re.compile(r"^[\W_]+$"),
]

OUTPUT_COLUMNS = [
    "lang",
    "title",
    "user",
    "comment",
    "added_text",
    "removed_text",
    "label",
    "source",
    "timestamp",
    "revid",
    "old_revid",
    "tags",
    "context_score",
    "sampling_bucket",
    "dedup_key",
]


def _safe_text(value: object) -> str:
    return str(value or "")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_iso(value: object) -> str:
    text = _safe_text(value).strip()
    if not text:
        return _iso_now()
    if text.endswith("Z") or "+" in text[10:]:
        return text
    return f"{text}Z"


def _parse_ts(text: str) -> datetime | None:
    raw = _safe_text(text).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_revert_comment(comment: str, tags: list[str]) -> bool:
    lowered = _safe_text(comment).lower()
    if any(word in lowered for word in REVERT_WORDS):
        return True
    tag_set = {str(tag).lower() for tag in tags}
    return bool(tag_set & REVERT_TAGS)


def _strip_html(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", _safe_text(text))
    clean = no_tags.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", clean).strip()


def _normalize_for_key(text: str, *, max_len: int = 600) -> str:
    compact = re.sub(r"\s+", " ", _safe_text(text).strip().lower())
    return compact[:max_len]


def _dedup_key(
    *,
    lang: str,
    revid: int,
    old_revid: int,
    timestamp: str,
    title: str,
    user: str,
    comment: str,
    added_text: str,
    removed_text: str,
) -> str:
    raw = "||".join(
        [
            _safe_text(lang),
            str(int(revid or 0)),
            str(int(old_revid or 0)),
            _normalize_for_key(timestamp, max_len=40),
            _normalize_for_key(title),
            _normalize_for_key(user, max_len=80),
            _normalize_for_key(comment),
            _normalize_for_key(added_text),
            _normalize_for_key(removed_text),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def _dedup_index_path(output_dir: Path, lang: str) -> Path:
    return output_dir / f"dedup_{lang}.txt"


def _load_dedup_index(output_dir: Path, lang: str) -> set[str]:
    index_path = _dedup_index_path(output_dir, lang)
    if not index_path.exists():
        return set()
    try:
        return {line.strip() for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except Exception:
        return set()


def _append_dedup_index(output_dir: Path, lang: str, keys: set[str]) -> None:
    if not keys:
        return
    index_path = _dedup_index_path(output_dir, lang)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        for key in sorted(keys):
            handle.write(f"{key}\n")


def _context_score(*, title: str, comment: str, added_text: str, removed_text: str, tags: list[str], is_revert: bool) -> int:
    score = 0
    tag_set = {str(tag).lower() for tag in tags}
    lowered_comment = _safe_text(comment).lower()
    text_blob = " \n ".join([_safe_text(title), _safe_text(comment), _safe_text(added_text), _safe_text(removed_text)])

    if tag_set & REVERT_TAGS:
        score += 6
    if is_revert:
        score += 4
    if any(word in lowered_comment for word in REVERT_WORDS):
        score += 2

    for pattern in CONTEXT_PATTERNS:
        if pattern.search(text_blob):
            score += 2

    if not (tag_set & REVERT_TAGS) and "revert" not in lowered_comment:
        score += 1
    return score


def _sample_easy_negative(dedup_key: str, easy_negative_rate: float) -> bool:
    if easy_negative_rate <= 0:
        return False
    if easy_negative_rate >= 1:
        return True
    h = int(dedup_key[:8], 16) / float(0xFFFFFFFF)
    return h < easy_negative_rate


def _extract_compare_cells(html_diff: str) -> tuple[str, str]:
    deleted = re.findall(r'<td class="diff-deletedline"[^>]*>(.*?)</td>', html_diff, flags=re.IGNORECASE | re.DOTALL)
    added = re.findall(r'<td class="diff-addedline"[^>]*>(.*?)</td>', html_diff, flags=re.IGNORECASE | re.DOTALL)
    deleted_text = _strip_html(" ".join(deleted))
    added_text = _strip_html(" ".join(added))
    return added_text, deleted_text


def _fetch_compare_diff(site: pywikibot.Site, old_revid: int, new_revid: int) -> tuple[str, str]:
    try:
        payload = Request(
            site=site,
            parameters={
                "action": "compare",
                "fromrev": str(old_revid),
                "torev": str(new_revid),
                "prop": "diff",
            },
        ).submit()
    except Exception:
        return "", ""

    compare = payload.get("compare", {})
    html_body = _safe_text(compare.get("*") or compare.get("body"))
    if not html_body:
        return "", ""

    return _extract_compare_cells(html_body)


def _state_path(output_dir: Path, lang: str) -> Path:
    return output_dir / f"state_{lang}.json"


def _load_state(output_dir: Path, lang: str) -> dict[str, Any]:
    path = _state_path(output_dir, lang)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(output_dir: Path, lang: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_shard(
    *,
    lang: str,
    family: str,
    max_changes: int,
    max_diffs: int,
    output_dir: Path,
    include_non_reverts: bool,
    source_prefix: str,
    easy_negative_rate: float,
    min_context_score: int,
    disable_auto_sampling: bool,
) -> Path:
    site = pywikibot.Site(code=lang, fam=family)

    state = _load_state(output_dir, lang)
    last_ts = _parse_ts(str(state.get("last_timestamp", "")))
    last_revid = int(state.get("last_revid", 0) or 0)

    rows: list[dict[str, Any]] = []
    seen_keys = _load_dedup_index(output_dir, lang)
    new_keys: set[str] = set()
    diffs_done = 0
    changes_seen = 0
    dedup_dropped = 0
    sampled_dropped = 0

    for change in site.recentchanges(total=max_changes, changetype="edit"):
        title = _safe_text(change.get("title"))
        user = _safe_text(change.get("user"))
        comment = _safe_text(change.get("comment"))
        tags = [str(tag) for tag in (change.get("tags", []) or [])]
        timestamp = _to_iso(change.get("timestamp"))
        revid = int(change.get("revid") or 0)
        old_revid = int(change.get("old_revid") or 0)

        parsed_ts = _parse_ts(timestamp)
        if parsed_ts is not None and last_ts is not None:
            if parsed_ts < last_ts:
                continue
            if parsed_ts == last_ts and revid <= last_revid:
                continue

        is_revert = _is_revert_comment(comment, tags)
        if not include_non_reverts and not is_revert:
            continue

        added_text = ""
        removed_text = ""
        if revid and old_revid and diffs_done < max_diffs:
            added_text, removed_text = _fetch_compare_diff(site, old_revid, revid)
            diffs_done += 1

        dedup_key = _dedup_key(
            lang=lang,
            revid=revid,
            old_revid=old_revid,
            timestamp=timestamp,
            title=title,
            user=user,
            comment=comment,
            added_text=added_text,
            removed_text=removed_text,
        )
        if dedup_key in seen_keys or dedup_key in new_keys:
            dedup_dropped += 1
            continue

        ctx_score = _context_score(
            title=title,
            comment=comment,
            added_text=added_text,
            removed_text=removed_text,
            tags=tags,
            is_revert=is_revert,
        )
        bucket = "positive" if is_revert else "hard_negative"
        if not is_revert:
            if disable_auto_sampling:
                bucket = "negative_full"
            elif ctx_score < min_context_score:
                if not _sample_easy_negative(dedup_key, easy_negative_rate):
                    sampled_dropped += 1
                    continue
                bucket = "easy_negative_sampled"

        rows.append(
            {
                "lang": lang,
                "title": title,
                "user": user,
                "comment": comment,
                "added_text": added_text,
                "removed_text": removed_text,
                "label": 1 if is_revert else 0,
                "source": f"{source_prefix}_{lang}",
                "timestamp": timestamp,
                "revid": revid,
                "old_revid": old_revid,
                "tags": ",".join(tags[:20]),
                "context_score": int(ctx_score),
                "sampling_bucket": bucket,
                "dedup_key": dedup_key,
            }
        )
        changes_seen += 1
        new_keys.add(dedup_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if rows:
        out = output_dir / f"{lang}_harvest_{stamp}.parquet"
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_parquet(out, index=False)
        _append_dedup_index(output_dir, lang, new_keys)
        newest = max(rows, key=lambda item: (item["timestamp"], int(item.get("revid") or 0)))
        _save_state(
            output_dir,
            lang,
            {
                "last_timestamp": newest["timestamp"],
                "last_revid": int(newest.get("revid") or 0),
                "rows_written": len(rows),
                "diffs_done": diffs_done,
                "changes_seen": changes_seen,
                "dedup_dropped": dedup_dropped,
                "sampled_dropped": sampled_dropped,
                "dedup_index_size": len(seen_keys) + len(new_keys),
                "updated_at": _iso_now(),
                "latest_file": str(out),
            },
        )
        return out

    out = output_dir / f"{lang}_empty_{stamp}.parquet"
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(out, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collecte des modifications Vikidia/Wikipedia via Pywikibot et génère des shards parquet pour l'entraînement IA."
    )
    parser.add_argument("--langs", default="fr,en", help="Codes langue séparés par virgule (ex: fr,en)")
    parser.add_argument("--family", default="vikidia", help="Famille MediaWiki (vikidia, wikipedia, ...)")
    parser.add_argument("--max-changes", type=int, default=2000, help="Nombre maximal de changements récents à parcourir par langue")
    parser.add_argument("--max-diffs", type=int, default=400, help="Nombre maximal de diffs API compare à récupérer par langue")
    parser.add_argument(
        "--output-dir",
        default="data/ml_runs/shards",
        help="Dossier de sortie des fichiers parquet + fichiers state_<lang>.json",
    )
    parser.add_argument(
        "--reverts-only",
        action="store_true",
        help="Ne conserver que les modifications détectées comme reverts (label=1)",
    )
    parser.add_argument("--source-prefix", default="harvest", help="Préfixe de la colonne source")
    parser.add_argument(
        "--easy-negative-rate",
        type=float,
        default=0.2,
        help="Taux de conservation des négatifs faciles (0..1), utilisé en sampling auto contextuel",
    )
    parser.add_argument(
        "--min-context-score",
        type=int,
        default=4,
        help="Seuil de score contextuel (regex/tags) sous lequel un négatif devient 'facile' et peut être sous-échantillonné",
    )
    parser.add_argument(
        "--disable-auto-sampling",
        action="store_true",
        help="Désactive le sous-échantillonnage des négatifs faciles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    langs = [chunk.strip() for chunk in str(args.langs).split(",") if chunk.strip()]
    if not langs:
        raise SystemExit("Aucune langue fournie via --langs")

    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] family={args.family} langs={langs} max_changes={args.max_changes} max_diffs={args.max_diffs}")
    print(f"[INFO] include_non_reverts={not args.reverts_only}")
    print(
        f"[INFO] sampling: easy_negative_rate={args.easy_negative_rate:.2f} "
        f"min_context_score={args.min_context_score} disable_auto_sampling={bool(args.disable_auto_sampling)}"
    )

    for lang in langs:
        try:
            out = collect_shard(
                lang=lang,
                family=args.family,
                max_changes=max(args.max_changes, 1),
                max_diffs=max(args.max_diffs, 0),
                output_dir=output_dir,
                include_non_reverts=not args.reverts_only,
                source_prefix=args.source_prefix,
                easy_negative_rate=min(max(float(args.easy_negative_rate), 0.0), 1.0),
                min_context_score=int(args.min_context_score),
                disable_auto_sampling=bool(args.disable_auto_sampling),
            )
            print(f"[OK] {lang}: {out}")
        except Exception as exc:
            print(f"[WARN] {lang}: échec de collecte ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
