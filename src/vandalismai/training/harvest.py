# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pywikibot
from pywikibot.data.api import Request

from muffybot.env import get_int_env
from muffybot.paths import ENVIKIDIA_DIR, ROOT_DIR
from muffybot.wiki import connect_site, prepare_runtime

from .dataset import _is_revert_comment


def _safe_text(value: object) -> str:
    return str(value or "")


def _to_iso(value: object) -> str:
    text = _safe_text(value).strip()
    if not text:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
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


def _extract_changed_text(site: pywikibot.Site, old_revid: int, new_revid: int) -> tuple[str, str]:
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
        compare = payload.get("compare", {})
        html = _safe_text(compare.get("*") or compare.get("body"))
        return html[:4000], ""
    except Exception:
        return "", ""


def _state_path(output_dir: Path, lang: str) -> Path:
    return output_dir / f"state_{lang}.json"


def _load_state(output_dir: Path, lang: str) -> dict[str, Any]:
    path = _state_path(output_dir, lang)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def _save_state(output_dir: Path, lang: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_shard(
    *,
    lang: str,
    max_changes: int,
    max_diffs: int,
    output_dir: Path,
    include_non_reverts: bool = True,
) -> Path:
    workdir = ROOT_DIR if lang == "fr" else ENVIKIDIA_DIR
    prepare_runtime(workdir, script_name=f"ml_harvest_{lang}")
    site = connect_site(lang=lang, family="vikidia")

    state = _load_state(output_dir, lang)
    last_ts = _parse_ts(str(state.get("last_timestamp", "")))
    last_revid = int(state.get("last_revid", 0) or 0)

    rows: list[dict[str, Any]] = []
    diffs_done = 0
    changes_seen = 0
    hard_cap = max(get_int_env("ML_HARVEST_HARD_CAP", 50000), 1000)
    for idx, change in enumerate(site.recentchanges(total=max_changes, changetype="edit"), 1):
        if idx > hard_cap:
            break

        title = _safe_text(change.get("title"))
        user = _safe_text(change.get("user"))
        comment = _safe_text(change.get("comment"))
        tags = list(change.get("tags", []) or [])
        timestamp = _to_iso(change.get("timestamp"))
        revid = int(change.get("revid") or 0)
        old_revid = int(change.get("old_revid") or 0)
        parsed_ts = _parse_ts(timestamp)
        if parsed_ts is not None and last_ts is not None:
            if parsed_ts < last_ts:
                continue
            if parsed_ts == last_ts and revid <= last_revid:
                continue

        is_positive = _is_revert_comment(comment, tags)
        if not include_non_reverts and not is_positive:
            continue

        added_text = ""
        removed_text = ""
        if revid and old_revid and diffs_done < max_diffs:
            added_text, removed_text = _extract_changed_text(site, old_revid, revid)
            diffs_done += 1

        rows.append(
            {
                "lang": lang,
                "title": title,
                "user": user,
                "comment": comment,
                "added_text": added_text,
                "removed_text": removed_text,
                "label": 1 if is_positive else 0,
                "source": f"harvest_{lang}",
                "timestamp": timestamp,
                "revid": revid,
                "old_revid": old_revid,
                "tags": ",".join(str(tag) for tag in tags[:20]),
            }
        )
        changes_seen += 1

    if not rows:
        out = output_dir / f"{lang}_empty_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.parquet"
        pd.DataFrame(columns=["lang", "title", "user", "comment", "added_text", "removed_text", "label", "source", "timestamp"]).to_parquet(out, index=False)
        return out

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"{lang}_harvest_{stamp}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)

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
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "latest_file": str(out),
        },
    )
    return out
