# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

import re
import json
import sqlite3
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pywikibot
from pywikibot.data.api import Request
from pywikibot.data.api import Request

from muffybot.env import get_bool_env, get_int_env
from muffybot.files import read_json
from muffybot.paths import ENVIKIDIA_DIR, ROOT_DIR
from muffybot.wiki import connect_site, prepare_runtime

from ..schemas import TrainingSample

REVERT_TAGS = {"mw-rollback", "mw-undo", "mw-manual-revert", "mw-reverted"}
REVERT_WORDS = ("revert", "annulation", "rv ", "rvv", "rollback", "restauration")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_iso(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return _iso_now()
    if text.endswith("Z") or "+" in text[10:]:
        return text
    return f"{text}Z"


def _is_revert_comment(comment: str, tags: list[str]) -> bool:
    lowered = (comment or "").lower()
    if any(word in lowered for word in REVERT_WORDS):
        return True
    tag_set = {str(tag).lower() for tag in tags}
    return bool(tag_set & REVERT_TAGS)


def _safe_text(value: object) -> str:
    return str(value or "")


def _from_human_corpus(path: Path) -> list[TrainingSample]:
    rows: list[TrainingSample] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(
            TrainingSample(
                lang=_safe_text(payload.get("lang", "fr")),
                title=_safe_text(payload.get("title")),
                user=_safe_text(payload.get("reverter")),
                comment=_safe_text(payload.get("comment")),
                added_text=_safe_text(payload.get("added_text")),
                removed_text=_safe_text(payload.get("removed_text")),
                label=1,
                source="human_revert",
                timestamp=_to_iso(payload.get("timestamp")),
            )
        )
    return rows


def _from_bot_db(path: Path, *, lang: str) -> list[TrainingSample]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    rows: list[TrainingSample] = []
    for _change_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        rows.append(
            TrainingSample(
                lang=lang,
                title=_safe_text(item.get("title")),
                user=_safe_text(item.get("creator")),
                comment=_safe_text(item.get("comment")),
                added_text=_safe_text(item.get("added_text")),
                removed_text=_safe_text(item.get("removed_text")),
                label=1,
                source="bot_revert",
                timestamp=_to_iso(item.get("timestamp")),
            )
        )
    return rows


def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(html or "")).replace("&nbsp;", " ").strip()


def _extract_compare_cells(html_diff: str) -> tuple[str, str]:
    deleted_chunks = re.findall(r'<td class="diff-deletedline"[^>]*>(.*?)</td>', html_diff, flags=re.IGNORECASE | re.DOTALL)
    added_chunks = re.findall(r'<td class="diff-addedline"[^>]*>(.*?)</td>', html_diff, flags=re.IGNORECASE | re.DOTALL)
    deleted_text = _strip_html(html.unescape(" ".join(deleted_chunks)))
    added_text = _strip_html(html.unescape(" ".join(added_chunks)))
    return added_text, deleted_text


def _fetch_compare_diff(site: pywikibot.Site, from_rev: int, to_rev: int) -> tuple[str, str]:
    try:
        req = Request(
            site=site,
            parameters={
                "action": "compare",
                "fromrev": str(from_rev),
                "torev": str(to_rev),
                "prop": "diff",
                "format": "json",
            },
        )
        data = req.submit()
        compare = data.get("compare", {})
        body = _strip_html(compare.get("*") or compare.get("body") or "")
        return body, ""
    except Exception:
        return "", ""


def _collect_human_reverts_from_api(*, lang: str, days: int, max_rc: int, max_diffs: int) -> list[TrainingSample]:
    workdir = ROOT_DIR if lang == "fr" else ENVIKIDIA_DIR
    prepare_runtime(workdir, script_name=f"ml_human_reverts_{lang}")
    site = connect_site(lang=lang, family="vikidia")
    now = datetime.now(timezone.utc)
    rc_start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rc_end = (now.timestamp() - (days * 86400))
    rc_end_iso = datetime.fromtimestamp(rc_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[TrainingSample] = []
    params: dict[str, str] = {
        "action": "query",
        "list": "recentchanges",
        "rctype": "edit",
        "rcprop": "ids|title|user|timestamp|comment|tags",
        "rclimit": "500",
        "rcstart": rc_start,
        "rcend": rc_end_iso,
    }
    # Cap number of recentchanges pages fetched.
    fetched_rc = 0
    fetched_diffs = 0
    cont: dict[str, str] | None = {}
    while cont is not None and fetched_rc < max_rc and fetched_diffs < max_diffs:
        query: dict[str, str] = dict(params)
        if cont:
            query.update(cont)
        try:
            payload = Request(site=site, parameters=query).submit()
        except Exception:
            break

        changes = payload.get("query", {}).get("recentchanges", [])
        for change in changes:
            fetched_rc += 1
            if fetched_rc > max_rc or fetched_diffs >= max_diffs:
                break
            comment = _safe_text(change.get("comment"))
            tags = list(change.get("tags", []) or [])
            if not _is_revert_comment(comment, tags):
                continue
            revid = int(change.get("revid") or 0)
            old_revid = int(change.get("old_revid") or 0)
            if not revid or not old_revid:
                continue
            try:
                cmp_payload = Request(
                    site=site,
                    parameters={
                        "action": "compare",
                        "fromrev": str(old_revid),
                        "torev": str(revid),
                        "prop": "diff",
                    },
                ).submit()
                cmp_body = _safe_text(cmp_payload.get("compare", {}).get("*") or cmp_payload.get("compare", {}).get("body"))
            except Exception:
                continue
            added_text, deleted_text = _extract_compare_cells(cmp_body)
            # Revert diff: deleted part usually corresponds to vandal text that gets removed.
            suspected_vandal_added = deleted_text
            suspected_vandal_removed = added_text
            rows.append(
                TrainingSample(
                    lang=lang,
                    title=_safe_text(change.get("title")),
                    user=_safe_text(change.get("user")),
                    comment=comment,
                    added_text=suspected_vandal_added,
                    removed_text=suspected_vandal_removed,
                    label=1,
                    source="human_revert_api",
                    timestamp=_to_iso(change.get("timestamp")),
                )
            )
            fetched_diffs += 1

        cont = payload.get("continue")
    return rows


def _collect_recent_changes_lang(*, lang: str, max_rc: int) -> list[TrainingSample]:
    workdir = ROOT_DIR if lang == "fr" else ENVIKIDIA_DIR
    prepare_runtime(workdir, script_name=f"ml_collect_{lang}")
    site = connect_site(lang=lang, family="vikidia")

    total = None if max_rc == 0 else max_rc
    hard_cap = max(get_int_env("ML_MAX_RC_HARD_CAP", 5000), 100)
    rows: list[TrainingSample] = []
    for idx, change in enumerate(site.recentchanges(total=total, changetype="edit"), 1):
        if max_rc == 0 and idx > hard_cap:
            break
        title = _safe_text(change.get("title"))
        user = _safe_text(change.get("user"))
        comment = _safe_text(change.get("comment"))
        tags = list(change.get("tags", []) or [])
        new_revid = int(change.get("revid") or 0)
        old_revid = int(change.get("old_revid") or 0)
        added_text, removed_text = ("", "")
        if new_revid and old_revid:
            added_text, removed_text = _fetch_compare_diff(site, old_revid, new_revid)

        label = 1 if _is_revert_comment(comment, tags) else 0
        rows.append(
            TrainingSample(
                lang=lang,
                title=title,
                user=user,
                comment=comment,
                added_text=added_text,
                removed_text=removed_text,
                label=label,
                source="recentchanges",
                timestamp=_to_iso(change.get("timestamp")),
            )
        )
    return rows


def _from_intel_db(path: Path) -> list[TrainingSample]:
    if not path.exists():
        return []
    rows: list[TrainingSample] = []
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        query = """
            SELECT ts_utc, lang, title, creator, action, score, reason, matched_patterns, added_len, removed_len
            FROM change_events
            ORDER BY id DESC
            LIMIT 30000
        """
        for item in conn.execute(query).fetchall():
            action = _safe_text(item["action"]).lower()
            label = 1 if action == "reverted" else 0
            matched = _safe_text(item["matched_patterns"])
            added_len = int(item["added_len"] or 0)
            removed_len = int(item["removed_len"] or 0)
            synthetic_added = f"len_added={added_len} score={float(item['score'] or 0.0):.3f} patterns={matched}"
            synthetic_removed = f"len_removed={removed_len}"
            rows.append(
                TrainingSample(
                    lang=_safe_text(item["lang"] or "fr"),
                    title=_safe_text(item["title"]),
                    user=_safe_text(item["creator"]),
                    comment=_safe_text(item["reason"]),
                    added_text=synthetic_added,
                    removed_text=synthetic_removed,
                    label=label,
                    source="intel_change_events",
                    timestamp=_to_iso(item["ts_utc"]),
                )
            )
    except Exception:
        return rows
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return rows


def _from_harvest_shards(shards_dir: Path) -> list[TrainingSample]:
    rows: list[TrainingSample] = []
    if not shards_dir.exists():
        return rows
    for path in sorted(shards_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        for item in frame.to_dict(orient="records"):
            rows.append(
                TrainingSample(
                    lang=_safe_text(item.get("lang", "fr")),
                    title=_safe_text(item.get("title")),
                    user=_safe_text(item.get("user")),
                    comment=_safe_text(item.get("comment")),
                    added_text=_safe_text(item.get("added_text")),
                    removed_text=_safe_text(item.get("removed_text")),
                    label=int(item.get("label", 0) or 0),
                    source=_safe_text(item.get("source") or f"shard:{path.name}"),
                    timestamp=_to_iso(item.get("timestamp")),
                )
            )
    return rows


def collect_training_samples() -> list[TrainingSample]:
    max_rc = get_int_env("ML_MAX_RC_PER_LANG", 0)
    collect_rc = get_bool_env("ML_COLLECT_RECENTCHANGES", True)
    rows: list[TrainingSample] = []
    rows.extend(_from_human_corpus(ROOT_DIR / "human_reverts_corpus.jsonl"))
    rows.extend(_from_bot_db(ROOT_DIR / "vandalism_db.json", lang="fr"))
    rows.extend(_from_bot_db(ENVIKIDIA_DIR / "vandalism_db.json", lang="en"))
    rows.extend(_from_intel_db(ROOT_DIR / "vandalism_intel.sqlite3"))
    rows.extend(_from_harvest_shards(ROOT_DIR / "ml_runs" / "shards"))
    if get_bool_env("ML_FETCH_HUMAN_REVERTS", True):
        days = max(get_int_env("ML_HUMAN_REVERT_DAYS", 30), 1)
        max_rc = max(get_int_env("ML_HUMAN_REVERT_MAX_RC", 4000), 100)
        max_diffs = max(get_int_env("ML_HUMAN_REVERT_MAX_DIFFS", 400), 50)
        if get_bool_env("ML_FETCH_HUMAN_REVERTS_FR", True):
            try:
                rows.extend(_collect_human_reverts_from_api(lang="fr", days=days, max_rc=max_rc, max_diffs=max_diffs))
            except Exception:
                pass
        if get_bool_env("ML_FETCH_HUMAN_REVERTS_EN", True):
            try:
                rows.extend(_collect_human_reverts_from_api(lang="en", days=days, max_rc=max_rc, max_diffs=max_diffs))
            except Exception:
                pass
    if collect_rc:
        try:
            rows.extend(_collect_recent_changes_lang(lang="fr", max_rc=max_rc))
        except Exception:
            pass
        try:
            rows.extend(_collect_recent_changes_lang(lang="en", max_rc=max_rc))
        except Exception:
            pass
    return rows


def to_dataframe(rows: list[TrainingSample]) -> pd.DataFrame:
    data = [
        {
            "lang": row.lang,
            "title": row.title,
            "user": row.user,
            "comment": row.comment,
            "added_text": row.added_text,
            "removed_text": row.removed_text,
            "label": int(row.label),
            "source": row.source,
            "timestamp": row.timestamp,
        }
        for row in rows
    ]
    return pd.DataFrame(data)
