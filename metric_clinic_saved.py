"""Remember dashboard settings between runs: the last session, recent scenarios and
named saves, all in one small JSON file (data/clinic_state.json by default; set
METRIC_CLINIC_STATE to use another file, as the tests do)."""

from __future__ import annotations

import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_RECENT = 10


def state_path() -> str:
    return os.environ.get("METRIC_CLINIC_STATE", os.path.join(HERE, "data", "clinic_state.json"))


def read() -> dict:
    try:
        with open(state_path()) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    d.setdefault("last", None)
    d.setdefault("recent", [])
    d.setdefault("saved", {})
    return d


def write(d: dict) -> None:
    os.makedirs(os.path.dirname(state_path()), exist_ok=True)
    tmp = state_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, state_path())


def record(settings: dict, summary: str) -> None:
    """Remember these settings as the last session and at the top of the recent list.
    Does nothing if they're unchanged since the last call."""
    d = read()
    if d["last"] and d["last"]["settings"] == settings:
        return
    entry = {"settings": settings, "summary": summary, "at": datetime.now().isoformat(timespec="seconds")}
    d["last"] = entry
    d["recent"] = [entry] + [e for e in d["recent"] if e["settings"] != settings][:MAX_RECENT - 1]
    write(d)


def save_named(name: str, settings: dict, summary: str) -> None:
    d = read()
    d["saved"][name] = {"settings": settings, "summary": summary,
                        "at": datetime.now().isoformat(timespec="seconds")}
    write(d)


def delete_named(name: str) -> None:
    d = read()
    d["saved"].pop(name, None)
    write(d)


def when(entry: dict) -> str:
    t = datetime.fromisoformat(entry["at"])
    return t.strftime("%H:%M") if t.date() == datetime.now().date() else t.strftime("%b %-d %H:%M")
