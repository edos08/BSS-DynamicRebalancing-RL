"""Collect travel times with historical traffic from the TomTom Routing API (issue #56).

One request per (OD pair, day type, hour). ``computeTravelTimeFor=all`` returns, in one call, the
no-traffic time and the time with **historic** traffic for the requested departure, plus the
route geometry (needed later to attribute delays to road classes).

Design:

* **Cache = source of truth.** Every successful response is stored as JSON under
  ``<cache>/<day_type>/h<HH>/pair_<id>.json`` with the request parameters (never the key) and the
  time it was made. A run skips what is cached, so collection is resumable and the fit can be
  rerun offline.
* **Free tier.** At most ``max_requests`` per run (default 2,400 of the 2,500/day allowance) at
  ``max_qps``; a quota answer (HTTP 403/429) stops the run cleanly.
* **Departure dates** are fixed in a plan file (``plan.json`` in the cache) so every request of a
  collection uses the same reference days. TomTom only accepts future departures, so the plan
  dates must stay in the future for the whole collection.
* The API key is read from ``TOMTOM_API_KEY`` and never written anywhere.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

API_URL = "https://api.tomtom.com/routing/1/calculateRoute/{o_lat:.6f},{o_lon:.6f}:{d_lat:.6f},{d_lon:.6f}/json"
TIMEZONE = "America/New_York"
KEY_ENV = "TOMTOM_API_KEY"
DAY_TYPES = ("weekday", "saturday", "sunday")

#: fetch(url, params) -> (http_status, parsed_json_or_None). Injected in tests.
Fetch = Callable[[str, dict], tuple[int, dict | None]]


@dataclass(frozen=True)
class CollectionPlan:
    """Reference days (local dates) and request options shared by a whole collection."""

    days: dict[str, str] = field(default_factory=dict)  # day_type -> YYYY-MM-DD
    hours: tuple[int, ...] = tuple(range(24))
    travel_mode: str = "van"
    route_type: str = "fastest"

    def __post_init__(self) -> None:
        if set(self.days) != set(DAY_TYPES):
            raise ValueError(f"plan needs dates for {DAY_TYPES}, got {sorted(self.days)}")
        for day_type, date in self.days.items():
            weekday = dt.date.fromisoformat(date).weekday()
            expected = {"saturday": 5, "sunday": 6}.get(day_type)
            if (expected is None and weekday > 4) or (expected is not None and weekday != expected):
                raise ValueError(f"{date} is not a {day_type}")
        if not self.hours or any(not 0 <= h < 24 for h in self.hours):
            raise ValueError("hours must be within 0..23")

    def depart_at(self, day_type: str, hour: int) -> dt.datetime:
        date = dt.date.fromisoformat(self.days[day_type])
        return dt.datetime(date.year, date.month, date.day, hour, tzinfo=ZoneInfo(TIMEZONE))

    def first_departure(self) -> dt.datetime:
        return min(self.depart_at(d, min(self.hours)) for d in self.days)

    @classmethod
    def default(cls, today: dt.date, lead_days: int = 14) -> CollectionPlan:
        """Tuesday, Saturday and Sunday of the first full week starting ``lead_days`` from ``today``."""
        start = today + dt.timedelta(days=lead_days)
        monday = start + dt.timedelta(days=(7 - start.weekday()) % 7)
        return cls(days={
            "weekday": (monday + dt.timedelta(days=1)).isoformat(),
            "saturday": (monday + dt.timedelta(days=5)).isoformat(),
            "sunday": (monday + dt.timedelta(days=6)).isoformat(),
        })

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> CollectionPlan:
        data = json.loads(path.read_text())
        return cls(days=data["days"], hours=tuple(data["hours"]), travel_mode=data["travel_mode"],
                   route_type=data["route_type"])


@dataclass(frozen=True)
class Query:
    pair_id: int
    day_type: str
    hour: int
    o_lat: float
    o_lon: float
    d_lat: float
    d_lon: float

    def cache_path(self, cache: Path) -> Path:
        return cache / self.day_type / f"h{self.hour:02d}" / f"pair_{self.pair_id:04d}.json"


def queries(od: pl.DataFrame, plan: CollectionPlan) -> Iterator[Query]:
    """All queries, pair-major so that a partial collection has full daily profiles for its pairs."""
    for row in od.sort("pair_id").iter_rows(named=True):
        for day_type in DAY_TYPES:
            for hour in plan.hours:
                yield Query(row["pair_id"], day_type, hour, row["o_lat"], row["o_lon"], row["d_lat"], row["d_lon"])


def request_params(query: Query, plan: CollectionPlan) -> tuple[str, dict]:
    url = API_URL.format(o_lat=query.o_lat, o_lon=query.o_lon, d_lat=query.d_lat, d_lon=query.d_lon)
    params = {
        "departAt": plan.depart_at(query.day_type, query.hour).isoformat(),
        "traffic": "true",
        "computeTravelTimeFor": "all",
        "travelMode": plan.travel_mode,
        "routeType": plan.route_type,
        "routeRepresentation": "polyline",
    }
    return url, params


def urllib_fetch(api_key: str, timeout: float = 30.0) -> Fetch:
    def fetch(url: str, params: dict) -> tuple[int, dict | None]:
        full = url + "?" + urllib.parse.urlencode({**params, "key": api_key})
        try:
            with urllib.request.urlopen(full, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:  # message has the status only, never the URL/key
            try:
                body = json.loads(exc.read())
            except (ValueError, OSError):
                body = None
            return exc.code, body
        except (urllib.error.URLError, TimeoutError) as exc:
            return 0, {"error": type(exc).__name__}
    return fetch


@dataclass
class RunResult:
    requested: int = 0
    saved: int = 0
    errors: int = 0
    already_cached: int = 0
    remaining: int = 0
    stopped: str = "done"  # done | max_requests | quota | too_many_errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def collect(
    od: pl.DataFrame,
    cache: str | Path,
    fetch: Fetch,
    *,
    plan: CollectionPlan | None = None,
    max_requests: int = 2400,
    max_qps: float = 2.0,
    max_consecutive_errors: int = 20,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Run one batch of uncached queries. Creates ``plan.json`` on first use (or uses ``plan``)."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    plan_path = cache / "plan.json"
    if plan_path.exists():
        stored = CollectionPlan.load(plan_path)
        if plan is not None and plan != stored:
            raise ValueError(f"{plan_path} exists with a different plan; use a new cache directory")
        plan = stored
    else:
        plan = plan or CollectionPlan.default(now().date())
        plan.save(plan_path)
    if plan.first_departure() <= now() + dt.timedelta(hours=1):
        raise ValueError(f"plan dates {plan.days} are no longer in the future; start a new cache with a new plan")

    result = RunResult()
    todo = []
    for q in queries(od, plan):
        if q.cache_path(cache).exists():
            result.already_cached += 1
        else:
            todo.append(q)

    consecutive_errors = 0
    interval = 1.0 / max_qps
    for i, q in enumerate(todo):
        if result.requested >= max_requests:
            result.stopped = "max_requests"
            break
        url, params = request_params(q, plan)
        started = time.monotonic()
        status, body = fetch(url, params)
        result.requested += 1
        if status == 200 and body and "routes" in body:
            path = q.cache_path(cache)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"query": asdict(q), "params": params, "requested_at": now().isoformat(), "response": body}
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, separators=(",", ":")))
            tmp.replace(path)
            result.saved += 1
            consecutive_errors = 0
        elif status in (403, 429):
            result.stopped = "quota"
            break
        else:
            result.errors += 1
            consecutive_errors += 1
            _log_error(cache, q, status, body, now())
            if consecutive_errors >= max_consecutive_errors:
                result.stopped = "too_many_errors"
                break
        wait = interval - (time.monotonic() - started)
        if wait > 0 and i + 1 < len(todo):
            sleep(wait)
    result.remaining = len(todo) - result.saved
    return result


def _log_error(cache: Path, q: Query, status: int, body: dict | None, when: dt.datetime) -> None:
    with open(cache / "errors.jsonl", "a") as fh:
        fh.write(json.dumps({"query": asdict(q), "status": status, "body": body, "at": when.isoformat()}) + "\n")


def status(od: pl.DataFrame, cache: str | Path) -> dict:
    """Counts of cached vs remaining queries, per day type."""
    cache = Path(cache)
    plan = CollectionPlan.load(cache / "plan.json")
    out = {"plan": asdict(plan), "total": 0, "cached": 0, "by_day_type": {}}
    for q in queries(od, plan):
        done = q.cache_path(cache).exists()
        d = out["by_day_type"].setdefault(q.day_type, {"total": 0, "cached": 0})
        d["total"] += 1
        d["cached"] += done
        out["total"] += 1
        out["cached"] += done
    return out


def api_key_from_env() -> str:
    """The TomTom key from the settings (environment, ./.env or the user file; see .env.example)."""
    import bss_settings
    return bss_settings.require(KEY_ENV)
