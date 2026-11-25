import datetime as dt
import json

import polars as pl
import pytest

from bss_scenario.traffic import tomtom
from bss_scenario.traffic.tomtom import CollectionPlan, collect, queries, request_params, status

NOW = dt.datetime(2026, 11, 25, 12, tzinfo=dt.timezone.utc)
PLAN = CollectionPlan(days={"weekday": "2026-12-15", "saturday": "2026-12-19", "sunday": "2026-12-20"},
                      hours=(3, 8))


def od(n=2):
    return pl.DataFrame({"pair_id": list(range(n)), "o_lat": [42.37] * n, "o_lon": [-71.08] * n,
                         "d_lat": [42.373] * n, "d_lon": [-71.119] * n})


class FakeAPI:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or {}

    def __call__(self, url, params):
        self.calls.append((url, params))
        n = len(self.calls)
        if n in self.fail:
            return self.fail[n]
        hour = int(params["departAt"][11:13])
        slow = 1.3 if hour == 8 else 1.0
        return 200, {"routes": [{"summary": {"noTrafficTravelTimeInSeconds": 600,
                                             "historicTrafficTravelTimeInSeconds": int(600 * slow)},
                                 "legs": [{"points": [{"latitude": 42.37, "longitude": -71.08}]}]}]}


def run(cache, api, **kw):
    return collect(od(), cache, api, plan=PLAN, now=lambda: NOW, sleep=lambda s: None, **kw)


def test_plan_validation_and_default():
    with pytest.raises(ValueError, match="not a saturday"):
        CollectionPlan(days={"weekday": "2026-12-15", "saturday": "2026-12-18", "sunday": "2026-12-20"})
    with pytest.raises(ValueError, match="not a weekday"):
        CollectionPlan(days={"weekday": "2026-12-19", "saturday": "2026-12-19", "sunday": "2026-12-20"})
    p = CollectionPlan.default(dt.date(2026, 11, 25))
    assert p.days == {"weekday": "2026-12-15", "saturday": "2026-12-19", "sunday": "2026-12-20"}


def test_request_params_local_departure_and_no_key():
    q = next(queries(od(1), PLAN))
    url, params = request_params(q, PLAN)
    assert url.startswith("https://api.tomtom.com/routing/1/calculateRoute/42.370000,-71.080000:")
    assert params["departAt"] == "2026-12-15T03:00:00-05:00"  # EST in December
    assert params["computeTravelTimeFor"] == "all" and params["travelMode"] == "van"
    assert "key" not in params


def test_collect_caches_and_resumes(tmp_path):
    api = FakeAPI()
    r = run(tmp_path, api, max_requests=5)
    assert (r.requested, r.saved, r.stopped, r.remaining) == (5, 5, "max_requests", 7)  # 2 pairs x 3 days x 2 h
    r = run(tmp_path, api)
    assert (r.already_cached, r.requested, r.saved, r.stopped, r.remaining) == (5, 7, 7, "done", 0)
    assert run(tmp_path, api).requested == 0
    assert len(api.calls) == 12
    rec = json.loads((tmp_path / "weekday" / "h08" / "pair_0001.json").read_text())
    assert rec["response"]["routes"][0]["summary"]["historicTrafficTravelTimeInSeconds"] == 780
    assert rec["params"]["departAt"] == "2026-12-15T08:00:00-05:00"
    assert "key" not in json.dumps(rec)
    s = status(od(), tmp_path)
    assert (s["total"], s["cached"]) == (12, 12)


def test_queries_are_pair_major():
    qs = list(queries(od(), PLAN))
    assert [q.pair_id for q in qs[:6]] == [0] * 6


def test_quota_stops_and_errors_are_logged(tmp_path):
    api = FakeAPI(fail={2: (500, {"detailedError": {"code": "x"}}), 4: (429, None)})
    r = run(tmp_path, api)
    assert (r.requested, r.saved, r.errors, r.stopped) == (4, 2, 1, "quota")
    errors = (tmp_path / "errors.jsonl").read_text().splitlines()
    assert len(errors) == 1 and json.loads(errors[0])["status"] == 500
    r = run(tmp_path, FakeAPI())  # retried next run
    assert r.saved == 10 and r.remaining == 0


def test_too_many_errors(tmp_path):
    api = FakeAPI(fail={i: (0, {"error": "URLError"}) for i in range(1, 20)})
    assert run(tmp_path, api, max_consecutive_errors=3).stopped == "too_many_errors"


def test_plan_is_fixed_per_cache_and_must_be_future(tmp_path):
    run(tmp_path, FakeAPI(), max_requests=1)
    other = CollectionPlan(days={"weekday": "2026-12-22", "saturday": "2026-12-26", "sunday": "2026-12-27"})
    with pytest.raises(ValueError, match="different plan"):
        collect(od(), tmp_path, FakeAPI(), plan=other, now=lambda: NOW, sleep=lambda s: None)
    late = dt.datetime(2026, 12, 15, 9, tzinfo=dt.timezone.utc)
    with pytest.raises(ValueError, match="no longer in the future"):
        collect(od(), tmp_path, FakeAPI(), now=lambda: late, sleep=lambda s: None)


def test_api_key_from_env(monkeypatch):
    monkeypatch.delenv(tomtom.KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="TOMTOM_API_KEY"):
        tomtom.api_key_from_env()
    monkeypatch.setenv(tomtom.KEY_ENV, " abc ")
    assert tomtom.api_key_from_env() == "abc"
