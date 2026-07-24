"""
End-to-end dry run of f5_model v2 against a MOCKED MLB Stats API.

Fake world: 6 teams playing every other day for 30 days, 60 league batters,
platoon splits, pitcher hands, and linescores wired so each side's F5 runs
equal the runs charged to the opposing starter (real signal for the fit).

Slate on 2026-07-20:
    G1001  DET @ CLE  -> clean game, verdict expected
    G1002  KC  @ CWS  -> projected away lineup + low-PA callup; CWS starter is
                         a LEFTY so KC batters get vl platoon multipliers
    G1003  BOS @ NYY  -> BOS starter has 2 starts in 30 days -> pitcher "--",
                         no verdict (prediction needs all four scores)
Pipeline exercised: schema-v2 cache -> backtest (fit + tuning + params file)
-> live scoring with platoon adjustment and verdict output.
"""
import io
import json
import random
import shutil
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta

import f5_model as m

random.seed(42)
SLATE = date(2026, 7, 20)

TEAMS = {
    114: "Cleveland Guardians", 116: "Detroit Tigers", 118: "Kansas City Royals",
    145: "Chicago White Sox", 111: "Boston Red Sox", 147: "New York Yankees",
}
TEAM_IDS = list(TEAMS)

def roster(tid):
    return [tid * 100 + i for i in range(1, 10)]

def starter_ids(tid):
    return [91000 + tid * 10 + 1, 91000 + tid * 10 + 2]

def hand_of(pid):
    return "L" if pid % 3 == 0 else "R"

PITCHER_NAMES = {p: f"{TEAMS[t].split()[-1]} SP{i+1}"
                 for t in TEAM_IDS for i, p in enumerate(starter_ids(t))}
IL_RETURN_PITCHER = starter_ids(111)[0]
CALLUP_BATTER = 11810
CWS_SP1 = starter_ids(145)[0]
assert hand_of(CWS_SP1) == "L", "test setup: CWS SP1 must be a lefty"

def batter_line(pid):
    r = random.Random(pid)
    pa = 8 if pid == CALLUP_BATTER else r.randint(35, 60)
    avg = round(r.uniform(.180, .330), 3)
    obp = round(min(avg + r.uniform(.03, .10), .450), 3)
    slg = round(avg + r.uniform(.08, .28), 3)
    return {"plateAppearances": pa, "avg": f"{avg:.3f}",
            "obp": f"{obp:.3f}", "slg": f"{slg:.3f}"}

def platoon_split_stats(pid, season):
    """Seeded season + vl/vr splits; some hitters get big platoon skews."""
    r = random.Random(pid * 7 + season)
    obp, slg = round(r.uniform(.290, .380), 3), round(r.uniform(.350, .520), 3)
    skew = r.uniform(-0.12, 0.12)          # vl boost, vr equal-and-opposite
    def entry(pa, mult):
        return {"plateAppearances": pa,
                "obp": f"{max(obp * mult, .150):.3f}",
                "slg": f"{max(slg * mult, .200):.3f}"}
    return {"season": entry(r.randint(300, 550), 1.0),
            "vl": entry(r.randint(45, 160), 1.0 + skew),
            "vr": entry(r.randint(150, 400), 1.0 - skew / 3)}

# ---------------------------------------------------------------- fake season
GAMES, FINALS, PITCHER_STARTS = {}, [], {}

def make_feed(pk, d, home, away, hsp, asp):
    r = random.Random(pk)
    r5 = {"home": r.choice([0, 1, 1, 2, 2, 3, 4]),   # charged to HOME starter
          "away": r.choice([0, 1, 1, 2, 2, 3, 4])}   # charged to AWAY starter
    plays = []
    for side, pid in (("home", hsp), ("away", asp)):
        for _ in range(r5[side]):
            plays.append({"about": {"inning": r.randint(1, 5)},
                          "matchup": {"pitcher": {"id": pid}},
                          "runners": [{"movement": {"end": "score"}}]})
        plays.append({"about": {"inning": 7},                 # reliever noise
                      "matchup": {"pitcher": {"id": 88888}},
                      "runners": [{"movement": {"end": "score"}}]})
    # linescore: F5 runs consistent with the plays; innings 6-9 add noise
    innings = [{"num": 1, "away": {"runs": r5["home"]}, "home": {"runs": r5["away"]}}]
    innings += [{"num": n, "away": {"runs": 0}, "home": {"runs": 0}}
                for n in range(2, 6)]
    innings += [{"num": n, "away": {"runs": r.choice([0, 0, 1])},
                 "home": {"runs": r.choice([0, 0, 1])}} for n in range(6, 10)]
    box_sides, gd_players = {}, {}
    for side, tid, pid in (("home", home, hsp), ("away", away, asp)):
        outs = r.choice([15, 16, 17, 18, 18, 19, 20])
        gd_players[f"ID{pid}"] = {"pitchHand": {"code": hand_of(pid)}}
        box_sides[side] = {
            "team": {"id": tid, "name": TEAMS[tid]},
            "battingOrder": roster(tid),
            "pitchers": [pid, 77777],
            "players": {f"ID{pid}": {
                "person": {"fullName": PITCHER_NAMES[pid]},
                "stats": {"pitching": {
                    "inningsPitched": f"{outs // 3}.{outs % 3}",
                    "hits": r.randint(2, 9), "baseOnBalls": r.randint(0, 4),
                    "strikeOuts": r.randint(2, 10),
                    "battersFaced": 18 + outs}}}},
        }
        PITCHER_STARTS.setdefault(pid, []).append((d.isoformat(), pk))
    return {"gameData": {"game": {"pk": pk},
                         "datetime": {"officialDate": d.isoformat()},
                         "status": {"abstractGameState": "Final"},
                         "players": gd_players},
            "liveData": {"plays": {"allPlays": plays},
                         "linescore": {"innings": innings},
                         "boxscore": {"teams": box_sides}}}

pk = 5000
for day_off in range(30, 0, -1):
    d = SLATE - timedelta(days=day_off)
    if day_off % 2 == 0:
        continue
    order = TEAM_IDS[:]
    random.Random(day_off).shuffle(order)
    for i in range(0, 6, 2):
        home, away = order[i], order[i + 1]
        start_no = (day_off // 2) % 2
        hsp, asp = starter_ids(home)[start_no], starter_ids(away)[start_no]
        if hsp == IL_RETURN_PITCHER and day_off > 8:
            hsp = starter_ids(home)[1]
        if asp == IL_RETURN_PITCHER and day_off > 8:
            asp = starter_ids(away)[1]
        GAMES[pk] = make_feed(pk, d, home, away, hsp, asp)
        f5r = GAMES[pk]["liveData"]["linescore"]["innings"]
        FINALS.append({"game_id": pk, "game_date": d.isoformat(), "status": "Final",
                       "home_id": home, "home_name": TEAMS[home],
                       "home_score": random.Random(pk).randint(0, 8),
                       "away_id": away, "away_name": TEAMS[away],
                       "away_score": random.Random(pk + 1).randint(0, 8)})
        pk += 1

LEAGUE_BATTERS = [batter_line(90000 + i) for i in range(60)]

SLATE_GAMES = [
    (1001, 116, 114, starter_ids(116)[0], starter_ids(114)[0]),
    (1002, 118, 145, starter_ids(118)[0], CWS_SP1),
    (1003, 111, 147, IL_RETURN_PITCHER,  starter_ids(147)[0]),
]
KC_PROJECTED_LINEUP = roster(118)[:8] + [CALLUP_BATTER]
kc_recent = max((f for f in FINALS if 118 in (f["home_id"], f["away_id"])),
                key=lambda f: f["game_date"])
kc_side = "home" if kc_recent["home_id"] == 118 else "away"
GAMES[kc_recent["game_id"]]["liveData"]["boxscore"]["teams"][kc_side][
    "battingOrder"] = KC_PROJECTED_LINEUP

# ------------------------------------------------------------------- mock API
def mock_api_get(endpoint, params):
    if endpoint == "stats":
        if params.get("offset", 0) > 0:
            return {"stats": [{"splits": []}]}
        return {"stats": [{"splits": [{"stat": b} for b in LEAGUE_BATTERS]}]}
    if endpoint == "game":
        return GAMES[int(params["gamePk"])]
    if endpoint == "game_boxscore":
        g = int(params["gamePk"])
        if g in GAMES:
            return GAMES[g]["liveData"]["boxscore"]
        lineups = {1001: {"away": roster(116), "home": roster(114)},
                   1002: {"away": [], "home": roster(145)},
                   1003: {"away": roster(111), "home": roster(147)}}
        return {"teams": {s: {"battingOrder": lineups[g][s]}
                          for s in ("home", "away")}}
    if endpoint == "people":
        hyd = params.get("hydrate", "")
        ids = [int(x) for x in str(params["personIds"]).split(",")]
        if "statSplits" in hyd:
            season = int(hyd.split("season=")[1].rstrip(")"))
            people = []
            for i in ids:
                sp = platoon_split_stats(i, season)
                people.append({"id": i, "fullName": f"Batter {i}", "stats": [
                    {"type": {"displayName": "season"},
                     "splits": [{"stat": sp["season"]}]},
                    {"type": {"displayName": "statSplits"},
                     "splits": [{"split": {"code": c}, "stat": sp[c]}
                                for c in ("vl", "vr")]}]})
            return {"people": people}
        if "byDateRange" in hyd:
            return {"people": [
                {"id": i, "fullName": f"Batter {i}",
                 "stats": [{"splits": [{"stat": batter_line(i)}]}]} for i in ids]}
        if "gameLog" in hyd:
            pid = ids[0]
            splits = [{"date": d, "game": {"gamePk": g},
                       "stat": {"gamesStarted": 1}}
                      for d, g in sorted(PITCHER_STARTS.get(pid, []))]
            return {"people": [{"id": pid, "fullName": PITCHER_NAMES.get(pid, "?"),
                                "pitchHand": {"code": hand_of(pid)},
                                "stats": [{"splits": splits}]}]}
    if endpoint == "schedule":
        games = []
        for g, away, home, asp, hsp in SLATE_GAMES:
            games.append({"gamePk": g,
                          "status": {"abstractGameState": "Preview",
                                     "detailedState": "Scheduled"},
                          "teams": {
                            "away": {"team": {"id": away, "name": TEAMS[away]},
                                     "probablePitcher": {"id": asp,
                                        "fullName": PITCHER_NAMES[asp]}},
                            "home": {"team": {"id": home, "name": TEAMS[home]},
                                     "probablePitcher": {"id": hsp,
                                        "fullName": PITCHER_NAMES[hsp]}}}})
        return {"dates": [{"games": games}]}
    raise ValueError(f"unmocked endpoint {endpoint}")

def mock_schedule(**kw):
    start = datetime.strptime(kw["start_date"], "%m/%d/%Y").date().isoformat()
    end = datetime.strptime(kw["end_date"], "%m/%d/%Y").date().isoformat()
    out = [f for f in FINALS if start <= f["game_date"] <= end]
    if "team" in kw:
        out = [f for f in out if kw["team"] in (f["home_id"], f["away_id"])]
    return out

# ---------------------------------------------------------------------- run
if m.CACHE_DIR.exists():
    shutil.rmtree(m.CACHE_DIR)
m.api_get = mock_api_get
m.statsapi.schedule = mock_schedule
m.API_DELAY = 0

print(f"Fake league: {len(FINALS)} finals, {len(PITCHER_STARTS)} starters, "
      f"IL-return {PITCHER_NAMES[IL_RETURN_PITCHER]} has "
      f"{len(PITCHER_STARTS[IL_RETURN_PITCHER])} starts, "
      f"CWS SP1 throws {hand_of(CWS_SP1)}\n")

# -- 1) backtest over the last 12 mock days --
params = m.backtest(SLATE, start_date=SLATE - timedelta(days=12))
assert m.MODEL_PARAMS_FILE.exists()
for key in ("b0", "b1", "b2", "no_bet_edge", "strong_edge"):
    assert key in params, key
rows = json.loads(open(f"f5_backtest_rows_{SLATE.year}.json").read())
assert len(rows) >= 10, f"only {len(rows)} backtest rows"
assert all(r["outcome"] in ("home", "away", "tie") for r in rows)
# linescore wiring: F5 outcome fields present and consistent
for r in rows:
    gs = json.loads((m.GAME_CACHE_DIR / f'{r["gamePk"]}.json').read_text())
    assert gs["schema"] == m.SCHEMA_VERSION
    assert r["home_f5"] == gs["f5_runs"]["home"]
print("backtest assertions OK\n")

# -- 1b) weight sweep over the captured components --
assert all("features" in r for r in rows)
best = m.sweep(SLATE, apply=True)
assert (m.CACHE_DIR / "sweep_best.json").exists()
assert 0.4 <= best["hitter_form_weight"] <= 1.0
assert abs(sum(best["pitcher_weights"].values()) - 1.0) < 1e-6
assert abs(sum(best["slash_weights"].values()) - 1.0) < 1e-6
assert all("hf_comps" in r["features"]["home"] for r in
           json.loads(open(f"f5_backtest_rows_{SLATE.year}.json").read())
           if "features" in r)
applied = json.loads(m.MODEL_PARAMS_FILE.read_text())
assert applied["pitcher_weights"] == best["pitcher_weights"]
print("sweep assertions OK\n")

# -- 1c) market: CSV import covering backtest games, then ROI report --
import csv as _csv
with open("mock_odds.csv", "w", newline="") as fh:
    w = _csv.writer(fh)
    w.writerow(["date", "away", "home", "away_ml", "home_ml"])
    for f in FINALS:
        r = random.Random(f["game_id"] + 999)
        home_ml = r.choice([-150, -130, -115, 100, 110, 125])
        away_ml = {-150: 130, -130: 110, -115: -105, 100: -120,
                   110: -130, 125: -145}[home_ml]
        w.writerow([f["game_date"], f["away_name"], f["home_name"],
                    away_ml, home_ml])
m.import_odds_csv("mock_odds.csv")
arch = m.load_odds_archive(SLATE.year)
assert len(arch) == len(FINALS)
assert abs(m.devig(-110, -110) - 0.5) < 1e-9
assert abs(m.ml_to_prob(-150) - 0.6) < 1e-9
buf = io.StringIO()
with redirect_stdout(buf):
    m.market_report(SLATE)
mreport = buf.getvalue()
print(mreport)
assert "HOLDOUT" in mreport and "ROI" in mreport
assert "market_edge_min" in json.loads(m.MODEL_PARAMS_FILE.read_text())
print("market backtest assertions OK\n")

# -- 1c2) frozen evaluation runs and tunes nothing --
params_before = m.MODEL_PARAMS_FILE.read_text()
buf = io.StringIO()
with redirect_stdout(buf):
    m.evaluate(SLATE.year)
ereport = buf.getvalue()
print(ereport)
assert "FROZEN EVALUATION" in ereport
assert "pre-registered hypothesis bands" in ereport
assert "accuracy" in ereport and "log-loss" in ereport
assert m.MODEL_PARAMS_FILE.read_text() == params_before  # untouched
print("evaluate assertions OK\n")

# -- 1c3) crossfit smoke: duplicate rows as a fake second season --
import shutil as _sh
_sh.copy(f"f5_backtest_rows_{SLATE.year}.json", "f5_backtest_rows_2025.json")
buf = io.StringIO()
with redirect_stdout(buf):
    xbest = m.crossfit((2025, SLATE.year), apply=False)
xreport = buf.getvalue()
print(xreport[-600:])
assert "WINNER" in xreport and "cross log-loss" in xreport
assert abs(sum(xbest["pitcher_weights"].values()) - 1.0) < 1e-6
assert abs(sum(xbest["slash_weights"].values()) - 1.0) < 1e-6
assert (m.CACHE_DIR / "cross_best.json").exists()
import os as _os
_os.remove("f5_backtest_rows_2025.json")
print("crossfit assertions OK\n")

# -- 1d) fetch-odds for the slate via mocked Odds API --
import os
os.environ["ODDS_API_KEY"] = "test-key"
SLATE_EVENTS = [{"id": f"ev{g}", "away_team": TEAMS[a], "home_team": TEAMS[h]}
                for g, a, h, _, _ in SLATE_GAMES]
def mock_odds_http(path, params):
    if path.endswith("/events"):
        return SLATE_EVENTS
    eid = path.split("/")[-2]
    ev = next(e for e in SLATE_EVENTS if e["id"] == eid)
    return {"home_team": ev["home_team"], "away_team": ev["away_team"],
            "bookmakers": [{"key": "draftkings", "markets": [{
                "key": m.ODDS_MARKET_KEY, "outcomes": [
                    {"name": ev["home_team"], "price": -125},
                    {"name": ev["away_team"], "price": 105}]}]}]}
m._odds_http = mock_odds_http
got = m.fetch_odds_day(SLATE)
assert got == 3
print("fetch-odds assertions OK\n")

# -- 2) score the slate with the fitted model --
buf = io.StringIO()
with redirect_stdout(buf):
    results = m.score_slate(SLATE)
console = buf.getvalue()
print(console)

by_pk = {r["gamePk"]: r for r in results}
g1, g2, g3 = by_pk[1001], by_pk[1002], by_pk[1003]

assert g1["batters_available"] == {"qualified": 18, "total": 18, "pct": 100.0}
assert g1["prediction"] is not None
assert g1["prediction"]["mode"] == "market"          # odds were archived
assert g1["prediction"]["tier"] in ("STRONG", "LEAN", "NO VALUE")
assert g1["market"]["home_ml"] == -125
assert "MARKET:" in console and "VERDICT" in console
assert abs(g1["prediction"]["mkt_p_home"]
           - m.devig(-125, 105)) < 1e-3

# platoon: KC batters face the lefty -> at least one real multiplier != 1.0
kc_mults = [b.get("platoon_mult") for b in g2["away"]["batters"]
            if "platoon_mult" in b]
assert kc_mults and any(abs(mu - 1.0) > 0.005 for mu in kc_mults), kc_mults
assert all(1 - m.PLATOON_CLAMP <= mu <= 1 + m.PLATOON_CLAMP for mu in kc_mults)
assert g2["home"]["pitcher_hand"] == "L"
assert g2["away"]["lineup"] == "PROJECTED"
assert g2["batters_available"]["qualified"] == 17

# IL return: pitcher score None -> no prediction, batting still scored
assert g3["away"]["pitcher_score"] is None
assert g3["prediction"] is None
assert g3["away"]["batting_score"] is not None

snap = json.loads(m.SNAPSHOT_FILE.read_text())
assert len(snap["batting"]["avg"]) == 60
print("ALL V2 E2E ASSERTIONS PASS")
