"""
F5 Streakiness Model — v2
=========================
Scores MLB games for the FIRST 5 INNINGS based on recent form ("streakiness").

Per team, per game:
  BATTING SCORE (0-100) = 70% hitter form + 30% team runs/game
    - hitter form: each confirmed starter's last-14-day AVG/OBP/SLG,
      percentile-ranked vs all qualified MLB batters over the same window,
      averaged into one number per batter, then simple-averaged across the 9.
    - team runs/game: last-14-day runs per game, percentile-ranked vs all 30 teams.
  PITCHER SCORE (0-100) = weighted percentiles over the starter's LAST 3 STARTS:
      runs allowed through 5 IP (40%, lower=better)
      WHIP                      (30%, lower=better)
      K%  (K/BF)                (20%, higher=better)
      IP per start              (10%, higher=better)

Data-availability rules (v1.1 — no game is skipped, you judge the number):
  - Batter counts as "available" with >= MIN_BATTER_PA plate appearances in the
    14-day window; each game shows qualified/total batters and a percentage.
  - Pitcher score prints as -- when the starter has < 3 starts in the last
    30 days (IL returns), no probable is listed, or data is missing.
  - If a confirmed lineup isn't posted yet, the most recent game's lineup is
    used and the game is marked "lineup projected".

Usage:
  python f5_model.py snapshot                  # build/refresh league percentile snapshot
  python f5_model.py score                     # score today's slate
  python f5_model.py score --date 07/21/2026   # score a specific date
  (run `snapshot` first; it caches ~30 days of game feeds on first build, then
   only fetches new games on subsequent days)

v2 additions:
  - HANDEDNESS: each batter's form is multiplied by his season-long platoon
    factor vs the opposing starter's throwing hand (OBP+SLG vs hand / overall,
    clamped to +/-15%, >= 40 PA vs that hand, prior-season fallback).
  - PREDICTION: a logistic model fit on backtested history outputs
    P(home leads after 5), a pick, and a strength tier (STRONG / LEAN / NO BET).
  - BACKTEST: `python f5_model.py backtest` reconstructs historical slates
    with as-of snapshots, actual lineups/starters, and true F5 outcomes,
    fits the weights (chronological 70/30 train/validation), tunes the
    no-bet band, and saves params to f5_cache/model_params.json.
"""

import argparse
import json
import os
import sys
import time
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import statsapi
from math import exp, log

# --------------------------------------------------------------------------
# CONFIG — every knob from discovery lives here
# --------------------------------------------------------------------------
BATTING_WINDOW_DAYS = 14          # batter/team-runs lookback
PITCHER_LAST_N_STARTS = 3         # starts used in the pitcher score
PITCHER_FLAG_WINDOW_DAYS = 30     # pitcher must have >= 3 starts inside this window
MIN_BATTER_PA = 15                # below this in 14 days -> batter flagged


HITTER_FORM_WEIGHT = 0.70
TEAM_RUNS_WEIGHT = 0.30
SLASH_WEIGHTS = {"avg": 1 / 3, "obp": 1 / 3, "slg": 1 / 3}

PITCHER_WEIGHTS = {               # must sum to 1.0
    "runs_thru_5": 0.40,          # lower is better
    "whip": 0.30,                 # lower is better
    "k_pct": 0.20,                # higher is better
    "ip_per_start": 0.10,         # higher is better
}
PITCHER_LOWER_IS_BETTER = {"runs_thru_5", "whip"}

CACHE_DIR = Path("./f5_cache")
GAME_CACHE_DIR = CACHE_DIR / "games"
SNAPSHOT_FILE = CACHE_DIR / "league_snapshot.json"
API_DELAY = 0.25                  # polite pause between API hits (seconds)

# ---- v2 ----
SCHEMA_VERSION = 2                # game-summary cache schema (bump = refetch)
PLATOON_PA_FLOOR = 40             # min PA vs a hand for the split to count
PLATOON_CLAMP = 0.15              # platoon multiplier limited to 1 +/- this
BACKTEST_DEFAULT_START = "05/01"  # earliest date with a full pitcher window
TRAIN_FRACTION = 0.70             # chronological train/validation split
TARGET_BET_VOLUME = 0.70          # "balanced": bet ~70% of scored games
STRONG_TIER_FRACTION = 0.25       # top quartile of edges -> STRONG
SPLITS_CACHE_DIR = CACHE_DIR / "splits"
BATSTATS_CACHE_DIR = CACHE_DIR / "batstats"
LEAGUE_BAT_CACHE_DIR = CACHE_DIR / "league_bat"
MODEL_PARAMS_FILE = CACHE_DIR / "model_params.json"
FORWARD_LOG = Path("./f5_forward_log.json")

# ---- market / odds (The Odds API) ----
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_MARKET_KEY = "h2h_1st_5_innings"   # F5 moneyline market
ODDS_REGIONS = "us"
ODDS_BOOK_PREFERENCE = ["draftkings", "fanduel", "betmgm", "caesars"]
MARKET_EDGE_MIN = 0.03      # bet only if model P beats de-vigged market P by this
MARKET_EDGE_STRONG = 0.06
ODDS_DIR = CACHE_DIR / "odds"


def odds_archive_file(season):
    return Path(f"./f5_odds_{season}.json")


def rows_path(season):
    """Backtest rows file; window-tagged so 7d/14d universes never collide."""
    if BATTING_WINDOW_DAYS == 14:
        return Path(f"./f5_backtest_rows_{season}.json")   # back-compat
    return Path(f"./f5_backtest_rows_{season}_w{BATTING_WINDOW_DAYS}.json")


# --------------------------------------------------------------------------
# SMALL HELPERS
# --------------------------------------------------------------------------
def mlb_date(d: date) -> str:
    """MLB Stats API wants MM/DD/YYYY."""
    return d.strftime("%m/%d/%Y")


def safe_float(v):
    """Parse '.295' / '0.295' / numbers; return None on junk like '-.--'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ip_to_outs(ip) -> int:
    """'5.2' innings-pitched notation -> 17 outs. Accepts str or float."""
    s = str(ip)
    if "." in s:
        whole, frac = s.split(".", 1)
        return int(whole or 0) * 3 + int(frac or 0)
    return int(s or 0) * 3


def percentile(sorted_vals, x, higher_is_better=True):
    """Percentile rank of x within a pre-sorted list (midpoint tie handling)."""
    if not sorted_vals or x is None:
        return None
    n = len(sorted_vals)
    lo = bisect_left(sorted_vals, x)
    hi = bisect_right(sorted_vals, x)
    pct = 100.0 * (lo + (hi - lo) / 2.0) / n
    return round(pct if higher_is_better else 100.0 - pct, 1)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None




def clamp(v, lo, hi):
    return max(lo, min(hi, v))

API_MAX_RETRIES = 5
API_BACKOFF = (2, 5, 10, 20, 40)  # seconds between attempts


def api_get(endpoint, params):
    """statsapi.get with a politeness delay and retries for transient
    failures (DNS blips, dropped connections, timeouts, 5xx responses).
    Permanent errors (400/403/404) fail immediately."""
    import requests
    last_err = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            out = statsapi.get(endpoint, params)
            time.sleep(API_DELAY)
            return out
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status is not None and status < 500:
                raise                      # our fault -> don't retry
            last_err = e
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
        if attempt < API_MAX_RETRIES:
            wait = API_BACKOFF[min(attempt, len(API_BACKOFF) - 1)]
            print(f"    network hiccup ({type(last_err).__name__}) — "
                  f"retrying in {wait}s ({attempt + 1}/{API_MAX_RETRIES}) ...")
            time.sleep(wait)
    raise last_err


# --------------------------------------------------------------------------
# GAME FEED EXTRACTION + CACHE
# One API call per game returns boxscore AND play-by-play; we distill it to
# just what the model needs and cache it forever (finished games don't change).
# --------------------------------------------------------------------------
def extract_game_summary(feed):
    game_data = feed["gameData"]
    live = feed["liveData"]
    innings = live.get("linescore", {}).get("innings", [])

    def f5(side):
        return sum((i.get(side, {}).get("runs") or 0)
                   for i in innings if i.get("num", 99) <= 5)

    summary = {
        "schema": SCHEMA_VERSION,
        "gamePk": game_data["game"]["pk"],
        "date": game_data["datetime"]["officialDate"],  # YYYY-MM-DD
        "final": game_data["status"]["abstractGameState"] == "Final",
        "innings_played": len(innings),
        "f5_runs": {"home": f5("home"), "away": f5("away")},
    }

    # Runs charged to each pitcher in innings 1-5, attributed play-by-play.
    # NOTE (documented simplification): a runner the starter leaves on base who
    # scores after his exit is charged here to the reliever's play, not to the
    # starter — slightly kinder than official ER accounting.
    runs_thru_5 = {}
    for play in live.get("plays", {}).get("allPlays", []):
        if play.get("about", {}).get("inning", 99) > 5:
            continue
        pid = play.get("matchup", {}).get("pitcher", {}).get("id")
        scored = sum(
            1
            for r in play.get("runners", [])
            if r.get("movement", {}).get("end") == "score"
        )
        if pid and scored:
            runs_thru_5[pid] = runs_thru_5.get(pid, 0) + scored

    for side in ("home", "away"):
        box = live["boxscore"]["teams"][side]
        team = {
            "team_id": box["team"]["id"],
            "team_name": box["team"]["name"],
            "batting_order": box.get("battingOrder", []),
            "starter": None,
        }
        pitchers = box.get("pitchers", [])
        if pitchers:
            sid = pitchers[0]  # first pitcher listed = the starter
            player = box.get("players", {}).get(f"ID{sid}", {})
            pstats = player.get("stats", {}).get("pitching", {})
            hand = (feed["gameData"].get("players", {})
                    .get(f"ID{sid}", {}).get("pitchHand", {}).get("code"))
            team["starter"] = {
                "id": sid,
                "name": player.get("person", {}).get("fullName", ""),
                "hand": hand,
                "outs": ip_to_outs(pstats.get("inningsPitched", "0.0")),
                "hits": pstats.get("hits", 0),
                "bb": pstats.get("baseOnBalls", 0),
                "k": pstats.get("strikeOuts", 0),
                "bf": pstats.get("battersFaced", 0),
                "runs_thru_5": runs_thru_5.get(sid, 0),
            }
        summary[side] = team
    return summary


def get_game_summary(game_pk, verbose=False):
    """Cached distilled game feed. Only final games are written to cache."""
    GAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = GAME_CACHE_DIR / f"{game_pk}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached.get("schema") == SCHEMA_VERSION:
            return cached
        # stale schema -> refetch below
    if verbose:
        print(f"    fetching game {game_pk} ...")
    feed = api_get("game", {"gamePk": game_pk})
    summary = extract_game_summary(feed)
    if summary["final"]:
        cache_file.write_text(json.dumps(summary))
    return summary




# --------------------------------------------------------------------------
# PLATOON SPLITS (v2) — season-long vs-LHP / vs-RHP, disk-cached per season
# --------------------------------------------------------------------------
def _splits_cache_path(season):
    return SPLITS_CACHE_DIR / f"splits_{season}.json"


def ensure_platoon_splits(pids, season):
    """Return {pid: {"vl": {...}|None, "vr": {...}|None, "season": {...}|None}}
    fetching (and caching) any ids not already on disk."""
    SPLITS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _splits_cache_path(season)
    cache = json.loads(path.read_text()) if path.exists() else {}
    missing = [p for p in pids if str(p) not in cache]
    for i in range(0, len(missing), 50):
        chunk = missing[i:i + 50]
        hydrate = (f"stats(group=[hitting],type=[season,statSplits],"
                   f"sitCodes=[vl,vr],season={season})")
        resp = api_get("people", {
            "personIds": ",".join(str(p) for p in chunk), "hydrate": hydrate})
        got = set()
        for person in resp.get("people", []):
            pid = person.get("id")
            rec = {"vl": None, "vr": None, "season": None}
            for group in person.get("stats", []):
                for sp in group.get("splits", []):
                    st = sp.get("stat", {})
                    entry = {"pa": st.get("plateAppearances", 0),
                             "obp": safe_float(st.get("obp")),
                             "slg": safe_float(st.get("slg"))}
                    code = ((sp.get("split") or {}).get("code")
                            or sp.get("sitCode"))
                    if code in ("vl", "vr"):
                        rec[code] = entry
                    elif rec["season"] is None:
                        rec["season"] = entry
            cache[str(pid)] = rec
            got.add(pid)
        for pid in chunk:            # no stats at all (e.g. pitchers batting)
            if pid not in got:
                cache[str(pid)] = {"vl": None, "vr": None, "season": None}
    if missing:
        path.write_text(json.dumps(cache))
    return cache


def platoon_multiplier(pid, opp_hand, split_caches):
    """split_caches: list of season caches in priority order
    (live: [current, prior]; backtest: [prior] to avoid future leakage)."""
    if opp_hand not in ("L", "R"):
        return 1.0
    code = "vl" if opp_hand == "L" else "vr"
    for cache in split_caches:
        rec = cache.get(str(pid))
        if not rec:
            continue
        sp, tot = rec.get(code), rec.get("season")
        if not sp or not tot or sp.get("pa", 0) < PLATOON_PA_FLOOR:
            continue
        num = (sp.get("obp") or 0) + (sp.get("slg") or 0)
        den = (tot.get("obp") or 0) + (tot.get("slg") or 0)
        if den > 0 and num > 0:
            return clamp(num / den, 1 - PLATOON_CLAMP, 1 + PLATOON_CLAMP)
    return 1.0


# --------------------------------------------------------------------------
# LEAGUE SNAPSHOT — the percentile context everything is ranked against
# --------------------------------------------------------------------------
def build_snapshot(as_of: date):
    end = as_of - timedelta(days=1)  # windows end YESTERDAY relative to slate
    bat_start = as_of - timedelta(days=BATTING_WINDOW_DAYS)
    pit_start = as_of - timedelta(days=PITCHER_FLAG_WINDOW_DAYS)
    print(f"Building league snapshot as of {as_of} "
          f"(batting {bat_start}..{end}, pitching {pit_start}..{end})")

    # ---- 1) League batter distribution (qualified = >= MIN_BATTER_PA) ----
    print("  [1/3] league batter 14-day slash lines ...")
    avgs, obps, slgs = [], [], []
    offset, page = 0, 500
    while True:
        resp = api_get("stats", {
            "stats": "byDateRange", "group": "hitting",
            "startDate": mlb_date(bat_start), "endDate": mlb_date(end),
            "sportIds": 1, "gameType": "R", "playerPool": "ALL",
            "limit": page, "offset": offset,
        })
        splits = (resp.get("stats") or [{}])[0].get("splits", [])
        for s in splits:
            st = s.get("stat", {})
            if st.get("plateAppearances", 0) >= MIN_BATTER_PA:
                a, o, g = (safe_float(st.get("avg")), safe_float(st.get("obp")),
                           safe_float(st.get("slg")))
                if None not in (a, o, g):
                    avgs.append(a); obps.append(o); slgs.append(g)
        if len(splits) < page:
            break
        offset += page
    print(f"        {len(avgs)} qualified batters")

    # ---- 2) Team runs/game over the batting window ----
    print("  [2/3] team runs per game ...")
    team_runs = {}
    sched = statsapi.schedule(start_date=mlb_date(bat_start), end_date=mlb_date(end))
    time.sleep(API_DELAY)
    for g in sched:
        if g.get("status") != "Final" or g.get("game_type", "R") != "R":
            continue
        for tid, name, runs in (
            (g["home_id"], g["home_name"], g.get("home_score", 0)),
            (g["away_id"], g["away_name"], g.get("away_score", 0)),
        ):
            rec = team_runs.setdefault(tid, {"name": name, "runs": 0, "games": 0})
            rec["runs"] += runs or 0
            rec["games"] += 1
    for rec in team_runs.values():
        rec["rpg"] = round(rec["runs"] / rec["games"], 3) if rec["games"] else None
    rpg_sorted = sorted(r["rpg"] for r in team_runs.values() if r["rpg"] is not None)

    # ---- 3) League starter distribution (last 3 starts each) ----
    print("  [3/3] starter last-3-start metrics (fetches/caches game feeds) ...")
    sched30 = statsapi.schedule(start_date=mlb_date(pit_start), end_date=mlb_date(end))
    time.sleep(API_DELAY)
    final_pks = [g["game_id"] for g in sched30 if g.get("status") == "Final"
                 and g.get("game_type", "R") == "R"]
    print(f"        {len(final_pks)} final games in window")
    starts_by_pitcher = {}
    for i, pk in enumerate(final_pks, 1):
        if i % 25 == 0:
            print(f"        ... {i}/{len(final_pks)}")
        gs = get_game_summary(pk)
        for side in ("home", "away"):
            st = gs[side].get("starter")
            if st:
                starts_by_pitcher.setdefault(st["id"], []).append(
                    {**st, "date": gs["date"]})

    dist = {k: [] for k in PITCHER_WEIGHTS}
    qualified_starters = 0
    for pid, starts in starts_by_pitcher.items():
        if len(starts) < PITCHER_LAST_N_STARTS:
            continue
        starts.sort(key=lambda s: s["date"], reverse=True)
        m = pitcher_metrics(starts[:PITCHER_LAST_N_STARTS])
        if m:
            qualified_starters += 1
            for k in dist:
                dist[k].append(m[k])
    for k in dist:
        dist[k].sort()
    print(f"        {qualified_starters} starters with >= "
          f"{PITCHER_LAST_N_STARTS} starts in the window")

    snapshot = {
        "as_of": as_of.isoformat(),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "batting": {"avg": sorted(avgs), "obp": sorted(obps), "slg": sorted(slgs)},
        "team_runs": {str(tid): rec for tid, rec in team_runs.items()},
        "rpg_sorted": rpg_sorted,
        "pitching": dist,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(snapshot))
    print(f"Snapshot saved -> {SNAPSHOT_FILE}")
    return snapshot


def pitcher_metrics(last_starts):
    """Combine a pitcher's last N starts into the four score inputs."""
    outs = sum(s["outs"] for s in last_starts)
    ip = outs / 3.0
    hits = sum(s["hits"] for s in last_starts)
    bb = sum(s["bb"] for s in last_starts)
    k = sum(s["k"] for s in last_starts)
    bf = sum(s["bf"] for s in last_starts)
    if bf == 0:
        return None
    return {
        "runs_thru_5": round(mean([s["runs_thru_5"] for s in last_starts]), 3),
        "whip": round((hits + bb) / ip, 3) if ip > 0 else 9.9,
        "k_pct": round(k / bf, 4),
        "ip_per_start": round(ip / len(last_starts), 2),
    }


def load_snapshot(as_of: date):
    if not SNAPSHOT_FILE.exists():
        print("No league snapshot found — building one first.")
        return build_snapshot(as_of)
    snap = json.loads(SNAPSHOT_FILE.read_text())
    if snap.get("as_of") != as_of.isoformat():
        print(f"Snapshot is dated {snap.get('as_of')} but slate is {as_of} — rebuilding.")
        return build_snapshot(as_of)
    return snap




# --------------------------------------------------------------------------
# MARKET ODDS (v2.1) — fetch, archive, convert to probabilities
# --------------------------------------------------------------------------
def _odds_http(path, params):
    """GET from The Odds API with the same retry discipline as api_get."""
    import requests
    url = f"{ODDS_API_BASE}{path}"
    last_err = None
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code >= 500:
                raise requests.exceptions.HTTPError(response=r)
            if r.status_code >= 400:
                sys.exit(f"Odds API error {r.status_code}: {r.text[:300]}")
            remaining = r.headers.get("x-requests-remaining")
            if remaining is not None:
                _odds_http.remaining = remaining
            time.sleep(API_DELAY)
            return r.json()
        except (Exception,) as e:
            last_err = e
            if attempt < API_MAX_RETRIES:
                wait = API_BACKOFF[min(attempt, len(API_BACKOFF) - 1)]
                print(f"    odds network hiccup — retrying in {wait}s ...")
                time.sleep(wait)
    raise last_err


def norm_team(name):
    return " ".join(str(name).lower().split())


def odds_key(d_iso, away, home):
    return f"{d_iso}|{norm_team(away)}|{norm_team(home)}"


# --------------------------------------------------------------------------
# CLOSING LINE VALUE — shared snapshot plumbing (used by both models)
#
# The closing line is the sharpest number the market produces: by first pitch
# every lineup, scratch and weather report is priced in. Comparing the price
# we took against that close measures our EDGE without waiting on the game's
# outcome — which is the noisiest thing in the ledger. CLV converges in weeks
# where ROI takes seasons.
#
# Requires a paid Odds API plan (historical endpoints).
# --------------------------------------------------------------------------
CLV_MINUTES_BEFORE = 10      # snapshot this far before first pitch


def _iso_z(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def closing_snapshot_time(game_time_iso, minutes_before=CLV_MINUTES_BEFORE):
    """Timestamp we treat as 'the close' for one game. Backing off a few
    minutes from first pitch avoids catching in-play prices."""
    dt = datetime.fromisoformat(str(game_time_iso).replace("Z", "+00:00"))
    return _iso_z(dt - timedelta(minutes=minutes_before))


def historical_events(snapshot_iso, api_key):
    """Every event live in the historical snapshot nearest snapshot_iso."""
    resp = _odds_http("/historical/sports/baseball_mlb/events",
                      {"apiKey": api_key, "date": snapshot_iso})
    return resp.get("data", [])


def historical_event_odds(event_id, snapshot_iso, market_key, api_key):
    """One event's odds at a historical snapshot."""
    payload = _odds_http(
        f"/historical/sports/baseball_mlb/events/{event_id}/odds",
        {"apiKey": api_key, "regions": ODDS_REGIONS, "markets": market_key,
         "oddsFormat": "american", "date": snapshot_iso})
    return payload.get("data", {})


def clv_pending(log, now=None):
    """Logged bets that are past first pitch and still have no closing line.
    Grouped by snapshot time so one events call can serve several games."""
    now = now or datetime.now(timezone.utc)
    groups = {}
    for key, rec in log.items():
        if not rec.get("pick") or rec.get("close") is not None:
            continue
        gt = rec.get("game_time")
        if not gt:
            continue
        try:
            start = datetime.fromisoformat(str(gt).replace("Z", "+00:00"))
        except ValueError:
            continue
        if start > now:
            continue                      # hasn't closed yet; try tomorrow
        groups.setdefault(closing_snapshot_time(gt), []).append((key, rec))
    return groups


def clv_summary(rows, label="CLV"):
    """Print the beat-rate / average-CLV block shared by both models.
    rows: dicts with 'clv' (probability points, our side) and 'tier'."""
    scored = [r for r in rows if r.get("clv") is not None]
    if not scored:
        print(f"  {label}: no closing lines fetched yet "
              f"(run `clv` after games start).")
        return
    beat = sum(1 for r in scored if r["clv"] > 0)
    avg = mean([r["clv"] for r in scored])
    print(f"  {label}: {len(scored)} bets priced vs close \u2014 "
          f"beat the close {beat}/{len(scored)} ({beat / len(scored):.1%}), "
          f"average {avg:+.2%}")
    for tier in ("STRONG", "LEAN"):
        t = [r for r in scored if r.get("tier") == tier]
        if t:
            tb = sum(1 for r in t if r["clv"] > 0)
            print(f"      {tier:<7} {tb}/{len(t)} ({tb / len(t):.0%}), "
                  f"avg {mean([r['clv'] for r in t]):+.2%}")
    if len(scored) < 30:
        print("      (still early \u2014 CLV needs ~30+ bets for a trend, "
              "a few hundred to be solid)")
    elif beat / len(scored) >= 0.55:
        print("      55%+ beat rate is the bar for a real edge.")


def ml_to_prob(ml):
    ml = float(ml)
    return (-ml / (-ml + 100.0)) if ml < 0 else (100.0 / (ml + 100.0))


def devig(home_ml, away_ml):
    """2-way vig removal -> fair P(home). F5 ties push at most books,
    so the 2-way market matches our home/away/tie-push framing."""
    ph, pa = ml_to_prob(home_ml), ml_to_prob(away_ml)
    return ph / (ph + pa)


def load_odds_archive(season):
    f = odds_archive_file(season)
    return json.loads(f.read_text()) if f.exists() else {}


def save_odds_archive(season, archive):
    odds_archive_file(season).write_text(json.dumps(archive, indent=1))


def _extract_f5_market(event_odds):
    """Pick the preferred book's F5 h2h outcomes from an event-odds payload."""
    books = {b.get("key"): b for b in event_odds.get("bookmakers", [])}
    ordered = [books[k] for k in ODDS_BOOK_PREFERENCE if k in books]
    ordered += [b for k, b in books.items() if k not in ODDS_BOOK_PREFERENCE]
    for book in ordered:
        for mkt in book.get("markets", []):
            if mkt.get("key") != ODDS_MARKET_KEY:
                continue
            prices = {norm_team(o.get("name")): o.get("price")
                      for o in mkt.get("outcomes", [])}
            home = prices.get(norm_team(event_odds.get("home_team")))
            away = prices.get(norm_team(event_odds.get("away_team")))
            if home is not None and away is not None:
                return {"home_ml": home, "away_ml": away,
                        "book": book.get("key")}
    return None


def fetch_odds_day(d, historical=False):
    """Fetch F5 moneylines for date d into the season archive.
    Live mode: free tier. Historical mode: requires a paid Odds API plan."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("Set the ODDS_API_KEY environment variable "
                 "(free key at the-odds-api.com).")
    season = d.year
    archive = load_odds_archive(season)
    if historical:
        snap = f"{d.isoformat()}T16:00:00Z"      # morning-of US lines
        resp = _odds_http("/historical/sports/baseball_mlb/events",
                          {"apiKey": api_key, "date": snap})
        events = resp.get("data", [])
    else:
        resp = _odds_http("/sports/baseball_mlb/events",
                          {"apiKey": api_key,
                           "commenceTimeFrom": f"{d.isoformat()}T08:00:00Z",
                           "commenceTimeTo":
                               f"{(d + timedelta(days=1)).isoformat()}T09:00:00Z"})
        events = resp if isinstance(resp, list) else resp.get("data", [])
    got = skipped = 0
    for ev in events:
        ct = ev.get("commence_time")
        if ct:                     # game's local (ET) date; MLB season = EDT
            ev_date = (datetime.fromisoformat(ct.replace("Z", "+00:00"))
                       - timedelta(hours=4)).date().isoformat()
            if ev_date != d.isoformat():
                skipped += 1; continue      # belongs to another day's fetch
        else:
            ev_date = d.isoformat()
        key = odds_key(ev_date, ev.get("away_team"), ev.get("home_team"))
        if key in archive:
            skipped += 1; continue          # already archived: don't re-spend
        params = {"apiKey": api_key, "regions": ODDS_REGIONS,
                  "markets": ODDS_MARKET_KEY, "oddsFormat": "american"}
        if historical:
            path = (f"/historical/sports/baseball_mlb/events/"
                    f"{ev['id']}/odds")
            params["date"] = f"{d.isoformat()}T16:00:00Z"
            payload = _odds_http(path, params).get("data", {})
        else:
            payload = _odds_http(
                f"/sports/baseball_mlb/events/{ev['id']}/odds", params)
        mkt = _extract_f5_market(payload)
        if not mkt:
            continue
        archive[key] = {"date": ev_date,
                        "home": ev.get("home_team"),
                        "away": ev.get("away_team"), **mkt}
        got += 1
    save_odds_archive(season, archive)
    rem = getattr(_odds_http, "remaining", "?")
    note = f", {skipped} already archived" if skipped else ""
    print(f"  {d}: stored F5 lines for {got}/{len(events) - skipped} "
          f"games{note} -> {odds_archive_file(season)} "
          f"(API credits left: {rem})")
    return got


def import_odds_csv(path):
    """Import historical odds from CSV: date,away,home,away_ml,home_ml
    (date as YYYY-MM-DD). Appends to the per-season archives."""
    import csv
    counts = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            d_iso = row["date"].strip()
            season = int(d_iso[:4])
            archive = counts.setdefault(season, load_odds_archive(season))
            archive[odds_key(d_iso, row["away"], row["home"])] = {
                "date": d_iso, "home": row["home"].strip(),
                "away": row["away"].strip(),
                "home_ml": float(row["home_ml"]),
                "away_ml": float(row["away_ml"]), "book": "import"}
    for season, archive in counts.items():
        save_odds_archive(season, archive)
        print(f"  season {season}: archive now {len(archive)} games "
              f"-> {odds_archive_file(season)}")


# --------------------------------------------------------------------------
# SLATE SCORING
# --------------------------------------------------------------------------
def get_lineup(game_pk, side, team_id, slate_date):
    """Return (list_of_9_batter_ids, 'CONFIRMED'|'PROJECTED'|None)."""
    box = api_get("game_boxscore", {"gamePk": game_pk})
    order = box.get("teams", {}).get(side, {}).get("battingOrder", [])
    if len(order) >= 9:
        return order[:9], "CONFIRMED"
    # Fallback: most recent completed game's lineup for this team.
    lookback_start = slate_date - timedelta(days=10)
    sched = statsapi.schedule(
        team=team_id,
        start_date=mlb_date(lookback_start),
        end_date=mlb_date(slate_date - timedelta(days=1)),
    )
    time.sleep(API_DELAY)
    finals = [g for g in sched if g.get("status") == "Final"]
    if not finals:
        return None, None
    finals.sort(key=lambda g: g["game_date"], reverse=True)
    gs = get_game_summary(finals[0]["game_id"])
    prev_side = "home" if gs["home"]["team_id"] == team_id else "away"
    prev_order = gs[prev_side].get("batting_order", [])
    if len(prev_order) >= 9:
        return prev_order[:9], "PROJECTED"
    return None, None


def batter_form(person, snap, multiplier=1.0, slash_weights=None):
    """One batter -> (form 0-100 or None, flagged bool, detail dict)."""
    name = person.get("fullName", "?")
    splits = []
    for s in person.get("stats", []):
        splits.extend(s.get("splits", []))
    st = splits[0]["stat"] if splits else {}
    pa = st.get("plateAppearances", 0)
    if pa < MIN_BATTER_PA:
        return None, True, {"name": name, "pa": pa}
    pcts = {k: percentile(snap["batting"][k], safe_float(st.get(k)))
            for k in ("avg", "obp", "slg")}
    if any(v is None for v in pcts.values()):
        return None, True, {"name": name, "pa": pa}
    sw = slash_weights or SLASH_WEIGHTS
    form = clamp(sum(sw[k] * pcts[k] for k in sw) * multiplier, 0.0, 100.0)
    # per-stat components with the platoon multiplier applied (for sweeps;
    # the rare >100 clamp is ignored in components — negligible)
    comps = {k: pcts[k] * multiplier for k in pcts}
    return form, False, {"name": name, "pa": pa, "form": round(form, 1),
                         "platoon_mult": round(multiplier, 3),
                         "_comps": comps}


def fetch_batter_range_stats(lineup_ids, window_start, window_end,
                             cache_key=None):
    if cache_key and BATTING_WINDOW_DAYS != 14:
        cache_key = f"{cache_key}_w{BATTING_WINDOW_DAYS}"  # window-safe cache
    if cache_key:
        BATSTATS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f = BATSTATS_CACHE_DIR / f"{cache_key}.json"
        if f.exists():
            return json.loads(f.read_text())
    hydrate = (f"stats(group=[hitting],type=[byDateRange],"
               f"startDate={mlb_date(window_start)},"
               f"endDate={mlb_date(window_end)},sportId=1)")
    resp = api_get("people", {
        "personIds": ",".join(str(i) for i in lineup_ids), "hydrate": hydrate})
    if cache_key:
        f.write_text(json.dumps(resp))
    return resp


def team_batting_score(lineup_ids, team_id, snap, window_start, window_end,
                       opp_hand=None, split_caches=(), cache_key=None,
                       hitter_weight=None, slash_weights=None):
    """Returns a dict with the combined score AND its raw components, so
    weight sweeps can recombine without refetching anything."""
    hw = HITTER_FORM_WEIGHT if hitter_weight is None else hitter_weight
    resp = fetch_batter_range_stats(lineup_ids, window_start, window_end,
                                    cache_key)
    forms, flagged, batters, comp_lists = [], [], [], {"avg": [], "obp": [],
                                                       "slg": []}
    for person in resp.get("people", []):
        mult = platoon_multiplier(person.get("id"), opp_hand, split_caches)
        form, is_flagged, detail = batter_form(person, snap, mult,
                                               slash_weights)
        comps = detail.pop("_comps", None)
        batters.append(detail)
        if is_flagged:
            flagged.append(detail["name"])
        else:
            forms.append(form)
            for k in comp_lists:
                comp_lists[k].append(comps[k])
    hitter_form = mean(forms)
    hf_comps = {k: mean(v) for k, v in comp_lists.items()}

    rec = snap["team_runs"].get(str(team_id), {})
    rpg = rec.get("rpg")
    runs_pct = percentile(snap["rpg_sorted"], rpg)

    out = {"score": None, "hitter_form": hitter_form, "runs_pct": runs_pct,
           "hf_comps": hf_comps,
           "flagged": flagged, "batters": batters, "rpg": rpg}
    if hitter_form is not None and runs_pct is not None:
        out["score"] = round(hw * hitter_form + (1 - hw) * runs_pct, 1)
    return out


def starter_score(pitcher_id, slate_date, snap, weights=None):
    """Return (score, metrics, reason_if_unavailable)."""
    hydrate = f"stats(group=[pitching],type=[gameLog],season={slate_date.year})"
    resp = api_get("people", {"personIds": pitcher_id, "hydrate": hydrate})
    people = resp.get("people", [])
    if not people:
        return None, None, None, "pitcher not found"
    hand = people[0].get("pitchHand", {}).get("code")
    splits = []
    for s in people[0].get("stats", []):
        splits.extend(s.get("splits", []))
    cutoff = slate_date - timedelta(days=PITCHER_FLAG_WINDOW_DAYS)
    starts = [
        s for s in splits
        if s.get("stat", {}).get("gamesStarted", 0) >= 1
        and s.get("date", "") < slate_date.isoformat()
    ]
    starts.sort(key=lambda s: s.get("date", ""), reverse=True)
    recent = [s for s in starts if s.get("date", "") >= cutoff.isoformat()]
    if len(recent) < PITCHER_LAST_N_STARTS:
        return None, None, hand, (f"only {len(recent)} start(s) in last "
                                  f"{PITCHER_FLAG_WINDOW_DAYS} days")

    per_start = []
    for s in recent[:PITCHER_LAST_N_STARTS]:
        pk = s.get("game", {}).get("gamePk")
        gs = get_game_summary(pk) if pk else None
        entry = None
        if gs:
            for side in ("home", "away"):
                st = gs[side].get("starter")
                if st and st["id"] == pitcher_id:
                    entry = st
                    break
        if entry is None:
            # Fallback to the game log line (runs allowed stands in for runs-thru-5).
            st = s.get("stat", {})
            entry = {
                "outs": ip_to_outs(st.get("inningsPitched", "0.0")),
                "hits": st.get("hits", 0), "bb": st.get("baseOnBalls", 0),
                "k": st.get("strikeOuts", 0), "bf": st.get("battersFaced", 0),
                "runs_thru_5": st.get("runs", 0),
            }
        per_start.append(entry)

    m = pitcher_metrics(per_start)
    if m is None:
        return None, None, hand, "no batters faced in window"
    pcts = pitcher_pcts(m, snap)
    m["pcts"] = pcts
    score = combine_pitcher(pcts, weights)
    return round(score, 1), m, hand, None


def pitcher_pcts(metrics, snap):
    return {k: percentile(snap["pitching"][k], metrics[k],
                          higher_is_better=k not in PITCHER_LOWER_IS_BETTER)
            for k in PITCHER_WEIGHTS}


def combine_pitcher(pcts, weights=None):
    w = weights or PITCHER_WEIGHTS
    return sum(w[k] * pcts[k] for k in w)


def score_slate(slate_date: date, bets_only=False):
    snap = load_snapshot(slate_date)
    params = load_model_params()
    hitter_w = (params or {}).get("hitter_form_weight", HITTER_FORM_WEIGHT)
    pitcher_w = (params or {}).get("pitcher_weights", PITCHER_WEIGHTS)
    slash_w = (params or {}).get("slash_weights", SLASH_WEIGHTS)
    global BATTING_WINDOW_DAYS
    BATTING_WINDOW_DAYS = (params or {}).get("batting_window_days",
                                             BATTING_WINDOW_DAYS)
    window_end = slate_date - timedelta(days=1)
    window_start = slate_date - timedelta(days=BATTING_WINDOW_DAYS)

    sched = api_get("schedule", {
        "sportId": 1, "date": mlb_date(slate_date), "hydrate": "probablePitcher"})
    dates = sched.get("dates", [])
    games = dates[0].get("games", []) if dates else []
    if not games:
        print(f"No MLB games on {slate_date}.")
        return []

    odds_arch = load_odds_archive(slate_date.year)
    results = []
    for g in games:
        pk = g["gamePk"]
        detailed = g.get("status", {}).get("detailedState", "")
        away = g["teams"]["away"]; home = g["teams"]["home"]
        entry = {
            "gamePk": pk,
            "matchup": f'{away["team"]["name"]} @ {home["team"]["name"]}',
            "game_time": g.get("gameDate"),      # ISO UTC first pitch
        }
        if "Postponed" in detailed or "Cancelled" in detailed:
            entry["postponed"] = detailed
            results.append(entry); continue

        # -- pass 1: pitchers (scores + throwing hands) --
        nodes = {"away": away, "home": home}
        pitchers = {}
        for side in ("away", "home"):
            prob = nodes[side].get("probablePitcher") or {}
            p = {"name": prob.get("fullName"), "score": None,
                 "metrics": None, "hand": None}
            if prob.get("id"):
                score, metrics, hand, _reason = starter_score(
                    prob["id"], slate_date, snap, weights=pitcher_w)
                p.update({"score": score, "metrics": metrics, "hand": hand})
            pitchers[side] = p

        # -- pass 2: lineups vs the OPPOSING starter's hand --
        all_lineups = {}
        for side in ("away", "home"):
            team_id = nodes[side]["team"]["id"]
            lineup, lineup_status = get_lineup(pk, side, team_id, slate_date)
            all_lineups[side] = (lineup, lineup_status)
        lineup_ids = [i for lu, _ in all_lineups.values() if lu for i in lu]
        split_caches = []
        if lineup_ids:
            split_caches = [
                ensure_platoon_splits(lineup_ids, slate_date.year),
                ensure_platoon_splits(lineup_ids, slate_date.year - 1),
            ]

        qualified = 0
        lineup_batters = 0
        for side in ("away", "home"):
            opp = "home" if side == "away" else "away"
            p = pitchers[side]
            side_out = {
                "team": nodes[side]["team"]["name"],
                "batting_score": None,
                "pitcher": p["name"], "pitcher_score": p["score"],
                "pitcher_metrics": p["metrics"], "pitcher_hand": p["hand"],
                "rpg_14d": None, "lineup": "NONE", "batters": [],
            }
            lineup, lineup_status = all_lineups[side]
            if lineup:
                side_out["lineup"] = lineup_status
                bat = team_batting_score(
                    lineup, nodes[side]["team"]["id"], snap,
                    window_start, window_end,
                    opp_hand=pitchers[opp]["hand"],
                    split_caches=split_caches, hitter_weight=hitter_w,
                    slash_weights=slash_w)
                lineup_batters += len(bat["batters"])
                qualified += len(bat["batters"]) - len(bat["flagged"])
                side_out.update({"batting_score": bat["score"],
                                 "rpg_14d": bat["rpg"],
                                 "batters": bat["batters"]})
            else:
                lineup_batters += 9
            entry[side] = side_out

        entry["batters_available"] = {
            "qualified": qualified,
            "total": lineup_batters,
            "pct": round(100.0 * qualified / lineup_batters, 1)
                   if lineup_batters else None,
        }
        okey = odds_key(slate_date.isoformat(),
                        away["team"]["name"], home["team"]["name"])
        entry["market"] = odds_arch.get(okey)
        entry["prediction"] = predict_game(params, entry,
                                           market=entry["market"])
        results.append(entry)

    print_slate(results, slate_date, bets_only=bets_only)
    out_file = Path(f"./f5_scores_{slate_date.isoformat()}.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nFull detail written to {out_file}")
    n_logged = log_forward(results, slate_date)
    if n_logged:
        print(f"Forward test: {n_logged} verdict(s) recorded -> "
              f"{FORWARD_LOG} (grade with `track`)")
    return results


def load_model_params():
    if MODEL_PARAMS_FILE.exists():
        return json.loads(MODEL_PARAMS_FILE.read_text())
    return None


def predict_game(params, entry, market=None):
    """P(home leads after 5) from the fitted logistic. With market odds,
    the verdict is VALUE-based (model P vs de-vigged market P); without,
    it falls back to the backtest-tuned probability band."""
    if not params:
        return None
    a, h = entry.get("away", {}), entry.get("home", {})
    vals = (a.get("batting_score"), h.get("batting_score"),
            a.get("pitcher_score"), h.get("pitcher_score"))
    if any(v is None for v in vals):
        return None
    bat_diff = h["batting_score"] - a["batting_score"]
    pit_diff = h["pitcher_score"] - a["pitcher_score"]
    s = params.get("scale", 10.0)
    p_home = _sigmoid(params["b0"] + params["b1"] * bat_diff / s
                      + params["b2"] * pit_diff / s)
    out = {"p_home": round(p_home, 3)}

    if market:
        mkt_p_home = devig(market["home_ml"], market["away_ml"])
        value = p_home - mkt_p_home          # >0 -> home is the value side
        side = "home" if value > 0 else "away"
        v = abs(value)
        out.update({"mkt_p_home": round(mkt_p_home, 3),
                    "value": round(value, 3), "mode": "market"})
        edge_min = params.get("market_edge_min", MARKET_EDGE_MIN)
        if v < edge_min:
            out.update({"pick": None, "tier": "NO VALUE"})
        else:
            out.update({
                "pick": h["team"] if side == "home" else a["team"],
                "pick_ml": market["home_ml"] if side == "home"
                           else market["away_ml"],
                "tier": "STRONG" if v >= MARKET_EDGE_STRONG else "LEAN"})
        return out

    edge = abs(p_home - 0.5)
    out.update({"edge": round(edge, 3), "mode": "band"})
    if edge < params["no_bet_edge"]:
        out.update({"pick": None, "tier": "NO BET"})
    else:
        out.update({"pick": h["team"] if p_home >= 0.5 else a["team"],
                    "tier": "STRONG" if edge >= params["strong_edge"]
                            else "LEAN"})
    return out


def print_slate(results, slate_date, bets_only=False):
    rule = "\u2500" * 70
    if bets_only:
        bets = [r for r in results
                if r.get("prediction") and r["prediction"].get("pick")]
        bets.sort(key=lambda r: abs(r["prediction"].get("value") or
                                    r["prediction"].get("edge") or 0),
                  reverse=True)
        print(f"\nF5 Bets \u2014 {slate_date}  "
              f"({len(bets)} qualifying of {len(results)} games)")
        if not bets:
            print(rule)
            print("  No games clear the value threshold today. "
                  "The correct number of bets is sometimes zero.")
            print(rule)
            return
        results = bets
    else:
        print(f"\nF5 Streakiness Scores \u2014 {slate_date}")
    for r in results:
        print(rule)
        print(r["matchup"])
        if r.get("postponed"):
            print(f'  {r["postponed"]}')
            continue
        ba = r["batters_available"]
        line = (f'  Batters available: {ba["qualified"]}/{ba["total"]}'
                f' ({ba["pct"]:.0f}%)' if ba["pct"] is not None
                else "  Batters available: --")
        proj = [s for s in ("away", "home") if r[s].get("lineup") == "PROJECTED"]
        if proj:
            line += "  \u00b7  " + ", ".join(f"{s} lineup projected" for s in proj)
        print(line)
        print(f'  {"TEAM":<30}{"BAT":>7}{"PIT":>7}   STARTER')
        for side in ("away", "home"):
            s = r[side]
            def fmt(v):
                return f"{v:>7.1f}" if isinstance(v, (int, float)) else f'{"--":>7}'
            hand = f' ({s["pitcher_hand"]}HP)' if s.get("pitcher_hand") else ""
            print(f'  {s["team"]:<30}{fmt(s["batting_score"])}'
                  f'{fmt(s["pitcher_score"])}   {(s.get("pitcher") or "--") + hand}')
        pred = r.get("prediction")
        mkt = r.get("market")
        if mkt:
            def ml(v):
                return f"+{v:.0f}" if v > 0 else f"{v:.0f}"
            print(f'  MARKET: {ml(mkt["away_ml"])} / {ml(mkt["home_ml"])} '
                  f'home  ({mkt["book"]})')
        if pred:
            p_pick = max(pred["p_home"], 1 - pred["p_home"])
            if pred.get("mode") == "market":
                mp = pred["mkt_p_home"]
                if pred["tier"] == "NO VALUE":
                    print(f'  VERDICT: NO VALUE  (model {pred["p_home"]:.0%} '
                          f'vs mkt {mp:.0%} home)')
                else:
                    side_p = (pred["p_home"] if pred["pick"] == r["home"]["team"]
                              else 1 - pred["p_home"])
                    side_m = mp if pred["pick"] == r["home"]["team"] else 1 - mp
                    print(f'  VERDICT: {pred["pick"]} {ml(pred["pick_ml"])}  ·  '
                          f'model {side_p:.0%} vs mkt {side_m:.0%} '
                          f'(+{abs(pred["value"]):.1%})  ·  {pred["tier"]}')
            elif pred["tier"] == "NO BET":
                print(f'  VERDICT: NO BET  (P={p_pick:.0%} — inside no-bet band)')
            else:
                print(f'  VERDICT: {pred["pick"]}  ·  P={p_pick:.0%}  ·  {pred["tier"]}')
    print(rule)




# --------------------------------------------------------------------------
# BACKTEST (v2) — reconstruct history, fit weights, tune the no-bet band
# --------------------------------------------------------------------------
def league_batter_distribution(bat_start, end):
    """League 14-day slash distributions, disk-cached per window."""
    LEAGUE_BAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = LEAGUE_BAT_CACHE_DIR / f"{bat_start.isoformat()}_{end.isoformat()}.json"
    if f.exists():
        return json.loads(f.read_text())
    avgs, obps, slgs = [], [], []
    offset, page = 0, 500
    while True:
        resp = api_get("stats", {
            "stats": "byDateRange", "group": "hitting",
            "startDate": mlb_date(bat_start), "endDate": mlb_date(end),
            "sportIds": 1, "gameType": "R", "playerPool": "ALL",
            "limit": page, "offset": offset,
        })
        splits = (resp.get("stats") or [{}])[0].get("splits", [])
        for s in splits:
            st = s.get("stat", {})
            if st.get("plateAppearances", 0) >= MIN_BATTER_PA:
                a, o, g = (safe_float(st.get("avg")), safe_float(st.get("obp")),
                           safe_float(st.get("slg")))
                if None not in (a, o, g):
                    avgs.append(a); obps.append(o); slgs.append(g)
        if len(splits) < page:
            break
        offset += page
    dist = {"avg": sorted(avgs), "obp": sorted(obps), "slg": sorted(slgs)}
    f.write_text(json.dumps(dist))
    return dist


def snapshot_for_date(d, finals, starts_index):
    """As-of snapshot for date d, built from cached season data."""
    end = d - timedelta(days=1)
    bat_start = d - timedelta(days=BATTING_WINDOW_DAYS)
    pit_start = d - timedelta(days=PITCHER_FLAG_WINDOW_DAYS)

    team_runs = {}
    for g in finals:
        if not (bat_start.isoformat() <= g["game_date"] <= end.isoformat()):
            continue
        for tid, name, runs in (
            (g["home_id"], g["home_name"], g.get("home_score", 0)),
            (g["away_id"], g["away_name"], g.get("away_score", 0)),
        ):
            rec = team_runs.setdefault(tid, {"name": name, "runs": 0, "games": 0})
            rec["runs"] += runs or 0; rec["games"] += 1
    for rec in team_runs.values():
        rec["rpg"] = round(rec["runs"] / rec["games"], 3) if rec["games"] else None
    rpg_sorted = sorted(r["rpg"] for r in team_runs.values()
                        if r["rpg"] is not None)

    dist = {k: [] for k in PITCHER_WEIGHTS}
    for pid, starts in starts_index.items():
        recent = [s for s in starts
                  if pit_start.isoformat() <= s["date"] <= end.isoformat()]
        if len(recent) < PITCHER_LAST_N_STARTS:
            continue
        recent.sort(key=lambda s: s["date"], reverse=True)
        m = pitcher_metrics(recent[:PITCHER_LAST_N_STARTS])
        if m:
            for k in dist:
                dist[k].append(m[k])
    for k in dist:
        dist[k].sort()

    return {"batting": league_batter_distribution(bat_start, end),
            "team_runs": {str(t): r for t, r in team_runs.items()},
            "rpg_sorted": rpg_sorted, "pitching": dist}


def backtest_starter_score(pid, d, starts_index, snap):
    pit_start = d - timedelta(days=PITCHER_FLAG_WINDOW_DAYS)
    recent = [s for s in starts_index.get(pid, [])
              if pit_start.isoformat() <= s["date"] < d.isoformat()]
    if len(recent) < PITCHER_LAST_N_STARTS:
        return None
    recent.sort(key=lambda s: s["date"], reverse=True)
    m = pitcher_metrics(recent[:PITCHER_LAST_N_STARTS])
    if m is None:
        return None
    return pitcher_pcts(m, snap)      # caller combines with its weights


def backtest(end_date, start_date=None, season=None):
    if season is None or season == end_date.year:
        season = end_date.year
        sched_end = end_date - timedelta(days=1)
    else:                                   # completed past season
        sched_end = date(season, 11, 10)
    if start_date is None or start_date.year != season:
        mm, dd = BACKTEST_DEFAULT_START.split("/")
        start_date = date(season, int(mm), int(dd))

    print("  [1/5] season schedule ...")
    season_open = date(season, 3, 15)
    sched = statsapi.schedule(start_date=mlb_date(season_open),
                              end_date=mlb_date(sched_end))
    time.sleep(API_DELAY)
    finals = [g for g in sched if g.get("status") == "Final"
              and g.get("game_type", "R") == "R"]   # regular season only
    end = min(sched_end,
              date.fromisoformat(max(g["game_date"] for g in finals))) \
        if finals else sched_end
    print(f"Backtest season {season}: {start_date} .. {end}")
    print(f"        {len(finals)} final games")

    print("  [2/5] game feeds (cached after first run) ...")
    summaries = {}
    for i, g in enumerate(finals, 1):
        if i % 100 == 0:
            print(f"        ... {i}/{len(finals)}")
        summaries[g["game_id"]] = get_game_summary(g["game_id"])

    starts_index = {}
    all_batter_ids = set()
    for gs in summaries.values():
        for side in ("home", "away"):
            st = gs[side].get("starter")
            if st:
                starts_index.setdefault(st["id"], []).append(
                    {**st, "date": gs["date"]})
            all_batter_ids.update(gs[side].get("batting_order", [])[:9])

    print(f"  [3/5] prior-season ({season - 1}) platoon splits for "
          f"{len(all_batter_ids)} batters (no-leakage rule) ...")
    split_caches = [ensure_platoon_splits(sorted(all_batter_ids), season - 1)]

    print("  [4/5] reconstructing daily slates ...")
    rows, skipped = [], 0
    d = start_date
    while d <= end:
        todays = [g for g in finals if g["game_date"] == d.isoformat()]
        if todays:
            snap = snapshot_for_date(d, finals, starts_index)
            w_end = d - timedelta(days=1)
            w_start = d - timedelta(days=BATTING_WINDOW_DAYS)
            for g in todays:
                gs = summaries[g["game_id"]]
                if gs.get("innings_played", 0) < 5:
                    skipped += 1; continue
                sides = {}
                ok = True
                for side in ("home", "away"):
                    st = gs[side].get("starter")
                    lineup = gs[side].get("batting_order", [])[:9]
                    if not st or len(lineup) < 9:
                        ok = False; break
                    pit_pcts = backtest_starter_score(
                        st["id"], d, starts_index, snap)
                    if pit_pcts is None:
                        ok = False; break
                    sides[side] = {"starter": st, "lineup": lineup,
                                   "pit_pcts": pit_pcts,
                                   "pit": round(combine_pitcher(pit_pcts), 1)}
                if not ok:
                    skipped += 1; continue
                for side in ("home", "away"):
                    opp = "away" if side == "home" else "home"
                    bat = team_batting_score(
                        sides[side]["lineup"], gs[side]["team_id"], snap,
                        w_start, w_end,
                        opp_hand=sides[opp]["starter"].get("hand"),
                        split_caches=split_caches,
                        cache_key=f'{g["game_id"]}_{side}')
                    if bat["score"] is None:
                        ok = False; break
                    sides[side].update({
                        "bat": bat["score"],
                        "hitter_form": bat["hitter_form"],
                        "hf_comps": bat["hf_comps"],
                        "runs_pct": bat["runs_pct"]})
                if not ok:
                    skipped += 1; continue
                hr, ar = gs["f5_runs"]["home"], gs["f5_runs"]["away"]
                rows.append({
                    "date": d.isoformat(), "gamePk": g["game_id"],
                    "bat_diff": sides["home"]["bat"] - sides["away"]["bat"],
                    "pit_diff": sides["home"]["pit"] - sides["away"]["pit"],
                    "home_f5": hr, "away_f5": ar,
                    "outcome": "home" if hr > ar else
                               "away" if ar > hr else "tie",
                    "features": {
                        s: {"hitter_form": sides[s]["hitter_form"],
                            "hf_comps": sides[s]["hf_comps"],
                            "runs_pct": sides[s]["runs_pct"],
                            "pit_pcts": sides[s]["pit_pcts"]}
                        for s in ("home", "away")},
                })
            print(f"        {d}: {len(todays)} games, "
                  f"{len(rows)} rows total")
        d += timedelta(days=1)
    print(f"        {len(rows)} usable games, {skipped} skipped "
          f"(short/insufficient data)")

    rows_file = rows_path(season)
    rows_file.write_text(json.dumps(rows, indent=2))
    if season != end_date.year:
        # PAST-SEASON MODE: rows only. Never fit or touch the frozen params —
        # that would contaminate the out-of-sample experiment.
        print(f"  [5/5] past-season mode: rows saved -> {rows_file}")
        print(f"        Frozen params NOT modified. "
              f"Run `evaluate --season {season}` for the out-of-sample test.")
        return None
    print("  [5/5] fitting + tuning ...")
    params, report = fit_and_tune(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print(report)
    print(f"Model params saved -> {MODEL_PARAMS_FILE}")
    print(f"Backtest rows      -> {rows_file}")
    return params


def _solve3(H, g):
    """Solve 3x3 linear system H·x = g by Gaussian elimination."""
    a = [row[:] + [g[i]] for i, row in enumerate(H)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        for r in range(3):
            if r != c and a[c][c] != 0:
                f = a[r][c] / a[c][c]
                for k in range(c, 4):
                    a[r][k] -= f * a[c][k]
    return [a[i][3] / a[i][i] if a[i][i] else 0.0 for i in range(3)]


def _sigmoid(z):
    z = clamp(z, -35.0, 35.0)
    return 1.0 / (1.0 + exp(-z))


def fit_logistic(rows, scale=10.0, l2=1e-4, iters=30):
    """Newton-Raphson logistic regression (3 params, converges in ~8 steps).
    Tiny L2 ridge keeps the Hessian invertible on small/separable samples."""
    data = [(1.0, r["bat_diff"] / scale, r["pit_diff"] / scale,
             1.0 if r["outcome"] == "home" else 0.0)
            for r in rows if r["outcome"] != "tie"]
    b = [0.0, 0.0, 0.0]
    if not data:
        return tuple(b)
    for _ in range(iters):
        g = [0.0, 0.0, 0.0]
        H = [[0.0] * 3 for _ in range(3)]
        for x0, x1, x2, y in data:
            xs = (x0, x1, x2)
            p = _sigmoid(b[0] * x0 + b[1] * x1 + b[2] * x2)
            w = max(p * (1 - p), 1e-9)
            err = p - y
            for i in range(3):
                g[i] += err * xs[i]
                for j in range(3):
                    H[i][j] += w * xs[i] * xs[j]
        for i in range(3):
            g[i] += l2 * b[i]
            H[i][i] += l2
        step = _solve3(H, g)
        for i in range(3):
            b[i] -= step[i]
        if max(abs(s) for s in step) < 1e-9:
            break
    return b[0], b[1], b[2]


def log_loss(rows, b0, b1, b2, scale=10.0):
    """Mean negative log-likelihood on decisive rows (lower = better)."""
    dec = [r for r in rows if r["outcome"] != "tie"]
    if not dec:
        return None
    total = 0.0
    for r in dec:
        p = _sigmoid(b0 + b1 * r["bat_diff"] / scale
                     + b2 * r["pit_diff"] / scale)
        p = clamp(p, 1e-9, 1 - 1e-9)
        y = 1.0 if r["outcome"] == "home" else 0.0
        total -= y * log(p) + (1 - y) * log(1 - p)
    return total / len(dec)


def fit_and_tune(rows):
    rows = sorted(rows, key=lambda r: r["date"])
    n_train = int(len(rows) * TRAIN_FRACTION)
    train, val = rows[:n_train], rows[n_train:]
    b0, b1, b2 = fit_logistic(train)
    scale = 10.0

    def prob(r):
        z = b0 + b1 * r["bat_diff"] / scale + b2 * r["pit_diff"] / scale
        return 1.0 / (1.0 + exp(-z))

    def acc(rs):
        dec = [r for r in rs if r["outcome"] != "tie"]
        if not dec:
            return None
        hit = sum(1 for r in dec
                  if (prob(r) >= 0.5) == (r["outcome"] == "home"))
        return hit / len(dec)

    # tune the no-bet band on validation: smallest edge cutoff that keeps
    # ~TARGET_BET_VOLUME of games in play
    edges = sorted((abs(prob(r) - 0.5) for r in val), reverse=True)
    k = max(1, int(len(edges) * TARGET_BET_VOLUME))
    no_bet_edge = round(edges[k - 1], 3) if edges else 0.03
    ks = max(1, int(len(edges) * STRONG_TIER_FRACTION))
    strong_edge = round(edges[ks - 1], 3) if edges else 0.10

    bets = [r for r in val if abs(prob(r) - 0.5) >= no_bet_edge]
    tie_rate_all = (sum(1 for r in val if r["outcome"] == "tie") / len(val)
                    if val else None)
    tie_rate_bets = (sum(1 for r in bets if r["outcome"] == "tie") / len(bets)
                     if bets else None)

    params = {"b0": round(b0, 4), "b1": round(b1, 4), "b2": round(b2, 4),
              "scale": scale, "no_bet_edge": no_bet_edge,
              "strong_edge": strong_edge,
              "fitted_on": f'{rows[0]["date"]}..{rows[n_train-1]["date"]}'
                           if train else "",
              "validated_on": f'{val[0]["date"]}..{val[-1]["date"]}'
                              if val else ""}

    def pct(x):
        return f"{x:.1%}" if x is not None else "--"
    report = (f"\n  games: {len(rows)} (train {len(train)} / val {len(val)})"
              f"\n  coefficients: b0={params['b0']} bat={params['b1']} "
              f"pit={params['b2']} (per {scale} score pts)"
              f"\n  accuracy (decisive games): train {pct(acc(train))} / "
              f"VALIDATION {pct(acc(val))}"
              f"\n  validation w/ no-bet band (edge>={no_bet_edge}): "
              f"{pct(acc(bets))} on {len(bets)}/{len(val)} games bet"
              f"\n  tie rate: {pct(tie_rate_all)} all / {pct(tie_rate_bets)} "
              f"on bets  ·  STRONG edge >= {strong_edge}\n")
    return params, report




# --------------------------------------------------------------------------
# WEIGHT SWEEP — recombine backtest components under many weight configs,
# pick by log-loss on a selection set, confirm on an untouched test tail.
# --------------------------------------------------------------------------
def _pitcher_weight_grid(step=0.1):
    """All non-negative 4-metric weight combos summing to 1.0."""
    n = round(1.0 / step)
    grid = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d4 = n - a - b - c
                grid.append({"runs_thru_5": a * step, "whip": b * step,
                             "k_pct": c * step, "ip_per_start": d4 * step})
    return grid


def _rescore_rows(rows, hitter_w, pitcher_w, slash_w=None):
    sw = slash_w or SLASH_WEIGHTS
    out = []
    for r in rows:
        f = r["features"]
        def bat(s):
            hf = sum(sw[k] * f[s]["hf_comps"][k] for k in sw)
            return hitter_w * hf + (1 - hitter_w) * f[s]["runs_pct"]
        def pit(s):
            return combine_pitcher(f[s]["pit_pcts"], pitcher_w)
        out.append({"date": r["date"], "outcome": r["outcome"],
                    "bat_diff": bat("home") - bat("away"),
                    "pit_diff": pit("home") - pit("away")})
    return out


def _slash_grid(step=0.05):
    n = round(1.0 / step)
    return [{"avg": a * step, "obp": b * step, "slg": (n - a - b) * step}
            for a in range(n + 1) for b in range(n + 1 - a)]


def _eval_config(tr, sel, hw, pw, sw):
    trs = _rescore_rows(tr, hw, pw, sw)
    sels = _rescore_rows(sel, hw, pw, sw)
    b = fit_logistic(trs)
    ll = log_loss(sels, *b)
    return ll, b


def sweep(end_date, apply=False):
    season = end_date.year
    rows_file = rows_path(season)
    if not rows_file.exists():
        sys.exit(f"{rows_file} not found — run `backtest` first.")
    all_rows = json.loads(rows_file.read_text())
    rows = [r for r in all_rows if "features" in r
            and "hf_comps" in r["features"]["home"]]
    if len(rows) < len(all_rows):
        sys.exit("Backtest rows predate slash-component capture — rerun "
                 "`backtest` once (fast, everything is cached), then sweep.")
    rows.sort(key=lambda r: r["date"])
    n = len(rows)
    i1, i2 = int(n * 0.60), int(n * 0.80)
    tr, sel, te = rows[:i1], rows[i1:i2], rows[i2:]
    print(f"Staged sweep over {n} games: fit {len(tr)} / "
          f"select {len(sel)} / test {len(te)}")

    hitter_grid = [round(0.4 + 0.1 * i, 1) for i in range(7)]
    pitch_grid = _pitcher_weight_grid(0.1)
    best = {"hw": HITTER_FORM_WEIGHT, "pw": dict(PITCHER_WEIGHTS),
            "sw": dict(SLASH_WEIGHTS)}
    best["ll"], _ = _eval_config(tr, sel, best["hw"], best["pw"], best["sw"])
    print(f"  baseline (current weights): select logloss {best['ll']:.4f}")

    for stage, what in ((1, "blend+pitcher"), (2, "slash"),
                        (3, "blend+pitcher"), (4, "slash")):
        improved = False
        if what == "blend+pitcher":
            grid = [(hw, pw, best["sw"]) for hw in hitter_grid
                    for pw in pitch_grid]
        else:
            grid = [(best["hw"], best["pw"], sw)
                    for sw in _slash_grid(0.05)]
        for idx, (hw, pw, sw) in enumerate(grid, 1):
            if idx % 1000 == 0:
                print(f"    stage {stage} ({what}): {idx}/{len(grid)}")
            ll, _ = _eval_config(tr, sel, hw, pw, sw)
            if ll is not None and ll < best["ll"] - 1e-6:
                best.update({"hw": hw, "pw": dict(pw), "sw": dict(sw),
                             "ll": ll})
                improved = True
        print(f"  stage {stage} ({what}) done -> logloss {best['ll']:.4f}  "
              f"hitter={best['hw']}  slash={best['sw']}")
        if stage >= 3 and not improved:
            break

    # ---- winner only, on the untouched test tail ----
    tr_sel = _rescore_rows(tr + sel, best["hw"], best["pw"], best["sw"])
    te_r = _rescore_rows(te, best["hw"], best["pw"], best["sw"])
    b0, b1, b2 = fit_logistic(tr_sel)
    te_ll = log_loss(te_r, b0, b1, b2)
    dec = [r for r in te_r if r["outcome"] != "tie"]
    te_acc = (sum(1 for r in dec if
                  (_sigmoid(b0 + b1 * r["bat_diff"] / 10
                            + b2 * r["pit_diff"] / 10) >= 0.5)
                  == (r["outcome"] == "home")) / len(dec)) if dec else None
    edges = sorted((abs(_sigmoid(b0 + b1 * r["bat_diff"] / 10
                                 + b2 * r["pit_diff"] / 10) - 0.5)
                    for r in te_r), reverse=True)
    k = max(1, int(len(edges) * TARGET_BET_VOLUME))
    no_bet_edge = round(edges[k - 1], 3)
    ks = max(1, int(len(edges) * STRONG_TIER_FRACTION))
    strong_edge = round(edges[ks - 1], 3)

    def pct(x):
        return f"{x:.1%}" if x is not None else "--"
    print(f"\n  WINNER on untouched TEST tail ({len(te_r)} games):"
          f"\n    hitter_form_weight={best['hw']}"
          f"\n    slash_weights={best['sw']}"
          f"\n    pitcher_weights={best['pw']}"
          f"\n    test log-loss {te_ll:.4f} · accuracy {pct(te_acc)}\n")

    result = {"hitter_form_weight": best["hw"],
              "slash_weights": best["sw"],
              "pitcher_weights": best["pw"],
              "b0": round(b0, 4), "b1": round(b1, 4), "b2": round(b2, 4),
              "scale": 10.0,
              "batting_window_days": BATTING_WINDOW_DAYS,
              "no_bet_edge": no_bet_edge,
              "strong_edge": strong_edge,
              "select_logloss": round(best["ll"], 4),
              "test_logloss": round(te_ll, 4) if te_ll else None,
              "fitted_on": "2026 (sweep)"}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "sweep_best.json").write_text(json.dumps(result, indent=2))
    print(f"  Best config saved -> {CACHE_DIR / 'sweep_best.json'}")
    if apply:
        existing = load_model_params() or {}
        keep = {k: v for k, v in existing.items()
                if k in ("market_edge_min",)}
        result.update(keep)
        MODEL_PARAMS_FILE.write_text(json.dumps(result, indent=2))
        print(f"  APPLIED -> {MODEL_PARAMS_FILE}")
    else:
        print("  Rerun with --apply to make `score` use the winner.")
    return result


# --------------------------------------------------------------------------
# MARKET BACKTEST — ROI by value threshold against archived F5 odds
# --------------------------------------------------------------------------
def _american_profit(ml, won):
    """Units won per 1-unit stake (loss = -1, push handled by caller)."""
    if not won:
        return -1.0
    return (ml / 100.0) if ml > 0 else (100.0 / -ml)


def market_report(end_date):
    season = end_date.year
    rows_file = rows_path(season)
    if not rows_file.exists():
        sys.exit(f"{rows_file} not found — run `backtest` first.")
    rows = json.loads(rows_file.read_text())
    params = load_model_params()
    if not params:
        sys.exit("No model params — run `backtest` (and optionally "
                 "`sweep --apply`) first.")
    archive = load_odds_archive(season)
    if not archive:
        sys.exit(f"No odds archive ({odds_archive_file(season)}). "
                 "Use `fetch-odds` daily, `fetch-odds --historical` (paid "
                 "plan), or `import-odds file.csv`.")

    hw = params.get("hitter_form_weight", HITTER_FORM_WEIGHT)
    pw = params.get("pitcher_weights", PITCHER_WEIGHTS)
    s = params.get("scale", 10.0)

    joined = []
    for r in rows:
        gs = get_game_summary(r["gamePk"])
        rec = archive.get(odds_key(
            gs["date"], gs["away"]["team_name"], gs["home"]["team_name"]))
        if not rec:
            continue
        p_home = row_prob(r, params)
        joined.append({
            "date": r["date"], "outcome": r["outcome"],
            "p_home": p_home,
            "mkt_p_home": devig(rec["home_ml"], rec["away_ml"]),
            "home_ml": rec["home_ml"], "away_ml": rec["away_ml"]})
    if not joined:
        sys.exit("No overlap between backtest games and the odds archive.")
    joined.sort(key=lambda j: j["date"])
    n = len(joined)
    print(f"Market backtest: {n} games matched to odds "
          f"({joined[0]['date']} .. {joined[-1]['date']})\n")

    def simulate(games, thresh):
        bets = wins = pushes = 0
        units = 0.0
        for j in games:
            value = j["p_home"] - j["mkt_p_home"]
            if abs(value) < thresh:
                continue
            side = "home" if value > 0 else "away"
            ml = j["home_ml"] if side == "home" else j["away_ml"]
            bets += 1
            if j["outcome"] == "tie":
                pushes += 1                      # F5 ML pushes on tie
                continue
            won = j["outcome"] == side
            wins += won
            units += _american_profit(ml, won)
        dec = bets - pushes
        return {"thresh": thresh, "bets": bets, "pushes": pushes,
                "hit": wins / dec if dec else None,
                "roi": units / bets if bets else None,
                "units": round(units, 2)}

    n_tune = int(n * 0.80)
    tune, hold = joined[:n_tune], joined[n_tune:]
    print(f'  {"thresh":>7} {"bets":>5} {"hit%":>6} {"units":>7} {"ROI/bet":>8}'
          f'   (tuning set, first {len(tune)} games)')
    best_t, best_roi = MARKET_EDGE_MIN, None
    for t in [i / 100 for i in range(0, 11)]:
        r = simulate(tune, t)
        if r["bets"] >= max(10, len(tune) // 20) and r["roi"] is not None \
                and (best_roi is None or r["roi"] > best_roi):
            best_t, best_roi = t, r["roi"]
        hit = f'{r["hit"]:.1%}' if r["hit"] is not None else "--"
        roi = f'{r["roi"]:+.1%}' if r["roi"] is not None else "--"
        print(f'  {t:>7.2f} {r["bets"]:>5} {hit:>6} {r["units"]:>7} {roi:>8}')

    h = simulate(hold, best_t)
    hit = f'{h["hit"]:.1%}' if h["hit"] is not None else "--"
    roi = f'{h["roi"]:+.1%}' if h["roi"] is not None else "--"
    print(f"\n  Best tuning threshold: {best_t:.2f} "
          f"(ROI {best_roi:+.1%} on tuning set)"
          f"\n  HOLDOUT (last {len(hold)} games): {h['bets']} bets, "
          f"hit {hit}, {h['units']} units, ROI {roi}"
          f"\n\n  The HOLDOUT line is the honest one. Positive ROI there on a"
          f"\n  real sample is an edge; anything else is noise or overfit.")
    params["market_edge_min"] = best_t
    MODEL_PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print(f"\n  market_edge_min={best_t} saved to model params "
          f"(live verdicts use it).")




# --------------------------------------------------------------------------
# FROZEN EVALUATION — score a season with locked params; tune NOTHING
# --------------------------------------------------------------------------
def row_prob(r, params):
    """P(home) for a backtest row under the given (frozen) params."""
    hw = params.get("hitter_form_weight", HITTER_FORM_WEIGHT)
    pw = params.get("pitcher_weights", PITCHER_WEIGHTS)
    sw = params.get("slash_weights", SLASH_WEIGHTS)
    s = params.get("scale", 10.0)
    f = r.get("features")
    if f and "hf_comps" in f["home"]:
        def bat(sd):
            hf = sum(sw[k] * f[sd]["hf_comps"][k] for k in sw)
            return hw * hf + (1 - hw) * f[sd]["runs_pct"]
        bd = bat("home") - bat("away")
        pd = (combine_pitcher(f["home"]["pit_pcts"], pw)
              - combine_pitcher(f["away"]["pit_pcts"], pw))
    else:
        bd, pd = r["bat_diff"], r["pit_diff"]
    return _sigmoid(params["b0"] + params["b1"] * bd / s
                    + params["b2"] * pd / s)


def evaluate(season):
    """Pre-registered out-of-sample test. Reads the frozen model params and
    a completed season's backtest rows + odds archive. Writes nothing."""
    params = load_model_params()
    if not params:
        sys.exit("No frozen model params — this command evaluates the "
                 "CURRENT model on another season. Fit on 2026 first.")
    rows_file = rows_path(season)
    if not rows_file.exists():
        sys.exit(f"{rows_file} not found — run "
                 f"`backtest --season {season}` first.")
    rows = json.loads(rows_file.read_text())
    print(f"FROZEN EVALUATION — season {season}, {len(rows)} games, "
          f"params untouched\n")
    pw_days = params.get("batting_window_days", 14)
    if pw_days != BATTING_WINDOW_DAYS:
        print(f"  *** WARNING: params were selected on a {pw_days}-day "
              f"batting window but these rows use "
              f"{BATTING_WINDOW_DAYS}-day features. This is a hybrid no "
              f"experiment registered — treat every number below as "
              f"exploratory, not evidence. ***\n")
    fitted_on = str(params.get("fitted_on", ""))
    if str(season) in fitted_on:
        print(f"  *** WARNING: params coefficients were fit on data that "
              f"INCLUDES season {season} (fitted_on='{fitted_on}'). This "
              f"evaluation is partially IN-SAMPLE, not a frozen exam. ***\n")

    # -- accuracy + log-loss --
    dec = [r for r in rows if r["outcome"] != "tie"]
    probs = {id(r): row_prob(r, params) for r in rows}
    hits = sum(1 for r in dec
               if (probs[id(r)] >= 0.5) == (r["outcome"] == "home"))
    ll = 0.0
    for r in dec:
        p = clamp(probs[id(r)], 1e-9, 1 - 1e-9)
        y = 1.0 if r["outcome"] == "home" else 0.0
        ll -= y * log(p) + (1 - y) * log(1 - p)
    tie_rate = 1 - len(dec) / len(rows) if rows else 0
    print(f"  decisive games: {len(dec)} (tie rate {tie_rate:.1%})")
    print(f"  accuracy {hits / len(dec):.1%} · log-loss {ll / len(dec):.4f} "
          f"(coin flip = 0.6931)\n")

    # -- calibration --
    print(f'  {"pick conf":>10} {"games":>6} {"pred":>6} {"actual":>7}')
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
    for lo, hi in buckets:
        bucket = [r for r in dec
                  if lo <= max(probs[id(r)], 1 - probs[id(r)]) < hi]
        if not bucket:
            continue
        pred = mean([max(probs[id(r)], 1 - probs[id(r)]) for r in bucket])
        act = mean([1.0 if (probs[id(r)] >= 0.5) == (r["outcome"] == "home")
                    else 0.0 for r in bucket])
        print(f'  {f"{lo:.0%}-{hi:.0%}":>10} {len(bucket):>6} '
              f'{pred:>6.1%} {act:>7.1%}')

    # -- market: frozen threshold + PRE-REGISTERED edge bands --
    archive = load_odds_archive(season)
    if not archive:
        print(f"\n  (no {season} odds archive — market section skipped)")
        return
    joined = []
    for r in rows:
        gs = get_game_summary(r["gamePk"])
        rec = archive.get(odds_key(gs["date"], gs["away"]["team_name"],
                                   gs["home"]["team_name"]))
        if rec:
            joined.append((r, probs[id(r)],
                           devig(rec["home_ml"], rec["away_ml"]),
                           rec["home_ml"], rec["away_ml"]))
    print(f"\n  matched to odds: {len(joined)} games")

    def band_report(label, lo, hi):
        bets = wins = pushes = 0
        units = 0.0
        for r, p, mp, hml, aml in joined:
            v = p - mp
            if not (lo <= abs(v) < hi):
                continue
            side = "home" if v > 0 else "away"
            ml = hml if side == "home" else aml
            bets += 1
            if r["outcome"] == "tie":
                pushes += 1; continue
            won = r["outcome"] == side
            wins += won
            units += _american_profit(ml, won)
        d2 = bets - pushes
        hit = f"{wins / d2:.1%}" if d2 else "--"
        roi = f"{units / bets:+.1%}" if bets else "--"
        print(f'  {label:>14} {bets:>5} {hit:>6} {units:>8.2f} {roi:>8}')

    t = params.get("market_edge_min", MARKET_EDGE_MIN)
    print(f'\n  {"edge band":>14} {"bets":>5} {"hit%":>6} {"units":>8} '
          f'{"ROI/bet":>8}')
    band_report(f"frozen >={t:.2f}", t, 9.9)
    print("  pre-registered hypothesis bands:")
    band_report("0.03-0.07", 0.03, 0.07)
    band_report("0.07-0.10", 0.07, 0.10)
    band_report(">=0.10", 0.10, 9.9)

    # -- betting by MODEL CONFIDENCE (pick prob), any price vs +edge filter --
    def conf_report(label, lo, hi, need_edge):
        bets = wins = pushes = 0
        units = 0.0
        for r, p, mp, hml, aml in joined:
            conf = max(p, 1 - p)
            if not (lo <= conf < hi):
                continue
            side = "home" if p >= 0.5 else "away"
            side_edge = (p - mp) if side == "home" else (mp - p)
            if need_edge and side_edge < 0.03:
                continue
            ml = hml if side == "home" else aml
            bets += 1
            if r["outcome"] == "tie":
                pushes += 1; continue
            won = r["outcome"] == side
            wins += won
            units += _american_profit(ml, won)
        d2 = bets - pushes
        hit = f"{wins / d2:.1%}" if d2 else "--"
        roi = f"{units / bets:+.1%}" if bets else "--"
        print(f'  {label:>22} {bets:>5} {hit:>6} {units:>8.2f} {roi:>8}')

    print(f'\n  {"model confidence":>22} {"bets":>5} {"hit%":>6} '
          f'{"units":>8} {"ROI/bet":>8}')
    for lo, hi, lab in ((0.50, 0.55, "50-55%"), (0.55, 0.60, "55-60%"),
                        (0.60, 1.01, "60%+")):
        conf_report(f"{lab} any price", lo, hi, False)
        conf_report(f"{lab} + edge>=3%", lo, hi, True)
    print("\n  Nothing was tuned or saved by this command.")




# --------------------------------------------------------------------------
# CROSS-SEASON FIT — find weights whose signal survives BETWEEN seasons
# --------------------------------------------------------------------------
def _season_rows(season):
    f = rows_path(season)
    if not f.exists():
        sys.exit(f"{f} not found — run `backtest --season {season}` first.")
    rows = [r for r in json.loads(f.read_text())
            if "features" in r and "hf_comps" in r["features"]["home"]]
    if not rows:
        sys.exit(f"season {season} rows lack components — rerun its backtest.")
    rows.sort(key=lambda r: r["date"])
    return rows


def _acc(rescored, b):
    dec = [r for r in rescored if r["outcome"] != "tie"]
    if not dec:
        return None
    return sum(1 for r in dec if
               (_sigmoid(b[0] + b[1] * r["bat_diff"] / 10
                         + b[2] * r["pit_diff"] / 10) >= 0.5)
               == (r["outcome"] == "home")) / len(dec)


def _cross_objective(rows_a, rows_b, hw, pw, sw):
    """Fit each season, score on the OTHER; return mean log-loss (+ detail)."""
    ra = _rescore_rows(rows_a, hw, pw, sw)
    rb = _rescore_rows(rows_b, hw, pw, sw)
    ba, bb = fit_logistic(ra), fit_logistic(rb)
    l_ab, l_ba = log_loss(rb, *ba), log_loss(ra, *bb)
    if l_ab is None or l_ba is None:
        return None, None
    return (l_ab + l_ba) / 2, {
        "a_to_b": {"ll": l_ab, "acc": _acc(rb, ba)},
        "b_to_a": {"ll": l_ba, "acc": _acc(ra, bb)}}


def crossfit(seasons=(2025, 2026), apply=False):
    sa, sb = sorted(seasons)
    rows_a, rows_b = _season_rows(sa), _season_rows(sb)
    print(f"Cross-season fit: {sa} ({len(rows_a)} games) <-> "
          f"{sb} ({len(rows_b)} games)")
    print("  objective: mean of [fit one season -> log-loss on the other]"
          "\n  coin-flip reference log-loss: 0.6931\n")

    hitter_grid = [round(0.4 + 0.1 * i, 1) for i in range(7)]
    pitch_grid = _pitcher_weight_grid(0.1)
    best = {"hw": HITTER_FORM_WEIGHT, "pw": dict(PITCHER_WEIGHTS),
            "sw": dict(SLASH_WEIGHTS)}
    best["ll"], best["detail"] = _cross_objective(
        rows_a, rows_b, best["hw"], best["pw"], best["sw"])
    print(f"  baseline (default weights): cross log-loss {best['ll']:.4f}")
    cur = load_model_params()
    if cur and "hitter_form_weight" in cur:
        ll_cur, det_cur = _cross_objective(
            rows_a, rows_b, cur["hitter_form_weight"],
            cur.get("pitcher_weights", PITCHER_WEIGHTS),
            cur.get("slash_weights", SLASH_WEIGHTS))
        print(f"  current fitted params:      cross log-loss {ll_cur:.4f}")
        if ll_cur is not None and ll_cur < best["ll"]:
            best.update({"hw": cur["hitter_form_weight"],
                         "pw": dict(cur.get("pitcher_weights",
                                            PITCHER_WEIGHTS)),
                         "sw": dict(cur.get("slash_weights", SLASH_WEIGHTS)),
                         "ll": ll_cur, "detail": det_cur})

    for stage, what in ((1, "blend+pitcher"), (2, "slash"),
                        (3, "blend+pitcher"), (4, "slash")):
        improved = False
        if what == "blend+pitcher":
            grid = [(hw, pw, best["sw"]) for hw in hitter_grid
                    for pw in pitch_grid]
        else:
            grid = [(best["hw"], best["pw"], sw) for sw in _slash_grid(0.05)]
        for idx, (hw, pw, sw) in enumerate(grid, 1):
            if idx % 1000 == 0:
                print(f"    stage {stage} ({what}): {idx}/{len(grid)}")
            ll, det = _cross_objective(rows_a, rows_b, hw, pw, sw)
            if ll is not None and ll < best["ll"] - 1e-6:
                best.update({"hw": hw, "pw": dict(pw), "sw": dict(sw),
                             "ll": ll, "detail": det})
                improved = True
        print(f"  stage {stage} ({what}) done -> cross log-loss "
              f"{best['ll']:.4f}")
        if stage >= 3 and not improved:
            break

    det = best["detail"]
    def pc(x):
        return f"{x:.1%}" if x is not None else "--"
    print(f"\n  WINNER: hitter={best['hw']}  slash={best['sw']}"
          f"\n          pitcher={best['pw']}"
          f"\n  fit {sa} -> {sb}: log-loss {det['a_to_b']['ll']:.4f}, "
          f"accuracy {pc(det['a_to_b']['acc'])}"
          f"\n  fit {sb} -> {sa}: log-loss {det['b_to_a']['ll']:.4f}, "
          f"accuracy {pc(det['b_to_a']['acc'])}")
    if best["ll"] >= 0.6929:
        print("\n  READ THIS: the best cross-season log-loss is at/above the"
              "\n  coin-flip line. NO weighting of these features carries"
              "\n  usable signal between seasons — the feature set, not the"
              "\n  weights, is the limit.")

    # pooled coefficients under the winning weights
    pooled = _rescore_rows(rows_a + rows_b, best["hw"], best["pw"],
                           best["sw"])
    b0, b1, b2 = fit_logistic(pooled)
    edges = sorted((abs(_sigmoid(b0 + b1 * r["bat_diff"] / 10
                                 + b2 * r["pit_diff"] / 10) - 0.5)
                    for r in pooled), reverse=True)
    k = max(1, int(len(edges) * TARGET_BET_VOLUME))
    ks = max(1, int(len(edges) * STRONG_TIER_FRACTION))
    result = {"hitter_form_weight": best["hw"],
              "slash_weights": best["sw"], "pitcher_weights": best["pw"],
              "b0": round(b0, 4), "b1": round(b1, 4), "b2": round(b2, 4),
              "scale": 10.0,
              "batting_window_days": BATTING_WINDOW_DAYS,
              "no_bet_edge": round(edges[k - 1], 3),
              "strong_edge": round(edges[ks - 1], 3),
              "cross_logloss": round(best["ll"], 4),
              "fitted_on": f"pooled {sa}+{sb}"}
    (CACHE_DIR / "cross_best.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Saved -> {CACHE_DIR / 'cross_best.json'}")
    if apply:
        existing = load_model_params() or {}
        for keep in ("market_edge_min",):
            if keep in existing:
                result[keep] = existing[keep]
        MODEL_PARAMS_FILE.write_text(json.dumps(result, indent=2))
        print(f"  APPLIED -> {MODEL_PARAMS_FILE}"
              f"\n  Both seasons are now burned as holdouts. The fresh exam"
              f"\n  is: backtest --season 2024, fetch its odds, then"
              f"\n  evaluate --season 2024.")
    else:
        print("  Rerun with --apply to adopt (then exam on 2024).")
    return result




# --------------------------------------------------------------------------
# FORWARD TEST (paper-trading ledger) — log frozen verdicts, grade reality
# --------------------------------------------------------------------------
def _params_hash():
    if not MODEL_PARAMS_FILE.exists():
        return None
    import hashlib
    return hashlib.md5(MODEL_PARAMS_FILE.read_bytes()).hexdigest()[:10]


def fetch_clv():
    """Backfill closing lines onto logged bets, and score CLV.

    For each bet past first pitch with no closing line yet, pull the
    historical snapshot 10 minutes before its own start time and record the
    de-vigged market probability of THE SIDE WE BET. CLV is that closing
    probability minus the one we got: positive means we bought our side
    cheaper than the market's final word.

    Costs one events call per distinct first-pitch time plus one odds call
    per bet, and only ever fetches a given bet once."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("Set the ODDS_API_KEY environment variable.")
    if not FORWARD_LOG.exists():
        print("  No forward log yet — nothing to price against the close.")
        return 0
    log = json.loads(FORWARD_LOG.read_text())
    groups = clv_pending(log)
    if not groups:
        print("  Every logged bet already has its closing line.")
        return 0

    total = sum(len(v) for v in groups.values())
    print(f"  {total} bet(s) across {len(groups)} first-pitch time(s) "
          f"need closing lines.")
    done = 0
    for snap in sorted(groups):
        recs = groups[snap]
        try:
            events = historical_events(snap, api_key)
        except SystemExit:
            raise
        except Exception as e:
            print(f"    {snap}: events lookup failed ({e}) — skipping")
            continue
        by_key = {}
        for ev in events:
            ct = ev.get("commence_time")
            ev_date = ((datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        - timedelta(hours=4)).date().isoformat()
                       if ct else None)
            by_key[odds_key(ev_date, ev.get("away_team"),
                            ev.get("home_team"))] = ev
        for key, rec in recs:
            k = odds_key(rec["date"], rec.get("away_team"),
                         rec.get("home_team"))
            ev = by_key.get(k)
            if not ev:
                print(f"    no closing event for {rec['matchup']} — skipping")
                continue
            try:
                payload = historical_event_odds(ev["id"], snap,
                                                ODDS_MARKET_KEY, api_key)
            except Exception as e:
                print(f"    odds fetch failed for {rec['matchup']}: {e}")
                continue
            mkt = _extract_f5_market(payload)
            if not mkt:
                continue
            close_p_home = devig(mkt["home_ml"], mkt["away_ml"])
            on_home = rec["pick"] == rec.get("home_team")
            close_p = close_p_home if on_home else 1 - close_p_home
            bet_p = (rec["mkt_p_home"] if on_home
                     else 1 - rec["mkt_p_home"])
            rec["close"] = {
                "snapshot": snap, "book": mkt.get("book"),
                "home_ml": mkt["home_ml"], "away_ml": mkt["away_ml"],
                "pick_ml": mkt["home_ml"] if on_home else mkt["away_ml"],
                "mkt_p": round(close_p, 4),
                "clv": round(close_p - bet_p, 4),
            }
            done += 1
    FORWARD_LOG.write_text(json.dumps(log, indent=1))
    rem = getattr(_odds_http, "remaining", "?")
    print(f"  priced {done}/{total} bet(s) against the close "
          f"(API credits left: {rem})")
    rows = [{"clv": r["close"]["clv"], "tier": r.get("tier")}
            for r in log.values() if r.get("close")]
    if rows:
        print()
        clv_summary(rows, label="CLOSING LINE VALUE")
    return done


def log_forward(results, slate_date):
    """Record today's market-mode verdicts. First log wins: re-running score
    later (with moved lines) never rewrites the bet you'd have placed."""
    log = json.loads(FORWARD_LOG.read_text()) if FORWARD_LOG.exists() else {}
    added = 0
    for r in results:
        pred, mkt = r.get("prediction"), r.get("market")
        if not pred or pred.get("mode") != "market":
            continue
        key = str(r["gamePk"])
        if key in log:
            # metadata-only backfill; never rewrites a recorded bet
            if r.get("game_time") and not log[key].get("game_time"):
                log[key]["game_time"] = r["game_time"]
                added += 1
            continue
        log[key] = {
            "gamePk": r["gamePk"], "date": slate_date.isoformat(),
            "game_time": r.get("game_time"),
            "matchup": r["matchup"], "pick": pred.get("pick"),
            "tier": pred["tier"], "p_home": pred["p_home"],
            "mkt_p_home": pred["mkt_p_home"],
            "pick_ml": pred.get("pick_ml"),
            "value": pred.get("value"),
            "home_team": r["home"]["team"], "away_team": r["away"]["team"],
            "params": _params_hash(),
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "graded": False,
        }
        added += 1
    if added:
        FORWARD_LOG.write_text(json.dumps(log, indent=1))
    return added


def track():
    if not FORWARD_LOG.exists():
        sys.exit("No forward log yet — run `score` on a slate with odds "
                 "first; verdicts are recorded automatically.")
    log = json.loads(FORWARD_LOG.read_text())
    cur_hash = _params_hash()
    hashes = {rec.get("params") for rec in log.values()}
    if len(hashes) > 1:
        print("*** WARNING: verdicts in this ledger were produced by "
              "DIFFERENT model params — the trial is not clean. ***")
    if cur_hash not in hashes and hashes:
        print("*** WARNING: current model params differ from the ones that "
              "logged these verdicts. Don't retune mid-trial. ***")

    today = date.today().isoformat()
    newly = 0
    for rec in log.values():
        if rec["graded"] or rec["date"] > today:
            continue
        try:
            gs = get_game_summary(rec["gamePk"])
        except Exception:
            continue
        if not gs.get("final"):
            continue
        if gs.get("innings_played", 0) < 5:
            rec.update({"graded": True, "outcome": "void", "units": 0.0})
            newly += 1; continue
        hr, ar = gs["f5_runs"]["home"], gs["f5_runs"]["away"]
        outcome = ("tie" if hr == ar else
                   rec["home_team"] if hr > ar else rec["away_team"])
        units = 0.0
        if rec["pick"]:
            if outcome == "tie":
                units = 0.0                      # F5 push
            else:
                units = _american_profit(rec["pick_ml"],
                                         outcome == rec["pick"])
        rec.update({"graded": True, "outcome": outcome,
                    "units": round(units, 3)})
        newly += 1
    FORWARD_LOG.write_text(json.dumps(log, indent=1))

    graded = [r for r in log.values() if r["graded"]]
    picks = [r for r in graded if r["pick"] and r["outcome"] != "void"]
    pending = [r for r in log.values() if not r["graded"]]
    print(f"FORWARD TEST — {len(log)} verdicts logged, {len(graded)} graded "
          f"({newly} just now), {len(pending)} pending\n")

    recent = sorted(graded, key=lambda r: r["date"])[-10:]
    for r in recent:
        res = ("PUSH" if r["outcome"] == "tie" else "VOID"
               if r["outcome"] == "void" else
               "WIN " if r["pick"] == r["outcome"] else "LOSS")
        pick = r["pick"] or "(no bet)"
        ml = f'{r["pick_ml"]:+.0f}' if r.get("pick_ml") else ""
        u = f'{r.get("units", 0):+.2f}' if r["pick"] else ""
        print(f'  {r["date"]}  {r["matchup"]:<42} {pick:<24}{ml:>6} '
              f'{res if r["pick"] else "":<5}{u:>7}')

    def ledger(rows, label):
        dec = [r for r in rows if r["outcome"] not in ("tie", "void")]
        wins = sum(1 for r in dec if r["pick"] == r["outcome"])
        pushes = sum(1 for r in rows if r["outcome"] == "tie")
        units = sum(r.get("units", 0) for r in rows)
        hit = f"{wins / len(dec):.1%}" if dec else "--"
        roi = f"{units / len(rows):+.1%}" if rows else "--"
        print(f'  {label:>10}: {len(rows)} bets '
              f'({wins}-{len(dec) - wins}-{pushes}), {units:+.2f} units, '
              f'ROI {roi}, hit {hit}')

    if picks:
        print()
        ledger(picks, "ALL BETS")
        for tier in ("STRONG", "LEAN"):
            rows = [r for r in picks if r["tier"] == tier]
            if rows:
                ledger(rows, tier)
        dec = [r for r in picks if r["outcome"] != "tie"]
        if dec:
            claimed = mean([max(r["p_home"], 1 - r["p_home"]) for r in dec])
            actual = sum(1 for r in dec if r["pick"] == r["outcome"]) / len(dec)
            print(f"\n  calibration in the wild: model claimed "
                  f"{claimed:.1%} on its picks, delivered {actual:.1%}")
        closed = [r for r in log.values() if r.get("close")]
        if closed:
            print()
            clv_summary([{"clv": r["close"]["clv"], "tier": r.get("tier")}
                          for r in closed], label="CLOSING LINE VALUE")

    else:
        print("\n  No graded bets yet — verdicts grade automatically once "
              "games go final.")




# --------------------------------------------------------------------------
# DAILY — the whole morning routine in one command
# --------------------------------------------------------------------------
def daily(slate_date):
    steps = [
        ("Grading yesterday's verdicts", lambda: track_safe()),
        ("Pricing settled bets against the close",
         lambda: fetch_clv()),
        ("Fetching today's F5 odds", lambda: fetch_odds_day(slate_date)),
        ("Scoring the slate (bets only)",
         lambda: score_slate(slate_date, bets_only=True)),
        ("Exporting the site data", lambda: export_site(slate_date)),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {label}\n{'=' * 70}")
        try:
            fn()
        except SystemExit as e:
            # a missing key or empty ledger shouldn't kill the whole morning
            print(f"  (skipped: {e})")
        except Exception as e:
            print(f"  (step failed: {type(e).__name__}: {e} — continuing)")
    print(f"\n{'=' * 70}\nDone. Ledger: `python f5_model.py track` any time."
          f"\n{'=' * 70}")


def track_safe():
    if not FORWARD_LOG.exists():
        print("  No verdicts logged yet — nothing to grade on day one.")
        return
    track()




# --------------------------------------------------------------------------
# SITE EXPORT — one static data.json for the public dashboard
# --------------------------------------------------------------------------
def export_site(slate_date=None):
    slate_date = slate_date or date.today()
    site = Path("./docs")          # GitHub Pages serves / or /docs only
    site.mkdir(exist_ok=True)
    log = (json.loads(FORWARD_LOG.read_text())
           if FORWARD_LOG.exists() else {})
    graded = [r for r in log.values() if r.get("graded")]
    picks = [r for r in graded if r.get("pick")
             and r.get("outcome") != "void"]
    dec = [r for r in picks if r["outcome"] != "tie"]
    wins = sum(1 for r in dec if r["pick"] == r["outcome"])
    pushes = sum(1 for r in picks if r["outcome"] == "tie")
    units = sum(r.get("units", 0) for r in picks)

    by_day = {}
    for r in sorted(picks, key=lambda r: r["date"]):
        by_day[r["date"]] = by_day.get(r["date"], 0) + r.get("units", 0)
    cum, series = 0.0, []
    for d_iso, u in sorted(by_day.items()):
        cum += u
        series.append({"date": d_iso, "units": round(cum, 2)})

    todays = [r for r in log.values()
              if r["date"] == slate_date.isoformat()]
    bets = [{"matchup": r["matchup"], "pick": r["pick"],
             "ml": r.get("pick_ml"), "tier": r["tier"],
             "time": r.get("game_time"),
             "model_p": r["p_home"] if r["pick"] == r["home_team"]
                        else round(1 - r["p_home"], 3),
             "mkt_p": r["mkt_p_home"] if r["pick"] == r["home_team"]
                      else round(1 - r["mkt_p_home"], 3),
             "value": abs(r.get("value") or 0)}
            for r in todays if r.get("pick")]
    # first pitch order; games without a listed time sink to the bottom
    bets.sort(key=lambda b: (b["time"] is None, b["time"] or "",
                             -b["value"]))
    params = load_model_params() or {}

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slate_date": slate_date.isoformat(),
        "ledger": {
            "bets": len(picks), "wins": wins,
            "losses": len(dec) - wins, "pushes": pushes,
            "units": round(units, 2),
            "roi": round(units / len(picks), 4) if picks else None,
            "hit": round(wins / len(dec), 4) if dec else None,
            "claimed": round(mean([max(r["p_home"], 1 - r["p_home"])
                                   for r in dec]), 4) if dec else None,
            "clv_bets": len([r for r in picks
                             if (r.get("close") or {}).get("clv") is not None]),
            "clv_beat": len([r for r in picks
                             if ((r.get("close") or {}).get("clv") or 0) > 0]),
            "clv_avg": (round(mean([r["close"]["clv"] for r in picks
                              if (r.get("close") or {}).get("clv") is not None]), 4)
                        if any((r.get("close") or {}).get("clv") is not None
                               for r in picks) else None)
        },
        "series": series,
        "bets_today": bets,
        "evaluated_today": len(todays),
        "model": {
            "batting_window_days": params.get("batting_window_days", 14),
            "pitcher_starts": PITCHER_LAST_N_STARTS,
            "hitter_form_weight": params.get("hitter_form_weight",
                                             HITTER_FORM_WEIGHT),
            "slash_weights": params.get("slash_weights", SLASH_WEIGHTS),
            "pitcher_weights": params.get("pitcher_weights",
                                          PITCHER_WEIGHTS),
            "min_pa": MIN_BATTER_PA,
            "edge_min": params.get("market_edge_min", MARKET_EDGE_MIN),
        },
    }
    (site / "data.json").write_text(json.dumps(data, indent=1))

    # ---- full bet log for the "past bets" page (newest first) ----
    history = []
    for r in sorted(graded,
                    key=lambda r: (r["date"], r.get("game_time") or ""),
                    reverse=True):
        if not r.get("pick"):
            continue                      # NO VALUE verdicts aren't bets
        oc = r.get("outcome")
        result = ("push" if oc == "tie" else
                  "void" if oc == "void" else
                  "win" if r["pick"] == oc else "loss")
        history.append({
            "date": r["date"], "time": r.get("game_time"),
            "matchup": r["matchup"], "pick": r["pick"],
            "ml": r.get("pick_ml"), "tier": r["tier"],
            "model_p": r["p_home"] if r["pick"] == r["home_team"]
                       else round(1 - r["p_home"], 3),
            "mkt_p": r["mkt_p_home"] if r["pick"] == r["home_team"]
                     else round(1 - r["mkt_p_home"], 3),
            "value": abs(r.get("value") or 0),
            "result": result, "units": r.get("units", 0),
            "clv": (r.get("close") or {}).get("clv"),
            "close_ml": (r.get("close") or {}).get("pick_ml"),
        })
    (site / "history.json").write_text(json.dumps({
        "generated_at": data["generated_at"],
        "algo": "First Five Hot Streak",
        "ledger": data["ledger"],
        "bets": history,
    }, indent=1))

    print(f"  docs/data.json written — {len(bets)} bet(s) today, "
          f"ledger {wins}-{len(dec) - wins}-{pushes}, "
          f"{units:+.2f} units")
    print(f"  docs/history.json written — {len(history)} graded bet(s)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_date(s):
    if not s:
        return date.today()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    sys.exit(f"Unrecognized date: {s} (use MM/DD/YYYY)")


def main():
    ap = argparse.ArgumentParser(description="F5 streakiness model (v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_snap = sub.add_parser("snapshot", help="build/refresh the league snapshot")
    p_snap.add_argument("--date", default=None)
    p_score = sub.add_parser("score", help="score a slate of games")
    p_score.add_argument("--date", default=None)
    p_score.add_argument("--bets", action="store_true",
                         help="show only games that clear the value "
                              "threshold, sorted by edge")
    p_bt = sub.add_parser("backtest",
                          help="reconstruct history, fit weights, tune bands")
    p_bt.add_argument("--date", default=None, help="as-of date (default today)")
    p_bt.add_argument("--start", default=None,
                      help="first backtest date (default May 1)")
    p_bt.add_argument("--season", type=int, default=None,
                      help="backtest a completed past season (e.g. 2025)")
    for _p in (p_bt,):
        _p.add_argument("--bat-window", type=int, default=14,
                        help="batting lookback window in days (default 14)")
    p_sw = sub.add_parser("sweep",
                          help="search score weights over backtest components")
    p_sw.add_argument("--date", default=None)
    p_sw.add_argument("--apply", action="store_true",
                      help="write the winning config to model params")
    p_fo = sub.add_parser("fetch-odds", help="fetch F5 moneylines to archive")
    p_fo.add_argument("--date", default=None)
    p_fo.add_argument("--historical", action="store_true",
                      help="use paid historical endpoint")
    p_fo.add_argument("--start", default=None,
                      help="with --historical: fetch a date range")
    p_io = sub.add_parser("import-odds", help="import odds from CSV")
    p_io.add_argument("file")
    p_mk = sub.add_parser("market",
                          help="ROI backtest against archived odds")
    p_mk.add_argument("--date", default=None)
    p_ev = sub.add_parser("evaluate",
                          help="frozen-params out-of-sample season test")
    p_ev.add_argument("--season", type=int, required=True)
    p_cf = sub.add_parser("crossfit",
                          help="find weights that generalize between seasons")
    p_cf.add_argument("--seasons", default="2025,2026")
    p_cf.add_argument("--apply", action="store_true")
    sub.add_parser("track", help="grade forward-test verdicts vs reality")
    sub.add_parser("clv", help="price settled bets against the closing line")
    p_ex = sub.add_parser("export", help="write site/data.json for the dashboard")
    p_ex.add_argument("--date", default=None)
    p_dy = sub.add_parser("daily",
                          help="one-shot morning routine: grade yesterday, "
                               "fetch odds, show today's bets")
    p_dy.add_argument("--date", default=None)
    for _p in (p_sw, p_mk, p_ev, p_cf):
        _p.add_argument("--bat-window", type=int, default=14,
                        help="batting lookback window in days (default 14)")
    args = ap.parse_args()

    global BATTING_WINDOW_DAYS
    bw = getattr(args, "bat_window", None)
    if bw:
        BATTING_WINDOW_DAYS = bw
        if bw != 14:
            print(f"[batting window: {bw} days]")

    d = parse_date(getattr(args, "date", None))
    if args.cmd == "snapshot":
        build_snapshot(d)
    elif args.cmd == "backtest":
        backtest(d, parse_date(args.start) if args.start else None,
                 season=args.season)
    elif args.cmd == "sweep":
        sweep(d, apply=args.apply)
    elif args.cmd == "fetch-odds":
        if args.historical and args.start:
            cur = parse_date(args.start)
            while cur <= d:
                fetch_odds_day(cur, historical=True)
                cur += timedelta(days=1)
        else:
            fetch_odds_day(d, historical=args.historical)
    elif args.cmd == "import-odds":
        import_odds_csv(args.file)
    elif args.cmd == "market":
        market_report(d)
    elif args.cmd == "evaluate":
        evaluate(args.season)
    elif args.cmd == "track":
        track()
    elif args.cmd == "clv":
        fetch_clv()
    elif args.cmd == "export":
        export_site(d)
    elif args.cmd == "daily":
        daily(d)
    elif args.cmd == "crossfit":
        crossfit(tuple(int(s) for s in args.seasons.split(",")),
                 apply=args.apply)
    else:
        score_slate(d, bets_only=args.bets)


if __name__ == "__main__":
    main()