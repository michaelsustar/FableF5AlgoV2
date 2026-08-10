#!/usr/bin/env python3
"""
First Five Moneyline (v2) -- feature core.

Five features, all date-bounded, entered as home-minus-away differentials:

  1. streakiness   P(F5 runs >= k) under NB2(mu_14d, alpha_season)
  2. f5_rate       season-to-date F5 runs/game, shrunk
  3. woba_hand     season-to-date wOBA vs the opposing starter's hand
  4. k_pct_hand    season-to-date K% vs that hand
  5. pitch_recent  composite of the starter's last 3 starts (K%, FIP, IP/GS)

Design notes that matter later:
  * Level moves week to week (14d window); dispersion is structural and needs
    the season to estimate (6 games ranks it barely better than a coin flip).
  * k is the league F5 runs/game as of the slate date -- continuous, not the
    integer median, which collapses to 2.0 in every run environment.
  * No park factor: park lifts both offenses on a moneyline and mostly cancels.
  * Everything here reads as-of a date. Nothing may see a game at or after
    the slate date -- that is what contaminated the v1 backtest.

This module is pure computation. Fetching lives in the data layer; every
function takes already-gathered rows so the math can be tested offline.
"""

import json
from math import exp, log, isfinite

import f5_model as fm
import f5_runs as fr


# --------------------------------------------------------------------------
# TUNING CONSTANTS -- deliberate calculations, not swept weights
# --------------------------------------------------------------------------
LEVEL_WINDOW_DAYS = 14        # recent-form window for mu
ALPHA_PRIOR_GAMES = 25        # league-dispersion games mixed into a team alpha
RATE_PRIOR_GAMES = fr.PRIOR_GAMES
LEAGUE_K_FALLBACK = 2.30      # league F5 runs/game before enough games exist
MIN_LEAGUE_GAMES = 150        # below this, k falls back rather than wobbling
PITCHER_STARTS = 3
MIN_PA_HAND = 40              # plate appearances vs a hand for a split to count

# Linear weights for wOBA. Vintage barely matters: the model ranks teams into
# percentiles, and a monotone rescaling of a ranking is the same ranking.
WOBA_W = {"bb": 0.690, "hbp": 0.720, "1b": 0.890,
          "2b": 1.270, "3b": 1.620, "hr": 2.100}

# Batting-order share of first-five plate appearances.
#
# Derived, not guessed: over 5 innings a team bats until 15 outs and the order
# carries across innings, so early slots come up more often. Simulating that
# at league out-rates (~0.71 outs/PA, ~21 F5 PA per team) gives the shares
# below. Slot 1 sees about 1.45x slot 9 -- a real skew, but far milder than
# it feels. The vector is nearly invariant to run environment: sweeping the
# out rate from 0.68 to 0.74 moves every weight by <0.003, which is why these
# are safe as constants rather than something recomputed per season.
ORDER_WEIGHTS = [0.1344, 0.1278, 0.1208, 0.1143, 0.1086,
                 0.1041, 0.1005, 0.0970, 0.0926]


# --------------------------------------------------------------------------
# MLB API FIELD MAPPING
# The API returns verbose camelCase names and some numbers as strings. These
# translate a raw stat block into the short keys the math below expects, so
# every field name lives in exactly one place.
# --------------------------------------------------------------------------
def _num(v, default=0):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return v
    try:
        return float(str(v).strip())
    except ValueError:
        return default


def from_mlb_hitting(stat):
    """Raw MLB hitting stat block -> the keys woba()/k_pct() expect."""
    return {
        "pa": _num(stat.get("plateAppearances")),
        "ab": _num(stat.get("atBats")),
        "h": _num(stat.get("hits")),
        "2b": _num(stat.get("doubles")),
        "3b": _num(stat.get("triples")),
        "hr": _num(stat.get("homeRuns")),
        "bb": _num(stat.get("baseOnBalls")),
        "ibb": _num(stat.get("intentionalWalks")),
        "hbp": _num(stat.get("hitByPitch")),
        "sf": _num(stat.get("sacFlies")),
        "sh": _num(stat.get("sacBunts")),
        "so": _num(stat.get("strikeOuts")),
    }


def from_mlb_pitching(stat):
    """Raw MLB pitching stat block -> the keys fip()/pitching_recent() expect."""
    return {
        "ip": _num(stat.get("inningsPitched")),
        "bf": _num(stat.get("battersFaced")),
        "er": _num(stat.get("earnedRuns")),
        "hr": _num(stat.get("homeRuns")),
        "bb": _num(stat.get("baseOnBalls")),
        "hbp": _num(stat.get("hitByPitch")),
        "so": _num(stat.get("strikeOuts")),
        "gs": _num(stat.get("gamesStarted")),
    }


# --------------------------------------------------------------------------
# RATE STATS -- computed from counting stats we can date-bound
# --------------------------------------------------------------------------
def woba(line):
    """wOBA from a counting-stat dict. Returns None when the denominator is
    empty, so callers can fall back rather than divide by zero."""
    ab = line.get("ab", 0)
    bb = line.get("bb", 0)
    ibb = line.get("ibb", 0)
    hbp = line.get("hbp", 0)
    sf = line.get("sf", 0)
    h = line.get("h", 0)
    d = line.get("2b", 0)
    t = line.get("3b", 0)
    hr = line.get("hr", 0)
    singles = max(0, h - d - t - hr)
    ubb = max(0, bb - ibb)

    denom = ab + ubb + sf + hbp
    if denom <= 0:
        return None
    num = (WOBA_W["bb"] * ubb + WOBA_W["hbp"] * hbp + WOBA_W["1b"] * singles
           + WOBA_W["2b"] * d + WOBA_W["3b"] * t + WOBA_W["hr"] * hr)
    return num / denom


def k_pct(line):
    """Strikeouts per plate appearance. PA is derived when not supplied."""
    pa = line.get("pa")
    if not pa:
        pa = (line.get("ab", 0) + line.get("bb", 0) + line.get("hbp", 0)
              + line.get("sf", 0) + line.get("sh", 0))
    if pa <= 0:
        return None
    return line.get("so", 0) / pa


def fip(line, constant):
    """FIP from a pitcher counting line. `constant` comes from the league
    snapshot as of the same date -- it is what puts FIP on the ERA scale."""
    ip = innings(line.get("ip", 0))
    if ip <= 0:
        return None
    return ((13 * line.get("hr", 0)
             + 3 * (line.get("bb", 0) + line.get("hbp", 0))
             - 2 * line.get("so", 0)) / ip) + constant


def fip_constant(league_line):
    """League FIP constant: whatever makes league FIP equal league ERA."""
    ip = innings(league_line.get("ip", 0))
    er = league_line.get("er", 0)
    if ip <= 0:
        return 3.10
    era = 9 * er / ip
    raw = (13 * league_line.get("hr", 0)
           + 3 * (league_line.get("bb", 0) + league_line.get("hbp", 0))
           - 2 * league_line.get("so", 0)) / ip
    return era - raw


def innings(ip):
    """MLB innings are stored as 5.2 meaning 5 and 2/3. Convert to real."""
    if ip is None:
        return 0.0
    try:
        ip = float(ip)
    except (TypeError, ValueError):
        return 0.0
    whole = int(ip)
    frac = round((ip - whole) * 10)
    if frac >= 3:            # 5.3+ is malformed; treat as a full inning
        return float(whole + 1)
    return whole + frac / 3.0


# --------------------------------------------------------------------------
# STREAKINESS -- level from 14 days, dispersion from the season
# --------------------------------------------------------------------------
def window_mean(index_rows, as_of_iso, days, team_id=None):
    """Mean F5 runs over the trailing `days` before as_of. index_rows are
    the {"date","team_id","f5"} records f5_runs.season_f5_index produces."""
    from datetime import date as _date, timedelta
    y, m, d = (int(x) for x in as_of_iso.split("-"))
    start = (_date(y, m, d) - timedelta(days=days)).isoformat()
    tot = n = 0
    for r in index_rows:
        if r["date"] >= as_of_iso:
            break
        if r["date"] < start:
            continue
        if team_id is not None and r["team_id"] != team_id:
            continue
        tot += r["f5"]
        n += 1
    return (tot / n if n else None), n


def season_dispersion_asof(index_rows, as_of_iso):
    """Per-team NB2 alpha from season-to-date F5 runs, shrunk toward the
    league value. alpha = (var - mean)/mean^2, floored at 0 (a team cannot
    be more regular than Poisson in a way NB2 can express).

    Returns ({team_id: alpha}, league_alpha)."""
    by_team = {}
    league = []
    for r in index_rows:
        if r["date"] >= as_of_iso:
            break
        by_team.setdefault(r["team_id"], []).append(r["f5"])
        league.append(r["f5"])

    league_alpha = _alpha_mom(league)
    if league_alpha is None:
        league_alpha = 0.25

    out = {}
    for tid, runs in by_team.items():
        a = _alpha_mom(runs)
        n = len(runs)
        if a is None:
            out[tid] = league_alpha
        else:
            out[tid] = ((n * a + ALPHA_PRIOR_GAMES * league_alpha)
                        / (n + ALPHA_PRIOR_GAMES))
    return out, league_alpha


def _alpha_mom(runs):
    """Method-of-moments NB2 dispersion. None when the sample is too thin."""
    n = len(runs)
    if n < 2:
        return None
    mean = sum(runs) / n
    if mean <= 0:
        return None
    var = sum((x - mean) ** 2 for x in runs) / (n - 1)
    a = (var - mean) / (mean * mean)
    return max(0.0, a)


def league_k_asof(index_rows, as_of_iso):
    """The scoring bar: league F5 runs/game to date, left continuous.

    The integer median is 2 in essentially every run environment, so it
    self-calibrates in theory and never moves in practice. The mean does move.
    """
    tot = n = 0
    for r in index_rows:
        if r["date"] >= as_of_iso:
            break
        tot += r["f5"]
        n += 1
    if n < MIN_LEAGUE_GAMES:
        return LEAGUE_K_FALLBACK, n
    return tot / n, n


def streakiness(mu, alpha, k):
    """P(F5 runs >= k) under NB2(mu, alpha), interpolated for continuous k.

    Two offenses with the same mu score differently: the steady one clears a
    modest bar more often, the erratic one has fatter tails in both
    directions. No fitted weights -- this is a calculation.
    """
    if mu is None or mu <= 0:
        return None
    pmf = fr.run_pmf(mu, max(0.0, alpha or 0.0))

    surv = [0.0] * (len(pmf) + 1)
    for i in range(len(pmf) - 1, -1, -1):
        surv[i] = surv[i + 1] + pmf[i]

    lo = int(k)
    frac = k - lo
    if lo + 1 >= len(surv):
        return surv[-1]
    return surv[lo] + frac * (surv[lo + 1] - surv[lo])


# --------------------------------------------------------------------------
# LINEUP AGGREGATION -- weight by batting order, not a flat mean
# --------------------------------------------------------------------------
def lineup_value(per_batter, min_pa=MIN_PA_HAND, league_value=None):
    """Combine per-batter values into one lineup number, weighted by slot.

    per_batter: list in batting order of {"value", "pa"} (value may be None).
    Batters short of min_pa vs the hand fall back to the league value rather
    than dropping out, which would silently reweight the lineup.
    """
    num = den = 0.0
    used = 0
    for i, b in enumerate(per_batter[:len(ORDER_WEIGHTS)]):
        w = ORDER_WEIGHTS[i]
        v = b.get("value")
        if v is None or b.get("pa", 0) < min_pa:
            v = league_value
        if v is None:
            continue
        num += w * v
        den += w
        used += 1
    if den <= 0:
        return None, 0
    return num / den, used


# --------------------------------------------------------------------------
# RECENT PITCHING SCORE -- last 3 starts, collapsed to one number
# --------------------------------------------------------------------------
def pitching_recent(starts, fip_const, league):
    """One composite from a starter's last N starts.

    starts: list of per-start counting lines, most recent first.
    league: {"k_pct","fip","ip_per_start"} as of the date, for scaling.

    Components are z-scored against the league so they combine on one scale,
    then averaged equally -- deliberately not a fitted weight vector. Higher
    is a better pitcher.

    ip_per_start earns its slot: a starter who only goes four hands the rest
    of the first five to a reliever, which is a different bet than the model
    thinks it is pricing.
    """
    use = starts[:PITCHER_STARTS]
    if not use:
        return None, 0

    agg = {}
    for s in use:
        for key in ("so", "bb", "hbp", "hr", "er", "bf"):
            agg[key] = agg.get(key, 0) + s.get(key, 0)
        agg["ip"] = agg.get("ip", 0.0) + innings(s.get("ip", 0))

    ip = agg.get("ip", 0.0)
    if ip <= 0:
        return None, 0

    bf = agg.get("bf", 0)
    kp = (agg["so"] / bf) if bf else None
    f = (((13 * agg["hr"] + 3 * (agg["bb"] + agg["hbp"]) - 2 * agg["so"]) / ip)
         + fip_const)
    ipgs = ip / len(use)

    parts = []
    if kp is not None and league.get("k_pct"):
        parts.append((kp - league["k_pct"]) / max(1e-6, league.get("k_pct_sd", 0.04)))
    if league.get("fip"):
        # lower FIP is better, so the sign flips
        parts.append(-(f - league["fip"]) / max(1e-6, league.get("fip_sd", 0.90)))
    if league.get("ip_per_start"):
        parts.append((ipgs - league["ip_per_start"])
                     / max(1e-6, league.get("ip_per_start_sd", 0.90)))

    if not parts:
        return None, 0
    score = sum(parts) / len(parts)
    return (score if isfinite(score) else None), len(use)


# --------------------------------------------------------------------------
# DATA LAYER -- fetching, caching, and as-of assembly
#
# Cache keys are prefixed "ml_" throughout. V1 caches batter stats as
# "{gamePk}_{side}.json" for its 14-day window; reusing that key for a
# season-to-date fetch would silently return 14-day data. That bug would be
# invisible in the output, so the prefix is load-bearing, not cosmetic.
# --------------------------------------------------------------------------
import time
from datetime import date, timedelta

ML_BAT_CACHE = fm.CACHE_DIR / "ml_batstats"
ML_PITCH_CACHE = fm.CACHE_DIR / "ml_gamelogs"
ML_CHECKPOINT_DIR = fm.DATA_DIR / "ml_checkpoints"
SEASON_OPEN = (3, 15)


def season_open(season):
    return date(season, *SEASON_OPEN)


def fetch_batter_season_stats(lineup_ids, start, end, cache_key):
    """Season-to-date hitting lines for a lineup, date-bounded at `end`.

    end is the day BEFORE the slate, so nothing from the game being
    predicted (or after it) can enter the features."""
    ML_BAT_CACHE.mkdir(parents=True, exist_ok=True)
    f = ML_BAT_CACHE / f"{cache_key}.json"
    if f.exists():
        return json.loads(f.read_text())
    hydrate = (f"stats(group=[hitting],type=[byDateRange],"
               f"startDate={fm.mlb_date(start)},"
               f"endDate={fm.mlb_date(end)},sportId=1)")
    resp = fm.api_get("people", {
        "personIds": ",".join(str(i) for i in lineup_ids),
        "hydrate": hydrate})
    f.write_text(json.dumps(resp))
    return resp


def fetch_pitcher_gamelog(pid, season):
    """Every start a pitcher made in a season, with full counting stats.

    The distilled game cache omits HR/HBP/ER for starters, so FIP cannot be
    computed from it. One gameLog call per pitcher per season fills that gap
    and is then reusable for every date in the season.
    """
    ML_PITCH_CACHE.mkdir(parents=True, exist_ok=True)
    f = ML_PITCH_CACHE / f"{pid}_{season}.json"
    if f.exists():
        return json.loads(f.read_text())
    hydrate = f"stats(group=[pitching],type=[gameLog],season={season})"
    resp = fm.api_get("people", {"personIds": pid, "hydrate": hydrate})
    out = []
    for person in resp.get("people", []):
        for group in person.get("stats", []):
            for sp in group.get("splits", []):
                st = sp.get("stat", {})
                if _num(st.get("gamesStarted")) < 1:
                    continue
                line = from_mlb_pitching(st)
                line["date"] = sp.get("date", "")
                out.append(line)
    out.sort(key=lambda s: s["date"])
    f.write_text(json.dumps(out))
    return out


def starter_lines_asof(pid, season, as_of_iso, logs=None):
    """That pitcher's starts strictly before as_of, most recent first."""
    log = logs.get(pid) if logs is not None else fetch_pitcher_gamelog(pid, season)
    if not log:
        return []
    return sorted([s for s in log if s.get("date", "") < as_of_iso],
                  key=lambda s: s["date"], reverse=True)


def league_pitching_asof(all_logs, as_of_iso):
    """League scales for the pitching z-scores, using only starts before
    as_of. Returns the means, the spreads, and the FIP constant that puts
    FIP on the league ERA scale for this same population."""
    agg = {"ip": 0.0, "er": 0, "hr": 0, "bb": 0, "hbp": 0, "so": 0}
    per_pitcher = []
    for pid, log in all_logs.items():
        starts = [s for s in log if s.get("date", "") < as_of_iso]
        if not starts:
            continue
        for s in starts:
            agg["ip"] += innings(s.get("ip", 0))
            for k in ("er", "hr", "bb", "hbp", "so"):
                agg[k] += s.get(k, 0)
        if len(starts) < PITCHER_STARTS:
            continue
        recent = sorted(starts, key=lambda s: s["date"], reverse=True)[:PITCHER_STARTS]
        ip = sum(innings(s.get("ip", 0)) for s in recent)
        bf = sum(s.get("bf", 0) for s in recent)
        if ip <= 0 or bf <= 0:
            continue
        per_pitcher.append({
            "k_pct": sum(s.get("so", 0) for s in recent) / bf,
            "raw_fip": (13 * sum(s.get("hr", 0) for s in recent)
                        + 3 * sum(s.get("bb", 0) + s.get("hbp", 0)
                                  for s in recent)
                        - 2 * sum(s.get("so", 0) for s in recent)) / ip,
            "ip_per_start": ip / len(recent),
        })

    if agg["ip"] <= 0:
        return None
    const = fip_constant({"ip": agg["ip"], "er": agg["er"], "hr": agg["hr"],
                          "bb": agg["bb"], "hbp": agg["hbp"],
                          "so": agg["so"]})
    if not per_pitcher:
        return None

    def stats(key, offset=0.0):
        vals = [p[key] + offset for p in per_pitcher]
        m = sum(vals) / len(vals)
        var = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
               if len(vals) > 1 else 0.0)
        return m, max(1e-6, var ** 0.5)

    k_m, k_sd = stats("k_pct")
    f_m, f_sd = stats("raw_fip", const)
    i_m, i_sd = stats("ip_per_start")
    return {"k_pct": k_m, "k_pct_sd": k_sd,
            "fip": f_m, "fip_sd": f_sd,
            "ip_per_start": i_m, "ip_per_start_sd": i_sd,
            "fip_const": const, "n_pitchers": len(per_pitcher)}


def lineup_offense(resp, opp_hand, split_caches, league_woba, league_k):
    """Lineup wOBA and K% vs the opposing hand.

    Level is date-bounded season-to-date; the handedness adjustment is a
    multiplier from PRIOR-season splits. That split matters: season splits
    are whole-season files with no as-of bound, so using the current season
    would leak. Using the prior season in backtest but the current one live
    would be train/serve skew. Prior season in both is the only consistent,
    leak-free option, and platoon skill is stable enough to survive it.
    """
    wobas, kpcts = [], []
    for person in resp.get("people", []):
        pid = person.get("id")
        splits = []
        for s in person.get("stats", []):
            splits.extend(s.get("splits", []))
        if not splits:
            wobas.append({"value": None, "pa": 0})
            kpcts.append({"value": None, "pa": 0})
            continue
        line = from_mlb_hitting(splits[0].get("stat", {}))
        mult = fm.platoon_multiplier(pid, opp_hand, split_caches)
        w, kp = woba(line), k_pct(line)
        wobas.append({"value": w * mult if w is not None else None,
                      "pa": line["pa"]})
        # a tougher platoon matchup means more strikeouts, so the multiplier
        # inverts for K% -- it is a rate where higher is worse for the offense
        kpcts.append({"value": kp / mult if kp is not None and mult else None,
                      "pa": line["pa"]})

    lw, nw = lineup_value(wobas, league_value=league_woba)
    lk, _ = lineup_value(kpcts, league_value=league_k)
    return lw, lk, nw


# --------------------------------------------------------------------------
# CHECKPOINTED ROW BUILDER
# Rows are appended to a JSONL checkpoint as each date completes. A killed
# run resumes at the next unfinished date and loses nothing; every fetch is
# also cached on disk, so a resume re-reads rather than re-fetches.
# --------------------------------------------------------------------------
def checkpoint_path(season):
    return ML_CHECKPOINT_DIR / f"ml_rows_{season}.jsonl"


def load_checkpoint(season):
    """Return (rows, set_of_completed_dates). Tolerates a truncated final
    line, which is what a hard kill mid-write leaves behind."""
    f = checkpoint_path(season)
    rows, done = [], set()
    if not f.exists():
        return rows, done
    with open(f, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # partial write from an interrupted run
            if rec.get("_done_date"):
                done.add(rec["_done_date"])
            else:
                rows.append(rec)
    # Rows from a date that never got its completion marker belong to a run
    # that died mid-date. The resume rebuilds that date and appends again, so
    # keeping these would silently duplicate games in the fit.
    kept = [r for r in rows if r.get("date") in done]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"  checkpoint: discarded {dropped} row(s) from an "
              f"interrupted date (they will be rebuilt)")
    return kept, done


def append_checkpoint(season, rows, done_date):
    """Append a date's rows and its completion marker atomically enough:
    the marker is written last, so a crash mid-date leaves the date
    unmarked and it is simply redone on resume."""
    ML_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path(season), "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.write(json.dumps({"_done_date": done_date}) + "\n")
        fh.flush()


# --------------------------------------------------------------------------
# SELF-TEST -- runs offline, no API
# --------------------------------------------------------------------------
def _selftest():
    ok = lambda c, m: print(("  PASS  " if c else "  FAIL  ") + m) or c
    results = []

    # -- wOBA: a known line computed by hand
    line = {"ab": 500, "h": 150, "2b": 30, "3b": 3, "hr": 35,
            "bb": 80, "ibb": 5, "hbp": 8, "sf": 5, "so": 150}
    w = woba(line)
    singles = 150 - 30 - 3 - 35
    expect = ((0.690 * 75 + 0.720 * 8 + 0.890 * singles + 1.270 * 30
               + 1.620 * 3 + 2.100 * 35) / (500 + 75 + 5 + 8))
    results.append(ok(abs(w - expect) < 1e-9, f"wOBA math exact ({w:.4f})"))
    results.append(ok(0.30 < w < 0.50, "wOBA lands in a believable range"))
    results.append(ok(woba({"ab": 0}) is None, "wOBA guards an empty denominator"))

    # -- field mapping against a real byDateRange block (Ezequiel Tovar)
    raw = {"gamesPlayed": 12, "doubles": 1, "triples": 1, "homeRuns": 1,
           "strikeOuts": 10, "baseOnBalls": 1, "intentionalWalks": 0,
           "hits": 11, "hitByPitch": 0, "avg": ".244", "atBats": 45,
           "obp": ".255", "slg": ".378", "plateAppearances": 49,
           "sacBunts": 2, "sacFlies": 1}
    t = from_mlb_hitting(raw)
    results.append(ok(t["ab"] == 45 and t["h"] == 11 and t["hr"] == 1,
                      "hitting fields map correctly"))
    # PA must reconcile: AB + BB + HBP + SF + SH
    results.append(ok(t["pa"] == t["ab"] + t["bb"] + t["hbp"] + t["sf"] + t["sh"],
                      f"PA reconciles with its components ({t['pa']:.0f})"))
    tw = woba(t)
    results.append(ok(0.20 < tw < 0.34,
                      f"real .633-OPS line gives a believable wOBA ({tw:.3f})"))
    results.append(ok(abs(k_pct(t) - 10/49) < 1e-9,
                      f"K% off real line ({100*k_pct(t):.1f}%)"))
    # string numbers must not crash the mapper
    results.append(ok(from_mlb_pitching({"inningsPitched": "5.2",
                                         "battersFaced": 24})["ip"] == 5.2,
                      "string innings parse to a number"))

    # -- innings: 5.2 means five and two thirds, not five point two
    results.append(ok(abs(innings(5.2) - 5.6667) < 1e-3, "innings 5.2 -> 5.67"))
    results.append(ok(abs(innings(6.0) - 6.0) < 1e-9, "innings 6.0 -> 6.00"))

    # -- FIP constant recovers league ERA
    lg = {"ip": 43000.0, "er": 18500, "hr": 5000, "bb": 15000,
          "hbp": 1800, "so": 40000}
    c = fip_constant(lg)
    lgfip = fip(lg, c)
    era = 9 * lg["er"] / innings(lg["ip"])
    results.append(ok(abs(lgfip - era) < 1e-6,
                      f"league FIP == league ERA ({lgfip:.3f})"))
    results.append(ok(2.0 < c < 4.0, f"FIP constant is sane ({c:.3f})"))

    # -- streakiness: same mean, different dispersion -> different score
    k = 2.3
    steady = streakiness(2.3, 0.05, k)
    erratic = streakiness(2.3, 0.80, k)
    results.append(ok(steady > erratic,
                      f"steady beats erratic at equal mean "
                      f"({steady:.4f} vs {erratic:.4f})"))
    # and a better offense still beats a worse one
    results.append(ok(streakiness(3.0, 0.3, k) > streakiness(1.8, 0.3, k),
                      "higher mean scores higher at equal dispersion"))
    # continuity in k
    a, b = streakiness(2.3, 0.3, 2.0), streakiness(2.3, 0.3, 3.0)
    mid = streakiness(2.3, 0.3, 2.5)
    results.append(ok(b < mid < a, "score is monotone and continuous in k"))
    results.append(ok(streakiness(0, 0.3, k) is None, "zero mean guarded"))

    # -- dispersion recovery on synthetic seasons
    import random
    random.seed(4)
    rows = []
    for g in range(120):
        d = f"2026-{3 + g // 30:02d}-{1 + g % 28:02d}"
        # team 1 metronomic, team 2 boom-or-bust, same long-run mean
        rows.append({"date": d, "team_id": 1, "f5": random.choice([2, 2, 3, 2, 1, 3])})
        rows.append({"date": d, "team_id": 2, "f5": random.choice([0, 0, 6, 0, 5, 1])})
    rows.sort(key=lambda r: r["date"])
    alphas, lg_a = season_dispersion_asof(rows, "2026-12-31")
    results.append(ok(alphas[2] > alphas[1],
                      f"erratic team gets higher alpha "
                      f"({alphas[2]:.3f} vs {alphas[1]:.3f})"))

    # -- as-of integrity: a future game must not move a past answer
    early = season_dispersion_asof(rows, "2026-04-15")[0]
    late = season_dispersion_asof(rows, "2026-12-31")[0]
    results.append(ok(early != late, "as-of date actually changes the answer"))
    k_early, n_early = league_k_asof(rows, "2026-03-05")
    results.append(ok(k_early == LEAGUE_K_FALLBACK,
                      "k falls back before enough league games"))

    # -- window mean respects both ends of the window
    m, n = window_mean(rows, "2026-05-01", 14, team_id=1)
    results.append(ok(m is not None and n > 0, f"14d window found {n} games"))
    m_all, _ = window_mean(rows, "2026-05-01", 14)
    results.append(ok(m_all is not None, "league window mean computes"))

    # -- lineup weighting: top of the order must matter more
    good = {"value": 0.400, "pa": 200}
    bad = {"value": 0.250, "pa": 200}
    # Same nine hitters, two orders -- only placement differs.
    top_heavy, _ = lineup_value([good] * 4 + [bad] * 5)
    bot_heavy, _ = lineup_value([bad] * 5 + [good] * 4)
    results.append(ok(top_heavy > bot_heavy,
                      f"same lineup scores higher batting the good hitters "
                      f"first ({top_heavy:.4f} vs {bot_heavy:.4f})"))
    flat, _ = lineup_value([good] * 9)
    results.append(ok(abs(flat - 0.400) < 1e-9,
                      "a uniform lineup returns that value exactly"))
    thin = lineup_value([{"value": 0.9, "pa": 3}] * 9, league_value=0.320)[0]
    results.append(ok(abs(thin - 0.320) < 1e-9,
                      "sub-threshold batters fall back to league"))

    # -- pitching score: better line scores higher
    league = {"k_pct": 0.22, "fip": 4.10, "ip_per_start": 5.3}
    ace = [{"ip": 6.2, "so": 9, "bb": 1, "hbp": 0, "hr": 0, "bf": 24}] * 3
    scrub = [{"ip": 4.0, "so": 2, "bb": 4, "hbp": 1, "hr": 2, "bf": 21}] * 3
    sa, na = pitching_recent(ace, 3.15, league)
    ss, _ = pitching_recent(scrub, 3.15, league)
    results.append(ok(sa > ss, f"ace outscores scrub ({sa:.2f} vs {ss:.2f})"))
    results.append(ok(na == 3, "uses exactly the last 3 starts"))
    results.append(ok(pitching_recent([], 3.15, league)[0] is None,
                      "no starts -> None, not a crash"))

    print(f"\n{sum(1 for r in results if r)}/{len(results)} checks passed")
    return all(results)


# --------------------------------------------------------------------------
# SEASON-LONG PITCHER QUALITY
#
# The 3-start composite measures ~18 innings, which is a thin read on a
# pitcher. Season-to-date, shrunk toward league, measures talent; the 3-start
# score stays as the form term. Same level/talent split used on offense.
#
# Costs no new fetches: K% and BB% come from the cached game logs, and F5
# runs allowed comes from runs_thru_5 in the cached game summaries -- which
# is the right denominator for this bet, unlike full-game runs.
# --------------------------------------------------------------------------
PITCH_PRIOR_BF = 250          # league batters-faced mixed into a rate
PITCH_PRIOR_STARTS = 8        # league starts mixed into F5 runs allowed


def build_starter_f5_index(summaries):
    """{pitcher_id: [{date, runs_thru_5}]} from the cached game feeds."""
    idx = {}
    for gs in summaries.values():
        for side in ("home", "away"):
            st = gs[side].get("starter")
            if not st:
                continue
            idx.setdefault(st["id"], []).append(
                {"date": gs["date"], "r5": st.get("runs_thru_5", 0)})
    for v in idx.values():
        v.sort(key=lambda s: s["date"])
    return idx


def league_season_pitching_asof(all_logs, f5idx, as_of_iso):
    """League K%, BB% and F5 runs allowed per start, as of a date."""
    so = bb = bf = 0
    for log in all_logs.values():
        for s in log:
            if s.get("date", "") >= as_of_iso:
                continue
            so += s.get("so", 0)
            bb += s.get("bb", 0)
            bf += s.get("bf", 0)
    r5 = starts = 0
    for lst in f5idx.values():
        for s in lst:
            if s["date"] >= as_of_iso:
                continue
            r5 += s["r5"]
            starts += 1
    if bf <= 0 or starts <= 0:
        return None
    return {"k_pct": so / bf, "bb_pct": bb / bf, "f5ra": r5 / starts,
            "bf": bf, "starts": starts}


def pitcher_season_asof(pid, as_of_iso, all_logs, f5idx, league):
    """Season-to-date K%, BB%, F5 runs allowed for one starter, each shrunk
    toward the league value so a two-start sample cannot dominate."""
    so = bb = bf = 0
    for s in all_logs.get(pid, []):
        if s.get("date", "") >= as_of_iso:
            continue
        so += s.get("so", 0)
        bb += s.get("bb", 0)
        bf += s.get("bf", 0)
    r5 = starts = 0
    for s in f5idx.get(pid, []):
        if s["date"] >= as_of_iso:
            continue
        r5 += s["r5"]
        starts += 1
    if bf <= 0 and starts <= 0:
        return None
    k = ((so + PITCH_PRIOR_BF * league["k_pct"]) / (bf + PITCH_PRIOR_BF))
    b = ((bb + PITCH_PRIOR_BF * league["bb_pct"]) / (bf + PITCH_PRIOR_BF))
    f = ((r5 + PITCH_PRIOR_STARTS * league["f5ra"])
         / (starts + PITCH_PRIOR_STARTS))
    return {"k_pct": k, "bb_pct": b, "f5ra": f, "bf": bf, "starts": starts}



# --------------------------------------------------------------------------
# SEASON WALKER -- assemble one feature row per game, checkpointed by date
# --------------------------------------------------------------------------
ALL_FEATURES = ("d_streak", "d_f5rate", "d_woba", "d_kpct", "d_pitch",
                "d_pk", "d_pbb", "d_pf5ra")

# The fitted set, selected on the 2025 fit season via `diagnose`:
#   d_pk      season K%          strongest solo signal (+0.0084 vs base)
#   d_f5rate  season F5 runs     +0.0042
#   d_pitch   3-start form       +0.0042
#   d_streak  streakiness        ~0 alone, but its removal costs +0.0013 --
#                                it corrects the others rather than
#                                predicting directly
# Dropped as redundant or empty: d_woba, d_kpct, d_pbb, d_pf5ra.
# Rows still carry all eight columns, so any subset can be refit without
# rebuilding a season.
FEATURES = ("d_streak", "d_f5rate", "d_pitch", "d_pk")


def build_rows(season, start=None, end=None):
    """Walk a season and write one feature row per usable game.

    Safe to interrupt: rows are checkpointed per date and every fetch is
    cached, so a resume re-reads from disk rather than re-fetching.
    """
    import statsapi

    opens = season_open(season)
    start = start or date(season, 4, 15)   # needs a few weeks of history first
    end = end or date(season, 11, 10)

    print(f"Building ML rows for {season}: {start} .. {end}")
    print("  [1/5] season schedule ...")
    sched = statsapi.schedule(start_date=fm.mlb_date(opens),
                              end_date=fm.mlb_date(end))
    time.sleep(fm.API_DELAY)
    finals = [g for g in sched if g.get("status") == "Final"
              and g.get("game_type", "R") == "R"]
    print(f"        {len(finals)} final regular-season games")
    if not finals:
        print("        nothing to do.")
        return []

    print("  [2/5] game feeds (cached from earlier runs where possible) ...")
    summaries = {}
    for i, g in enumerate(finals, 1):
        if i % 200 == 0:
            print(f"        ... {i}/{len(finals)}")
        summaries[g["game_id"]] = fm.get_game_summary(g["game_id"])

    starter_ids, batter_ids = set(), set()
    for gs in summaries.values():
        for side in ("home", "away"):
            st = gs[side].get("starter")
            if st:
                starter_ids.add(st["id"])
            batter_ids.update(gs[side].get("batting_order", [])[:9])

    print(f"  [3/5] pitcher game logs for {len(starter_ids)} starters "
          f"(one call each, cached) ...")
    all_logs = {}
    for i, pid in enumerate(sorted(starter_ids), 1):
        if i % 50 == 0:
            print(f"        ... {i}/{len(starter_ids)}")
        all_logs[pid] = fetch_pitcher_gamelog(pid, season)

    print(f"  [4/5] prior-season ({season - 1}) platoon splits for "
          f"{len(batter_ids)} batters ...")
    split_caches = [fm.ensure_platoon_splits(sorted(batter_ids), season - 1)]

    index_rows = fr.season_f5_index(season)
    print(f"        F5 run index: {len(index_rows)} team-games")
    f5idx = build_starter_f5_index(summaries)
    print(f"        starter F5-runs-allowed index: {len(f5idx)} pitchers")

    rows, done = load_checkpoint(season)
    if done:
        print(f"        resuming: {len(rows)} rows over "
              f"{len(done)} completed dates")

    print("  [5/5] walking dates ...")
    by_date = {}
    for g in finals:
        by_date.setdefault(g["game_date"], []).append(g)

    skipped = 0
    d = start
    while d <= end:
        d_iso = d.isoformat()
        todays = by_date.get(d_iso, [])
        if not todays or d_iso in done:
            d += timedelta(days=1)
            continue

        # ---- as-of league context (all offline, from cached data) ----
        k_bar, n_lg = league_k_asof(index_rows, d_iso)
        alphas, lg_alpha = season_dispersion_asof(index_rows, d_iso)
        rates, lg_rate = fr.f5_rates_asof(index_rows, d_iso)
        lgp = league_pitching_asof(all_logs, d_iso)
        lgs = league_season_pitching_asof(all_logs, f5idx, d_iso)
        if lgp is None or lgs is None:
            d += timedelta(days=1)
            continue

        # ---- pass 1: fetch every lineup for the day, pool a league offense
        # baseline from them (costs no extra API calls) ----
        fetched, usable = {}, []
        for g in todays:
            gs = summaries[g["game_id"]]
            if gs.get("innings_played", 0) < 5:
                skipped += 1
                continue
            ok = True
            for side in ("home", "away"):
                st = gs[side].get("starter")
                if not st or len(gs[side].get("batting_order", [])[:9]) < 9:
                    ok = False
                    break
            if not ok:
                skipped += 1
                continue
            for side in ("home", "away"):
                lineup = gs[side]["batting_order"][:9]
                fetched[(g["game_id"], side)] = fetch_batter_season_stats(
                    lineup, opens, d - timedelta(days=1),
                    cache_key=f'ml_{g["game_id"]}_{side}')
            usable.append(g)

        pooled = {"ab": 0, "h": 0, "2b": 0, "3b": 0, "hr": 0, "bb": 0,
                  "ibb": 0, "hbp": 0, "sf": 0, "sh": 0, "so": 0, "pa": 0}
        for resp in fetched.values():
            for person in resp.get("people", []):
                sp = []
                for grp in person.get("stats", []):
                    sp.extend(grp.get("splits", []))
                if not sp:
                    continue
                line = from_mlb_hitting(sp[0].get("stat", {}))
                for k in pooled:
                    pooled[k] += line.get(k, 0)
        lg_woba = woba(pooled)
        lg_kpct = k_pct(pooled)

        # ---- pass 2: build the rows ----
        day_rows = []
        for g in usable:
            gs = summaries[g["game_id"]]
            sides = {}
            ok = True
            for side in ("home", "away"):
                opp = "away" if side == "home" else "home"
                st = gs[side]["starter"]
                opp_hand = gs[opp]["starter"].get("hand")
                tid = gs[side]["team_id"]

                mu, n_recent = window_mean(index_rows, d_iso,
                                           LEVEL_WINDOW_DAYS, team_id=tid)
                if mu is None or n_recent < 3:
                    ok = False
                    break
                streak = streakiness(mu, alphas.get(tid, lg_alpha), k_bar)

                lw, lk, _n = lineup_offense(
                    fetched[(g["game_id"], side)], opp_hand,
                    split_caches, lg_woba, lg_kpct)

                lines = starter_lines_asof(st["id"], season, d_iso,
                                           logs=all_logs)
                pscore, n_starts = pitching_recent(lines, lgp["fip_const"],
                                                   lgp)
                seas = pitcher_season_asof(st["id"], d_iso, all_logs,
                                           f5idx, lgs)
                if None in (streak, lw, lk, pscore) or seas is None:
                    ok = False
                    break
                sides[side] = {"streak": streak,
                               "f5rate": rates.get(tid, lg_rate),
                               "woba": lw, "kpct": lk, "pitch": pscore,
                               "pk": seas["k_pct"], "pbb": seas["bb_pct"],
                               "pf5ra": seas["f5ra"],
                               "n_starts": n_starts,
                               "season_bf": seas["bf"]}
            if not ok:
                skipped += 1
                continue

            hr_, ar_ = gs["f5_runs"]["home"], gs["f5_runs"]["away"]
            h, a = sides["home"], sides["away"]
            day_rows.append({
                "date": d_iso, "gamePk": g["game_id"],
                "d_streak": h["streak"] - a["streak"],
                "d_f5rate": h["f5rate"] - a["f5rate"],
                "d_woba": h["woba"] - a["woba"],
                "d_kpct": h["kpct"] - a["kpct"],
                "d_pitch": h["pitch"] - a["pitch"],
                "d_pk": h["pk"] - a["pk"],
                "d_pbb": h["pbb"] - a["pbb"],
                "d_pf5ra": h["pf5ra"] - a["pf5ra"],
                "home_f5": hr_, "away_f5": ar_,
                "outcome": "home" if hr_ > ar_ else
                           "away" if ar_ > hr_ else "tie",
                "sides": sides,
            })

        append_checkpoint(season, day_rows, d_iso)
        rows.extend(day_rows)
        print(f"        {d_iso}: {len(day_rows)} rows "
              f"({len(rows)} total, k={k_bar:.2f}, "
              f"{lgp['n_pitchers']} qualified starters)")
        d += timedelta(days=1)

    print(f"\n  {len(rows)} usable games, {skipped} skipped "
          f"(short game, missing lineup, or thin history)")
    out = fm.ARTIFACT_DIR / f"ml_rows_{season}.json"
    fm.write_json(out, rows)
    print(f"  Rows -> {out}")
    return rows


# --------------------------------------------------------------------------
# FIT -- standardized IRLS logistic over the five differentials
#
# Features are z-scored so coefficients are directly comparable (one SD of
# each feature) and the Hessian stays well conditioned. The scaler is saved
# with the params and reapplied verbatim at evaluation and scoring time --
# recomputing it on a new season would silently change what the coefficients
# mean.
# --------------------------------------------------------------------------
ML_PARAMS_FILE = fm.CACHE_DIR / "ml_model_params.json"
L2 = 1e-4


def decisive(rows):
    """F5 moneylines push on a tie, so ties carry no information for the
    fit. They still count against effective sample size."""
    return [r for r in rows if r["outcome"] != "tie"]


def make_scaler(rows, feats=FEATURES):
    sc = {}
    for k in feats:
        vals = [r[k] for r in rows]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        sc[k] = {"mean": m, "sd": max(1e-9, var ** 0.5)}
    return sc


def design(rows, scaler, feats=FEATURES):
    X, y = [], []
    for r in rows:
        X.append([1.0] + [(r[k] - scaler[k]["mean"]) / scaler[k]["sd"]
                          for k in feats])
        y.append(1.0 if r["outcome"] == "home" else 0.0)
    return X, y


def fit_logistic_irls(X, y, iters=50, l2=L2):
    p = len(X[0])
    b = [0.0] * p
    for _ in range(iters):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for xi, yi in zip(X, y):
            z = sum(b[j] * xi[j] for j in range(p))
            mu = fm._sigmoid(z)
            w = max(mu * (1 - mu), 1e-9)
            err = mu - yi
            for i in range(p):
                g[i] += err * xi[i]
                for j in range(p):
                    H[i][j] += w * xi[i] * xi[j]
        for i in range(p):
            g[i] += l2 * b[i]
            H[i][i] += l2
        try:
            step = fr._solve([row[:] for row in H], g)
        except ValueError:
            # Singular Hessian: perfectly collinear features or separation.
            # Keep the last good coefficients rather than crashing the run.
            print("    warning: singular Hessian -- stopping IRLS early "
                  "(check the correlation table above)")
            break
        for i in range(p):
            b[i] -= step[i]
        if max(abs(s) for s in step) < 1e-10:
            break
    return b


def predict_p(row, params):
    """Scores with the feature list stored in the params, not the module
    default -- otherwise changing FEATURES would silently reinterpret
    already-frozen coefficients."""
    sc = params["scaler"]
    b = params["coef"]
    feats = params.get("features", list(FEATURES))
    z = b[0] + sum(b[i + 1] * (row[k] - sc[k]["mean"]) / sc[k]["sd"]
                   for i, k in enumerate(feats))
    return fm._sigmoid(z)


def metrics(rows, params):
    dec = decisive(rows)
    if not dec:
        return None
    ll = 0.0
    hits = 0
    for r in dec:
        p = fm.clamp(predict_p(r, params), 1e-9, 1 - 1e-9)
        y = 1.0 if r["outcome"] == "home" else 0.0
        ll -= y * log(p) + (1 - y) * log(1 - p)
        hits += (p >= 0.5) == (r["outcome"] == "home")
    return {"n": len(dec), "logloss": ll / len(dec), "acc": hits / len(dec)}


def correlations(rows, feats=FEATURES):
    out = {}
    for i, a in enumerate(feats):
        for b_ in feats[i + 1:]:
            va = [r[a] for r in rows]
            vb = [r[b_] for r in rows]
            ma, mb = sum(va) / len(va), sum(vb) / len(vb)
            num = sum((x - ma) * (y - mb) for x, y in zip(va, vb))
            da = sum((x - ma) ** 2 for x in va) ** 0.5
            db = sum((y - mb) ** 2 for y in vb) ** 0.5
            out[(a, b_)] = num / (da * db) if da and db else 0.0
    return out


def load_rows(season):
    f = fm.ARTIFACT_DIR / f"ml_rows_{season}.json"
    if not f.exists():
        raise SystemExit(f"{f} not found -- run `build --season {season}`.")
    rows = json.loads(f.read_text())
    rows.sort(key=lambda r: (r["date"], r["gamePk"]))
    return rows


def calibration(rows, params, label="calibration"):
    dec = decisive(rows)
    print(f"\n  {label}")
    print(f'  {"confidence":>12} {"games":>6} {"claimed":>8} {"actual":>8}')
    for lo, hi in ((0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)):
        bucket = []
        for r in dec:
            p = predict_p(r, params)
            conf = max(p, 1 - p)
            if lo <= conf < hi:
                won = (p >= 0.5) == (r["outcome"] == "home")
                bucket.append((conf, won))
        if not bucket:
            continue
        claimed = sum(c for c, _ in bucket) / len(bucket)
        actual = sum(1 for _, w in bucket if w) / len(bucket)
        print(f'  {f"{lo:.0%}-{hi:.0%}":>12} {len(bucket):>6} '
              f'{claimed:>8.1%} {actual:>8.1%}')


def fit(season, save=True, feats=None):
    feats = tuple(feats) if feats else FEATURES
    unknown = [f for f in feats if f not in ALL_FEATURES]
    if unknown:
        raise SystemExit(f"unknown feature(s): {', '.join(unknown)}\n"
                         f"available: {', '.join(ALL_FEATURES)}")
    rows = load_rows(season)
    dec = decisive(rows)
    ties = len(rows) - len(dec)
    print(f"FIT -- season {season}")
    print(f"  {len(rows)} games, {ties} ties dropped "
          f"({ties / len(rows):.1%}), {len(dec)} decisive\n")

    # ---- pre-registered diagnostic: feature correlations ----
    print(f"  features: {', '.join(feats)}\n")
    print("  feature correlations (|r| > 0.85 flagged):")
    for (a, b_), r_ in sorted(correlations(dec, feats).items(),
                              key=lambda kv: -abs(kv[1])):
        flag = "  <-- HIGH" if abs(r_) > 0.85 else ""
        print(f"    {a:>9} vs {b_:<9} {r_:+.3f}{flag}")

    # ---- internal chronological split: an early read that spends no
    # holdout season. Not the real exam -- that is the frozen 2024 test.
    n_tr = int(len(dec) * 0.70)
    tr, va = dec[:n_tr], dec[n_tr:]
    sc_tr = make_scaler(tr, feats)
    X, y = design(tr, sc_tr, feats)
    b_tr = fit_logistic_irls(X, y)
    p_tr = {"coef": b_tr, "scaler": sc_tr, "features": list(feats)}
    m_tr, m_va = metrics(tr, p_tr), metrics(va, p_tr)
    print(f"\n  internal split (fit {len(tr)} / held {len(va)}):")
    print(f"    train      log-loss {m_tr['logloss']:.4f}  "
          f"acc {m_tr['acc']:.1%}")
    print(f"    held-out   log-loss {m_va['logloss']:.4f}  "
          f"acc {m_va['acc']:.1%}   (coin flip = 0.6931)")

    # ---- final params on the whole season ----
    scaler = make_scaler(dec, feats)
    X, y = design(dec, scaler, feats)
    coef = fit_logistic_irls(X, y)
    params = {
        "coef": [round(c, 5) for c in coef],
        "scaler": {k: {"mean": round(v["mean"], 6),
                       "sd": round(v["sd"], 6)} for k, v in scaler.items()},
        "features": list(feats),
        "fitted_on": f"{season} ({dec[0]['date']}..{dec[-1]['date']})",
        "n_games": len(dec),
        "internal_holdout_logloss": round(m_va["logloss"], 4),
    }
    print(f"\n  coefficients (per 1 SD of each feature, on all {len(dec)}):")
    print(f'    {"intercept":>10} {coef[0]:+.4f}')
    for i, k in enumerate(feats):
        print(f"    {k:>10} {coef[i + 1]:+.4f}")
    print("\n  A positive coefficient means the home side of that "
          "differential\n  raises P(home leads after five). d_kpct is the "
          "offense's own\n  strikeout rate, so a negative sign there is the "
          "expected direction.")

    calibration(dec, params, "in-sample calibration (not evidence)")

    if save:
        fm.write_json(ML_PARAMS_FILE, params)
        print(f"\n  Params -> {ML_PARAMS_FILE}")
        print("  These are now FROZEN. The exam is:\n"
              f"    build --season {season - 1}\n"
              f"    evaluate --season {season - 1}")
    return params


def load_params():
    if not ML_PARAMS_FILE.exists():
        raise SystemExit(f"No params at {ML_PARAMS_FILE} -- run `fit` first.")
    return json.loads(ML_PARAMS_FILE.read_text())


def evaluate(season):
    """Frozen out-of-sample exam. Tunes nothing, writes nothing."""
    params = load_params()
    if str(season) in str(params.get("fitted_on", "")):
        print(f"*** WARNING: params were fit on {params['fitted_on']} -- "
              f"this evaluation is IN-SAMPLE, not an exam. ***\n")
    rows = load_rows(season)
    dec = decisive(rows)
    print(f"FROZEN EVALUATION -- season {season}")
    print(f"  params fit on: {params['fitted_on']}")
    print(f"  {len(rows)} games, {len(dec)} decisive "
          f"(tie rate {1 - len(dec) / len(rows):.1%})\n")

    m = metrics(dec, params)
    print(f"  log-loss {m['logloss']:.4f}  (coin flip = 0.6931)")
    print(f"  accuracy {m['acc']:.1%}")

    # ---- coefficient stability: refit on THIS season and compare ----
    feats = params.get("features", list(FEATURES))
    sc2 = make_scaler(dec, feats)
    X, y = design(dec, sc2, feats)
    b2 = fit_logistic_irls(X, y)
    print("\n  coefficient stability (fit season vs this one):")
    print(f'    {"feature":>10} {"fitted":>9} {"refit":>9}   sign')
    flips = 0
    for i, k in enumerate(feats):
        a, b_ = params["coef"][i + 1], b2[i + 1]
        same = (a >= 0) == (b_ >= 0)
        flips += not same
        print(f"    {k:>10} {a:+9.4f} {b_:+9.4f}   "
              f"{'same' if same else 'FLIPPED'}")
    if flips:
        print(f"\n  {flips} coefficient(s) flipped sign between seasons. "
              f"That is the\n  signature of season-specific signal rather "
              f"than a stable effect.")

    calibration(dec, params, "out-of-sample calibration")

    # ---- market test: the one that decides anything ----
    archive = fm.load_odds_archive(season)
    if not archive:
        print(f"\n  No {season} odds archive -- market test skipped. "
              f"Accuracy alone\n  proves nothing: the market may already "
              f"price all of this.")
        return
    joined = []
    for r in dec:
        gs = fm.get_game_summary(r["gamePk"])
        rec = archive.get(fm.odds_key(gs["date"], gs["away"]["team_name"],
                                      gs["home"]["team_name"]))
        if rec:
            joined.append((r, predict_p(r, params),
                           fm.devig(rec["home_ml"], rec["away_ml"]),
                           rec["home_ml"], rec["away_ml"]))
    print(f"\n  matched to odds: {len(joined)} games")
    if not joined:
        return
    print(f'\n  {"edge >=":>9} {"bets":>5} {"hit%":>7} {"units":>8} '
          f'{"ROI/bet":>9}')
    for t in (0.03, 0.05, 0.08, 0.10):
        bets = wins = 0
        units = 0.0
        for r, p, mp, hml, aml in joined:
            v = p - mp
            if abs(v) < t:
                continue
            side = "home" if v > 0 else "away"
            bets += 1
            won = r["outcome"] == side
            wins += won
            units += fm._american_profit(hml if side == "home" else aml, won)
        hit = f"{wins / bets:.1%}" if bets else "--"
        roi = f"{units / bets:+.1%}" if bets else "--"
        print(f"  {t:>9.2f} {bets:>5} {hit:>7} {units:>8.2f} {roi:>9}")
    print("\n  Positive ROI here, on a season the model never saw, is the "
          "only\n  line in this report that constitutes evidence of an edge.")


# --------------------------------------------------------------------------
# DIAGNOSE -- is a backwards coefficient a wiring bug or a collinearity
# artifact? Fit each feature ALONE. A feature whose solo coefficient points
# the right way but goes negative in the joint fit is redundant, not broken.
# --------------------------------------------------------------------------
EXPECTED_SIGN = {"d_streak": +1, "d_f5rate": +1, "d_woba": +1,
                 "d_kpct": -1, "d_pitch": +1,
                 # home starter misses more bats -> good for home
                 "d_pk": +1,
                 # home starter walks more -> bad for home
                 "d_pbb": -1,
                 # home starter allows more F5 runs -> bad for home
                 "d_pf5ra": -1}


def diagnose(season):
    rows = load_rows(season)
    dec = decisive(rows)
    print(f"DIAGNOSTIC -- season {season}, {len(dec)} decisive games\n")

    base_ll = None
    ybar = sum(1 for r in dec if r["outcome"] == "home") / len(dec)
    base_ll = -(ybar * log(ybar) + (1 - ybar) * log(1 - ybar))
    print(f"  home wins {ybar:.1%} of decisive games")
    print(f"  intercept-only log-loss {base_ll:.4f}\n")

    print("  EACH FEATURE ALONE (its own marginal direction):")
    print(f'  {"feature":>10} {"coef":>9} {"expect":>7} {"":>8} '
          f'{"logloss":>9} {"vs base":>9}')
    solo = {}
    for k in ALL_FEATURES:
        sub_rows = [{**r} for r in dec]
        scaler = {k: make_scaler(dec)[k]}
        X, y = [], []
        for r in dec:
            X.append([1.0, (r[k] - scaler[k]["mean"]) / scaler[k]["sd"]])
            y.append(1.0 if r["outcome"] == "home" else 0.0)
        b = fit_logistic_irls(X, y)
        ll = 0.0
        for xi, yi in zip(X, y):
            p = fm.clamp(fm._sigmoid(b[0] + b[1] * xi[1]), 1e-9, 1 - 1e-9)
            ll -= yi * log(p) + (1 - yi) * log(1 - p)
        ll /= len(X)
        solo[k] = b[1]
        want = EXPECTED_SIGN[k]
        agree = (b[1] >= 0) == (want > 0)
        print(f"  {k:>10} {b[1]:+9.4f} {('+' if want > 0 else '-'):>7} "
              f"{('ok' if agree else 'BACKWARDS'):>8} "
              f"{ll:>9.4f} {base_ll - ll:>+9.4f}")

    params = load_params() if ML_PARAMS_FILE.exists() else None
    feats_p = params.get("features") if params else None
    if feats_p:
        print("\n  SOLO vs JOINT (a sign change here means redundancy, "
              "not a bug):")
        print(f'  {"feature":>10} {"solo":>9} {"joint":>9}   verdict')
        for i, k in enumerate(feats_p):
            s_, j_ = solo[k], params["coef"][i + 1]
            want = EXPECTED_SIGN[k]
            solo_ok = (s_ >= 0) == (want > 0)
            joint_ok = (j_ >= 0) == (want > 0)
            if solo_ok and joint_ok:
                v = "stable"
            elif solo_ok and not joint_ok:
                v = "redundant (collinear)"
            elif not solo_ok and not joint_ok:
                v = "WIRING SUSPECT"
            else:
                v = "unstable"
            print(f"  {k:>10} {s_:+9.4f} {j_:+9.4f}   {v}")

    print("\n  LEAVE-ONE-OUT (does dropping a feature hurt held-out "
          "log-loss?):")
    n_tr = int(len(dec) * 0.70)
    tr, va = dec[:n_tr], dec[n_tr:]

    def ll_with(feats):
        if not feats:
            return None
        sc = {k: make_scaler(tr)[k] for k in feats}
        Xt = [[1.0] + [(r[k] - sc[k]["mean"]) / sc[k]["sd"] for k in feats]
              for r in tr]
        yt = [1.0 if r["outcome"] == "home" else 0.0 for r in tr]
        b = fit_logistic_irls(Xt, yt)
        tot = 0.0
        for r in va:
            z = b[0] + sum(b[i + 1] * (r[k] - sc[k]["mean"]) / sc[k]["sd"]
                           for i, k in enumerate(feats))
            p = fm.clamp(fm._sigmoid(z), 1e-9, 1 - 1e-9)
            yv = 1.0 if r["outcome"] == "home" else 0.0
            tot -= yv * log(p) + (1 - yv) * log(1 - p)
        return tot / len(va)

    full = ll_with(list(ALL_FEATURES))
    print(f"  {'all five':>22} {full:.4f}")
    for k in ALL_FEATURES:
        rest = [f for f in FEATURES if f != k]
        ll = ll_with(rest)
        delta = ll - full
        note = "  <-- better without it" if delta < 0 else ""
        print(f"  {('without ' + k):>22} {ll:.4f}  ({delta:+.4f}){note}")
    print("\n  Held-out log-loss on the fit season. Lower is better; a "
          "feature\n  whose removal IMPROVES this is costing you.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="First Five Moneyline v2 (f5_ml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build", help="assemble feature rows for a season")
    p_b.add_argument("--season", type=int, required=True)
    p_b.add_argument("--start", default=None, help="YYYY-MM-DD")
    p_b.add_argument("--end", default=None, help="YYYY-MM-DD")

    p_f = sub.add_parser("fit", help="fit the model on a built season")
    p_f.add_argument("--season", type=int, required=True)
    p_f.add_argument("--features", default=None,
                     help="comma-separated subset (default: the selected "
                          "four). Rows carry all eight, so any subset "
                          "refits without rebuilding.")

    p_e = sub.add_parser("evaluate",
                         help="frozen out-of-sample exam on another season")
    p_e.add_argument("--season", type=int, required=True)

    p_d = sub.add_parser("diagnose",
                         help="solo vs joint coefficients, leave-one-out")
    p_d.add_argument("--season", type=int, required=True)

    sub.add_parser("selftest", help="run the offline math checks")

    args = ap.parse_args()
    if args.cmd == "selftest":
        raise SystemExit(0 if _selftest() else 1)
    if args.cmd == "build":
        iso = lambda s: date(*(int(x) for x in s.split("-"))) if s else None
        build_rows(args.season, iso(args.start), iso(args.end))
    elif args.cmd == "fit":
        fit(args.season,
            feats=[f.strip() for f in args.features.split(",")]
                  if args.features else None)
    elif args.cmd == "evaluate":
        evaluate(args.season)
    elif args.cmd == "diagnose":
        diagnose(args.season)


if __name__ == "__main__":
    main()