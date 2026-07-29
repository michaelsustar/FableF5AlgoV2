"""
F5 Runs Model — v1  ("First Five Totals")
=========================================
Predicts the RUN DISTRIBUTION of the first 5 innings — per team and combined —
and bets the F5 totals (over/under) market only when the model's probability
beats the de-vigged market price.

Companion to f5_model.py (the Hot Streak moneyline algo). This file is
standalone but imports f5_model for all shared infrastructure: API access,
the game cache, the league snapshot, batting/pitcher scoring, and odds
plumbing. Both models share caches, so nothing is fetched twice.

THE MODEL
---------
Each team's expected F5 runs is a Poisson regression (log link):

  log(lambda_team) = b0 + b_season * f_season
                        + b_recent * f_recent
                        + b_starter * f_starter
                        + b_park   * log(park_factor)

  f_season  = log( shrunk season-long team F5 runs/game / league F5 r/g )
              (empirical-Bayes shrinkage: PRIOR_GAMES of league-average
               games mixed in, so April rates don't swing wildly)
  f_recent  = (14-day Hot Streak batting score - 50) / 50
              (the SAME batting score the moneyline algo computes — frozen
               hot-streak params for hitter/slash weights, platoon included)
  f_starter = (50 - opposing starter's runs-thru-5 percentile) / 50
              (positive = leaky starter; percentile from the shared snapshot)
  park      = static run park factor of the HOME park (applied to both
              offenses; the b_park coefficient decides how much it matters)

The coefficients b_season / b_recent ARE the "season vs. recent" weights —
found by the fit on backtest data rather than hand-tuned. Overdispersion is
estimated from residuals; if F5 runs are noisier than Poisson (they usually
are), probabilities use a negative binomial with that dispersion.

The two team distributions are convolved numerically into a total-runs
distribution, giving P(over) / P(under) at any market line — plus per-team
expected runs for team-total context.

DISCIPLINE (same rules as the Hot Streak algo)
----------------------------------------------
- Fit on one season, `evaluate --season` FROZEN on the other. No retuning
  mid-trial; the ledger warns if params change under it.
- Forward ledger records market-mode verdicts only, first-log-wins.
- No site page until the model earns one in backtests.

Usage:
  python f5_runs.py backtest                # fit on this season's rows
  python f5_runs.py evaluate --season 2025  # frozen out-of-sample test
  python f5_runs.py score [--date MM/DD/YYYY] [--bets]
  python f5_runs.py fetch-odds [--date ...] [--historical --start ...]
  python f5_runs.py track                   # grade the forward ledger
  python f5_runs.py daily                   # grade -> odds -> score --bets

Prereq: the Hot Streak backtest rows for a season must exist
(`python f5_model.py backtest [--season YYYY]`) — this model fits on the
same reconstructed slates, reading them straight off disk.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from math import exp, log
from pathlib import Path

import f5_model as fm
from f5_model import (
    api_get, clamp, mean, mlb_date, odds_key, parse_date,
    get_game_summary, _american_profit, _odds_http,
)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
PRIOR_GAMES = 10                  # league-average games mixed into season rate
MAX_RUNS = 25                     # convolution support per team (0..MAX_RUNS)
MIN_ALPHA = 1e-6                  # below this dispersion -> plain Poisson
TRAIN_FRACTION = 0.70             # chronological train/validation split
L2_PENALTY = 1e-4                 # ridge on the GLM fit (not the intercept)

RUNS_EDGE_MIN = 0.05              # bet only if |model P - market P| >= this
RUNS_EDGE_STRONG = 0.08

ODDS_TOTALS_MARKET_KEY = "totals_1st_5_innings"
RUNS_PARAMS_FILE = fm.CACHE_DIR / "runs_model_params.json"
RUNS_FORWARD_LOG = Path("./f5r_forward_log.json")
SEASON_INDEX_DIR = fm.CACHE_DIR / "runs_index"

FEATURES = ("f_season", "f_recent", "f_starter", "log_park")  # after intercept

# Static run park factors, 1.00 = league average, indexed by HOME team id.
# Approximate multi-year (2023-25) run factors; absolute calibration matters
# little because b_park is fitted. Update this table as parks change.
PARK_FACTORS = {
    108: 1.02,   # LAA  Angel Stadium
    109: 1.04,   # ARI  Chase Field
    110: 1.01,   # BAL  Camden Yards
    111: 1.07,   # BOS  Fenway Park
    112: 1.00,   # CHC  Wrigley Field
    113: 1.06,   # CIN  Great American Ball Park
    114: 0.97,   # CLE  Progressive Field
    115: 1.12,   # COL  Coors Field
    116: 0.98,   # DET  Comerica Park
    117: 0.99,   # HOU  Daikin Park
    118: 1.04,   # KC   Kauffman Stadium
    119: 0.98,   # LAD  Dodger Stadium
    120: 1.00,   # WSH  Nationals Park
    121: 0.97,   # NYM  Citi Field
    133: 1.04,   # ATH  Sutter Health Park (Sacramento)
    134: 0.97,   # PIT  PNC Park
    135: 0.96,   # SD   Petco Park
    136: 0.92,   # SEA  T-Mobile Park
    137: 0.96,   # SF   Oracle Park
    138: 0.98,   # STL  Busch Stadium
    139: 0.97,   # TB   (see note: verify current home park's factor)
    140: 1.01,   # TEX  Globe Life Field
    141: 1.00,   # TOR  Rogers Centre
    142: 0.99,   # MIN  Target Field
    143: 1.03,   # PHI  Citizens Bank Park
    144: 1.01,   # ATL  Truist Park
    145: 1.01,   # CWS  Rate Field
    146: 0.97,   # MIA  loanDepot park
    147: 1.02,   # NYY  Yankee Stadium
    158: 0.99,   # MIL  American Family Field
}


def park_factor(home_team_id):
    return PARK_FACTORS.get(home_team_id, 1.00)


def runs_odds_archive_file(season):
    return Path(f"./f5r_odds_{season}.json")


# --------------------------------------------------------------------------
# DISTRIBUTIONS — Poisson / negative binomial pmf, and team convolution
# --------------------------------------------------------------------------
def run_pmf(lam, alpha):
    """P(runs = 0..MAX_RUNS) for one team. Var = lam + alpha*lam^2.
    alpha ~ 0 -> Poisson; alpha > 0 -> negative binomial (NB2)."""
    pmf = [0.0] * (MAX_RUNS + 1)
    if lam <= 0:
        pmf[0] = 1.0
        return pmf
    if alpha < MIN_ALPHA:
        p = exp(-lam)
        for k in range(MAX_RUNS + 1):
            pmf[k] = p
            p *= lam / (k + 1)
    else:
        r = 1.0 / alpha                      # NB "size"
        q = r / (r + lam)                    # success prob
        p = exp(r * log(q))                  # P(0) = q^r
        for k in range(MAX_RUNS + 1):
            pmf[k] = p
            p *= (r + k) / (k + 1) * (1.0 - q)
    s = sum(pmf)                             # renormalize the truncated tail
    return [v / s for v in pmf]


def convolve(pmf_a, pmf_b):
    """Distribution of (A + B) runs, truncated at MAX_RUNS total."""
    out = [0.0] * (MAX_RUNS + 1)
    for i, pa in enumerate(pmf_a):
        if pa == 0.0:
            continue
        for j, pb in enumerate(pmf_b):
            k = i + j
            if k > MAX_RUNS:
                break
            out[k] += pa * pb
    s = sum(out)
    return [v / s for v in out]


def over_under_probs(total_pmf, line):
    """(P(over), P(under), P(push)) at a totals line (x.5 or integer)."""
    p_over = sum(p for k, p in enumerate(total_pmf) if k > line)
    p_push = (total_pmf[int(line)]
              if float(line).is_integer() and 0 <= int(line) <= MAX_RUNS
              else 0.0)
    p_under = max(0.0, 1.0 - p_over - p_push)
    return p_over, p_under, p_push


# --------------------------------------------------------------------------
# POISSON GLM — hand-rolled IRLS, matching the codebase's no-dependency style
# --------------------------------------------------------------------------
def _solve(a_mat, b_vec):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b_vec)
    a = [row[:] + [b_vec[i]] for i, row in enumerate(a_mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular system in GLM fit")
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) \
               / a[r][r]
    return x


def fit_poisson_glm(obs, iters=50):
    """obs: list of (x_vector_without_intercept, y). Returns beta list
    [b0, b_season, b_recent, b_starter, b_park]."""
    p = len(FEATURES) + 1
    ybar = mean([y for _, y in obs]) or 1.0
    beta = [log(max(ybar, 0.05))] + [0.0] * (p - 1)
    for _ in range(iters):
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for x, y in obs:
            xv = [1.0] + list(x)
            eta = sum(b * v for b, v in zip(beta, xv))
            mu = exp(clamp(eta, -6.0, 4.0))       # sane lambda range
            r = y - mu
            for i in range(p):
                grad[i] += r * xv[i]
                for j in range(i, p):
                    hess[i][j] += mu * xv[i] * xv[j]
        for i in range(1, p):                     # ridge (skip intercept)
            grad[i] -= L2_PENALTY * beta[i]
            hess[i][i] += L2_PENALTY
        for i in range(p):                        # symmetrize
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = _solve(hess, grad)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-8:
            break
    return beta


def predict_lambda(beta, feats):
    eta = beta[0] + sum(b * feats[k] for b, k in zip(beta[1:], FEATURES))
    return exp(clamp(eta, -6.0, 4.0))


def estimate_alpha(obs, beta):
    """NB2 dispersion by method of moments on the fitted residuals."""
    num = den = 0.0
    for x, y in obs:
        mu = predict_lambda(beta, dict(zip(FEATURES, x)))
        num += (y - mu) ** 2 - mu
        den += mu * mu
    return max(0.0, num / den) if den else 0.0


# --------------------------------------------------------------------------
# SEASON F5 RATES — team runs-through-5 per game, built from the game cache
# --------------------------------------------------------------------------
def season_f5_index(season):
    """All cached final summaries for a season -> per-game team F5 runs.
    Returns list of {"date", "team_id", "f5"} sorted by date. The Hot Streak
    backtests already cached these games; nothing is fetched here."""
    rows = []
    if not fm.GAME_CACHE_DIR.exists():
        return rows
    for f in fm.GAME_CACHE_DIR.glob("*.json"):
        try:
            gs = json.loads(f.read_text())
        except Exception:
            continue
        d = gs.get("date", "")
        if not d.startswith(str(season)) or not gs.get("final"):
            continue
        if gs.get("innings_played", 0) < 5:
            continue
        for side in ("home", "away"):
            rows.append({"date": d,
                         "team_id": gs[side]["team_id"],
                         "f5": gs["f5_runs"][side]})
    rows.sort(key=lambda r: r["date"])
    return rows


def f5_rates_asof(index_rows, as_of_iso):
    """Shrunk team F5 runs/game and the league rate, using games BEFORE
    as_of. Returns ({team_id: shrunk_rate}, league_rate)."""
    totals = {}
    league_runs = league_games = 0
    for r in index_rows:
        if r["date"] >= as_of_iso:
            break
        t = totals.setdefault(r["team_id"], [0, 0])
        t[0] += r["f5"]; t[1] += 1
        league_runs += r["f5"]; league_games += 1
    league = (league_runs / league_games) if league_games else 2.3
    rates = {tid: (runs + PRIOR_GAMES * league) / (g + PRIOR_GAMES)
             for tid, (runs, g) in totals.items()}
    return rates, league


def live_season_index(slate_date):
    """Season-to-date F5 index for LIVE scoring: walks the schedule from
    Opening Day, fetching (and permanently caching) any final not yet in
    the game cache. First run after a layoff fetches the gap; daily runs
    fetch ~one slate. Cached per as-of date."""
    SEASON_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    f = SEASON_INDEX_DIR / f"{slate_date.isoformat()}.json"
    if f.exists():
        return json.loads(f.read_text())
    import statsapi
    start = date(slate_date.year, 3, 15)
    sched = statsapi.schedule(start_date=mlb_date(start),
                              end_date=mlb_date(slate_date - timedelta(days=1)))
    time.sleep(fm.API_DELAY)
    finals = [g for g in sched if g.get("status") == "Final"
              and g.get("game_type", "R") == "R"]
    rows, missing = [], 0
    for g in finals:
        pk = g["game_id"]
        cached = (fm.GAME_CACHE_DIR / f"{pk}.json").exists()
        if not cached:
            missing += 1
            if missing % 25 == 0:
                print(f"        ... caching game {missing}")
        gs = get_game_summary(pk)
        if gs.get("innings_played", 0) < 5:
            continue
        for side in ("home", "away"):
            rows.append({"date": gs["date"],
                         "team_id": gs[side]["team_id"],
                         "f5": gs["f5_runs"][side]})
    rows.sort(key=lambda r: r["date"])
    f.write_text(json.dumps(rows))
    if missing:
        print(f"        cached {missing} new game feed(s)")
    return rows


# --------------------------------------------------------------------------
# FEATURES from Hot Streak backtest rows (fit + evaluate share this)
# --------------------------------------------------------------------------
def _bat_score_from_features(side_feats, hw, slash_w):
    """Reconstruct the Hot Streak batting score from stored components,
    with the frozen hot-streak weights — same math as fm._rescore_rows."""
    comps = side_feats.get("hf_comps") or {}
    runs_pct = side_feats.get("runs_pct")
    if runs_pct is None or any(comps.get(k) is None
                               for k in ("avg", "obp", "slg")):
        return None
    hitter_form = sum(slash_w[k] * comps[k] for k in slash_w)
    return hw * hitter_form + (1 - hw) * runs_pct


def build_observations(season):
    """(observations, meta) from a season's Hot Streak backtest rows.
    Each game yields two observations — one per offense:
      x = (f_season, f_recent, f_starter, log_park),  y = that team's F5 runs
    meta carries per-game info for totals-level evaluation."""
    rows_file = fm.rows_path(season)
    if not rows_file.exists():
        sys.exit(f"{rows_file} not found — run "
                 f"`python f5_model.py backtest --season {season}` first "
                 f"(the runs model fits on the same reconstructed slates).")
    rows = json.loads(rows_file.read_text())
    params = fm.load_model_params() or {}
    hw = params.get("hitter_form_weight", fm.HITTER_FORM_WEIGHT)
    slash_w = params.get("slash_weights", fm.SLASH_WEIGHTS)

    index = season_f5_index(season)
    if not index:
        sys.exit("Game cache is empty for this season — run the Hot Streak "
                 "backtest first so the feeds are cached.")

    obs, meta, skipped = [], [], 0
    rate_cache = {}
    for r in sorted(rows, key=lambda r: r["date"]):
        d = r["date"]
        if d not in rate_cache:
            rate_cache[d] = f5_rates_asof(index, d)
        rates, league = rate_cache[d]

        gs_file = fm.GAME_CACHE_DIR / f'{r["gamePk"]}.json'
        if not gs_file.exists():
            skipped += 1; continue
        gs = json.loads(gs_file.read_text())
        home_id, away_id = gs["home"]["team_id"], gs["away"]["team_id"]
        pf = park_factor(home_id)

        game_obs = {}
        ok = True
        for side, tid in (("home", home_id), ("away", away_id)):
            opp = "away" if side == "home" else "home"
            bat = _bat_score_from_features(r["features"][side], hw, slash_w)
            opp_pit = (r["features"][opp].get("pit_pcts") or {}) \
                .get("runs_thru_5")
            team_rate = rates.get(tid)
            if bat is None or opp_pit is None or team_rate is None:
                ok = False; break
            feats = (log(team_rate / league),
                     (bat - 50.0) / 50.0,
                     (50.0 - opp_pit) / 50.0,
                     log(pf))
            game_obs[side] = (feats, r[f"{side}_f5"])
        if not ok:
            skipped += 1; continue
        obs.append(game_obs["home"])
        obs.append(game_obs["away"])
        meta.append({"date": d, "gamePk": r["gamePk"],
                     "home": game_obs["home"], "away": game_obs["away"],
                     "total": r["home_f5"] + r["away_f5"]})
    print(f"  {len(meta)} games -> {len(obs)} team observations "
          f"({skipped} skipped)")
    return obs, meta


# --------------------------------------------------------------------------
# FIT + EVALUATION METRICS
# --------------------------------------------------------------------------
def totals_metrics(meta, beta, alpha, line=4.5, label=""):
    """Model quality on game totals: RMSE, O/U log-loss & accuracy vs a
    league-average baseline, and simple calibration."""
    if not meta:
        return
    base_lam = mean([m["total"] for m in meta]) / 2.0
    sq = ll = bll = correct = n_dec = 0
    buckets = {}
    for m in meta:
        lam_h = predict_lambda(beta, dict(zip(FEATURES, m["home"][0])))
        lam_a = predict_lambda(beta, dict(zip(FEATURES, m["away"][0])))
        sq += (lam_h + lam_a - m["total"]) ** 2
        pmf = convolve(run_pmf(lam_h, alpha), run_pmf(lam_a, alpha))
        p_over, p_under, _ = over_under_probs(pmf, line)
        p = p_over / (p_over + p_under)
        went_over = m["total"] > line
        ll += -(log(max(p, 1e-9)) if went_over else log(max(1 - p, 1e-9)))
        bpmf = convolve(run_pmf(base_lam, alpha), run_pmf(base_lam, alpha))
        bo, bu, _ = over_under_probs(bpmf, line)
        bp = bo / (bo + bu)
        bll += -(log(max(bp, 1e-9)) if went_over
                 else log(max(1 - bp, 1e-9)))
        pick_over = p >= 0.5
        correct += (pick_over == went_over); n_dec += 1
        b = min(int(abs(p - 0.5) * 20), 4)      # 0.025-wide edge buckets
        rec = buckets.setdefault(b, [0, 0])
        rec[0] += (pick_over == went_over); rec[1] += 1
    n = len(meta)
    print(f"  {label}totals RMSE {(sq / n) ** 0.5:.3f} | "
          f"O/U {line} log-loss {ll / n:.4f} (baseline {bll / n:.4f}) | "
          f"side accuracy {correct / n_dec:.1%}")
    parts = []
    for b in sorted(buckets):
        w, t = buckets[b]
        lo = b * 0.025
        parts.append(f"edge {lo:.3f}+: {w}/{t} ({w / t:.0%})")
    print("    calibration by model edge: " + "  ".join(parts))


def load_runs_params():
    if RUNS_PARAMS_FILE.exists():
        return json.loads(RUNS_PARAMS_FILE.read_text())
    return None


def _runs_params_hash():
    if not RUNS_PARAMS_FILE.exists():
        return None
    import hashlib
    return hashlib.md5(RUNS_PARAMS_FILE.read_bytes()).hexdigest()[:10]


def backtest(season):
    """Fit the runs GLM on a season's reconstructed slates (chronological
    70/30 train/validation) and freeze the params."""
    print(f"Runs model fit — season {season}")
    obs, meta = build_observations(season)
    if len(meta) < 100:
        sys.exit("Not enough games to fit responsibly.")
    cut = int(len(meta) * TRAIN_FRACTION)
    train_meta, val_meta = meta[:cut], meta[cut:]
    train_obs = [o for m in train_meta for o in (m["home"], m["away"])]

    beta = fit_poisson_glm(train_obs)
    alpha = estimate_alpha(train_obs, beta)
    names = ["intercept"] + list(FEATURES)
    print("\n  fitted coefficients (log-lambda scale):")
    for nm, b in zip(names, beta):
        print(f"    {nm:<10} {b:+.4f}")
    print(f"    dispersion alpha = {alpha:.4f} "
          f"({'negative binomial' if alpha >= MIN_ALPHA else 'Poisson'})")
    print(f"\n  train ({len(train_meta)} games):")
    totals_metrics(train_meta, beta, alpha, label="")
    print(f"  validation ({len(val_meta)} games):")
    totals_metrics(val_meta, beta, alpha, label="")

    params = {
        "beta": [round(b, 6) for b in beta],
        "alpha": round(alpha, 6),
        "features": list(FEATURES),
        "prior_games": PRIOR_GAMES,
        "fit_season": season,
        "fit_games": len(train_meta),
        "edge_min": RUNS_EDGE_MIN,
        "edge_strong": RUNS_EDGE_STRONG,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
    }
    fm.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_PARAMS_FILE.write_text(json.dumps(params, indent=2))
    print(f"\nRuns model params saved -> {RUNS_PARAMS_FILE}")
    return params


def evaluate(season):
    """FROZEN out-of-sample test on another season. Never refits."""
    params = load_runs_params()
    if not params:
        sys.exit("No frozen runs params — run `backtest` first.")
    if params.get("fit_season") == season:
        print("*** WARNING: evaluating the SAME season the model was fit "
              "on — this is in-sample, not a real test. ***")
    print(f"Frozen evaluation — season {season} "
          f"(params fit on {params.get('fit_season')}, "
          f"hash {_runs_params_hash()})")
    _, meta = build_observations(season)
    totals_metrics(meta, params["beta"], params["alpha"], label="")


# --------------------------------------------------------------------------
# ODDS — F5 totals market, own archive, shared plumbing
# --------------------------------------------------------------------------
def load_runs_odds(season):
    f = runs_odds_archive_file(season)
    return json.loads(f.read_text()) if f.exists() else {}


def _extract_totals_market(event_odds):
    """Preferred book's F5 totals line + prices from an event-odds payload."""
    books = {b.get("key"): b for b in event_odds.get("bookmakers", [])}
    ordered = [books[k] for k in fm.ODDS_BOOK_PREFERENCE if k in books]
    ordered += [b for k, b in books.items()
                if k not in fm.ODDS_BOOK_PREFERENCE]
    for book in ordered:
        for mkt in book.get("markets", []):
            if mkt.get("key") != ODDS_TOTALS_MARKET_KEY:
                continue
            over = under = pt = None
            for o in mkt.get("outcomes", []):
                nm = str(o.get("name", "")).lower()
                if nm == "over":
                    over, pt = o.get("price"), o.get("point")
                elif nm == "under":
                    under = o.get("price")
            if over is not None and under is not None and pt is not None:
                return {"line": float(pt), "over_ml": over,
                        "under_ml": under, "book": book.get("key")}
    return None


def fetch_odds_day(d, historical=False):
    """Fetch F5 totals lines for date d into this model's archive."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("Set the ODDS_API_KEY environment variable.")
    season = d.year
    archive = load_runs_odds(season)
    if historical:
        snap = f"{d.isoformat()}T16:00:00Z"
        resp = _odds_http("/historical/sports/baseball_mlb/events",
                          {"apiKey": api_key, "date": snap})
        events = resp.get("data", [])
    else:
        resp = _odds_http("/sports/baseball_mlb/events",
                          {"apiKey": api_key,
                           "commenceTimeFrom": f"{d.isoformat()}T08:00:00Z",
                           "commenceTimeTo":
                               f"{(d + timedelta(days=1)).isoformat()}"
                               f"T09:00:00Z"})
        events = resp if isinstance(resp, list) else resp.get("data", [])
    got = skipped = 0
    for ev in events:
        ct = ev.get("commence_time")
        if ct:
            ev_date = (datetime.fromisoformat(ct.replace("Z", "+00:00"))
                       - timedelta(hours=4)).date().isoformat()
            if ev_date != d.isoformat():
                skipped += 1; continue
        else:
            ev_date = d.isoformat()
        key = odds_key(ev_date, ev.get("away_team"), ev.get("home_team"))
        if key in archive:
            skipped += 1; continue
        params = {"apiKey": api_key, "regions": fm.ODDS_REGIONS,
                  "markets": ODDS_TOTALS_MARKET_KEY, "oddsFormat": "american"}
        if historical:
            path = (f"/historical/sports/baseball_mlb/events/"
                    f"{ev['id']}/odds")
            params["date"] = f"{d.isoformat()}T16:00:00Z"
            payload = _odds_http(path, params).get("data", {})
        else:
            payload = _odds_http(
                f"/sports/baseball_mlb/events/{ev['id']}/odds", params)
        mkt = _extract_totals_market(payload)
        if not mkt:
            continue
        archive[key] = {"date": ev_date, "home": ev.get("home_team"),
                        "away": ev.get("away_team"), **mkt}
        got += 1
    runs_odds_archive_file(season).write_text(json.dumps(archive, indent=1))
    rem = getattr(_odds_http, "remaining", "?")
    note = f", {skipped} already archived" if skipped else ""
    print(f"  {d}: stored F5 totals for {got}/{len(events) - skipped} "
          f"games{note} -> {runs_odds_archive_file(season)} "
          f"(API credits left: {rem})")
    return got


# --------------------------------------------------------------------------
# LIVE SLATE SCORING
# --------------------------------------------------------------------------
def predict_total(params, lam_home, lam_away, market=None):
    """Per-team lambdas -> total distribution -> value verdict vs market."""
    alpha = params["alpha"]
    pmf = convolve(run_pmf(lam_home, alpha), run_pmf(lam_away, alpha))
    exp_total = sum(k * p for k, p in enumerate(pmf))
    out = {"lam_home": round(lam_home, 2), "lam_away": round(lam_away, 2),
           "exp_total": round(exp_total, 2)}
    if not market:
        out.update({"mode": "no-line", "pick": None, "tier": "NO LINE"})
        return out
    line = market["line"]
    p_over, p_under, p_push = over_under_probs(pmf, line)
    p = p_over / (p_over + p_under)          # push-conditioned, matches devig
    mkt_p = fm.devig(market["over_ml"], market["under_ml"])  # fair P(over)
    value = p - mkt_p
    side = "Over" if value > 0 else "Under"
    v = abs(value)
    out.update({"mode": "market", "line": line,
                "p_over": round(p, 3), "mkt_p_over": round(mkt_p, 3),
                "value": round(value, 3),
                "p_push": round(p_push, 3)})
    edge_min = params.get("edge_min", RUNS_EDGE_MIN)
    edge_strong = params.get("edge_strong", RUNS_EDGE_STRONG)
    if v < edge_min:
        out.update({"pick": None, "tier": "NO VALUE"})
    else:
        out.update({"pick": f"{side} {line:g}",
                    "pick_side": side.lower(),
                    "pick_ml": market["over_ml"] if side == "Over"
                               else market["under_ml"],
                    "tier": "STRONG" if v >= edge_strong else "LEAN"})
    return out


def score_slate(slate_date, bets_only=False):
    params = load_runs_params()
    if not params:
        sys.exit("No runs model params — run `python f5_runs.py backtest` "
                 "first.")
    hs_params = fm.load_model_params() or {}
    hw = hs_params.get("hitter_form_weight", fm.HITTER_FORM_WEIGHT)
    slash_w = hs_params.get("slash_weights", fm.SLASH_WEIGHTS)
    fm.BATTING_WINDOW_DAYS = hs_params.get("batting_window_days",
                                           fm.BATTING_WINDOW_DAYS)
    snap = fm.load_snapshot(slate_date)
    window_end = slate_date - timedelta(days=1)
    window_start = slate_date - timedelta(days=fm.BATTING_WINDOW_DAYS)

    print("  season F5 scoring rates ...")
    index = live_season_index(slate_date)
    rates, league = f5_rates_asof(index, slate_date.isoformat())
    print(f"        league F5 rate: {league:.2f} runs/team "
          f"({len(rates)} teams)")

    sched = api_get("schedule", {
        "sportId": 1, "date": mlb_date(slate_date),
        "hydrate": "probablePitcher"})
    dates = sched.get("dates", [])
    games = dates[0].get("games", []) if dates else []
    if not games:
        print(f"No MLB games on {slate_date}.")
        return []

    odds_arch = load_runs_odds(slate_date.year)
    results = []
    for g in games:
        pk = g["gamePk"]
        detailed = g.get("status", {}).get("detailedState", "")
        away, home = g["teams"]["away"], g["teams"]["home"]
        entry = {"gamePk": pk,
                 "matchup": f'{away["team"]["name"]} @ '
                            f'{home["team"]["name"]}',
                 "game_time": g.get("gameDate"),
                 "home_team": home["team"]["name"],
                 "away_team": away["team"]["name"]}
        if "Postponed" in detailed or "Cancelled" in detailed:
            entry["postponed"] = detailed
            results.append(entry); continue

        nodes = {"away": away, "home": home}
        pf = park_factor(home["team"]["id"])
        entry["park_factor"] = pf

        # starters: runs-thru-5 percentile is the feature; hand for platoon
        pitchers = {}
        for side in ("away", "home"):
            prob = nodes[side].get("probablePitcher") or {}
            p = {"name": prob.get("fullName"), "hand": None,
                 "runs5_pct": None}
            if prob.get("id"):
                _score, metrics, hand, _why = fm.starter_score(
                    prob["id"], slate_date, snap)
                p["hand"] = hand
                if metrics and metrics.get("pcts"):
                    p["runs5_pct"] = metrics["pcts"].get("runs_thru_5")
            pitchers[side] = p

        # lineups -> the same Hot Streak batting score
        all_lineups = {}
        for side in ("away", "home"):
            team_id = nodes[side]["team"]["id"]
            all_lineups[side] = fm.get_lineup(pk, side, team_id, slate_date)
        lineup_ids = [i for lu, _ in all_lineups.values() if lu for i in lu]
        split_caches = []
        if lineup_ids:
            split_caches = [
                fm.ensure_platoon_splits(lineup_ids, slate_date.year),
                fm.ensure_platoon_splits(lineup_ids, slate_date.year - 1)]

        lams = {}
        for side in ("away", "home"):
            opp = "home" if side == "away" else "away"
            tid = nodes[side]["team"]["id"]
            side_out = {"team": nodes[side]["team"]["name"],
                        "opp_starter": pitchers[opp]["name"],
                        "opp_runs5_pct": pitchers[opp]["runs5_pct"],
                        "season_f5_rate": None, "bat_score": None,
                        "lam": None}
            lineup, lineup_status = all_lineups[side]
            bat_score = None
            if lineup:
                side_out["lineup"] = lineup_status
                bat = fm.team_batting_score(
                    lineup, tid, snap, window_start, window_end,
                    opp_hand=pitchers[opp]["hand"],
                    split_caches=split_caches,
                    hitter_weight=hw, slash_weights=slash_w)
                bat_score = bat["score"]
                side_out["bat_score"] = bat_score
            rate = rates.get(tid)
            side_out["season_f5_rate"] = round(rate, 2) if rate else None
            if None not in (rate, bat_score,
                            pitchers[opp]["runs5_pct"]):
                feats = {"f_season": log(rate / league),
                         "f_recent": (bat_score - 50.0) / 50.0,
                         "f_starter":
                             (50.0 - pitchers[opp]["runs5_pct"]) / 50.0,
                         "log_park": log(pf)}
                lam = predict_lambda(params["beta"], feats)
                side_out["lam"] = round(lam, 2)
                lams[side] = lam
            entry[side] = side_out

        okey = odds_key(slate_date.isoformat(),
                        away["team"]["name"], home["team"]["name"])
        entry["market"] = odds_arch.get(okey)
        if len(lams) == 2:
            entry["prediction"] = predict_total(
                params, lams["home"], lams["away"], market=entry["market"])
        else:
            entry["prediction"] = None
        results.append(entry)

    print_slate(results, slate_date, bets_only=bets_only)
    out_file = Path(f"./f5r_scores_{slate_date.isoformat()}.json")
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nFull detail written to {out_file}")
    n_logged = log_forward(results, slate_date)
    if n_logged:
        print(f"Forward test: {n_logged} verdict(s) recorded -> "
              f"{RUNS_FORWARD_LOG} (grade with `track`)")
    return results


def print_slate(results, slate_date, bets_only=False):
    rule = "\u2500" * 70
    show = results
    if bets_only:
        show = [r for r in results
                if r.get("prediction") and r["prediction"].get("pick")]
        show.sort(key=lambda r: abs(r["prediction"].get("value") or 0),
                  reverse=True)
        print(f"\nF5 Totals Bets \u2014 {slate_date}  "
              f"({len(show)} qualifying of {len(results)} games)")
        if not show:
            print(rule)
            print("  No totals clear the value threshold today. "
                  "The correct number of bets is sometimes zero.")
            print(rule)
            return
    else:
        print(f"\nF5 Runs Model \u2014 {slate_date}  ({len(results)} games)")
    for r in show:
        print(rule)
        if r.get("postponed"):
            print(f"  {r['matchup']:<46} {r['postponed']}")
            continue
        pred = r.get("prediction")
        print(f"  {r['matchup']}")
        for side in ("away", "home"):
            s = r.get(side) or {}
            lam = f"{s['lam']:.2f}" if s.get("lam") is not None else "--"
            rate = (f"{s['season_f5_rate']:.2f}"
                    if s.get("season_f5_rate") is not None else "--")
            bat = (f"{s['bat_score']:.0f}"
                   if s.get("bat_score") is not None else "--")
            opp = (f"{s['opp_runs5_pct']:.0f}"
                   if s.get("opp_runs5_pct") is not None else "--")
            print(f"    {s.get('team', '?'):<24} xF5 {lam:>5}  "
                  f"(szn {rate}, bat {bat}, opp-SP runs5 pct {opp})")
        if not pred:
            print("    -- insufficient data for a projection --")
            continue
        line = f"    projected total: {pred['exp_total']:.2f}"
        if pred.get("mode") == "market":
            mkt = r["market"]
            line += (f"   MARKET: {pred['line']:g} "
                     f"(O {mkt['over_ml']:+.0f} / U {mkt['under_ml']:+.0f}, "
                     f"{mkt.get('book', '?')})")
        print(line)
        if pred.get("mode") == "market":
            verdict = pred.get("pick") or "NO VALUE"
            ml = (f" {pred['pick_ml']:+.0f}"
                  if pred.get("pick_ml") is not None else "")
            print(f"    P(over)={pred['p_over']:.0%} vs market "
                  f"{pred['mkt_p_over']:.0%}  ->  {verdict}{ml}  "
                  f"[{pred['tier']}]")
        else:
            print("    (no totals line posted yet — projection only)")


# --------------------------------------------------------------------------
# FORWARD LEDGER — first log wins, graded against actual F5 totals
# --------------------------------------------------------------------------
def log_forward(results, slate_date):
    log_data = (json.loads(RUNS_FORWARD_LOG.read_text())
                if RUNS_FORWARD_LOG.exists() else {})
    added = 0
    for r in results:
        pred = r.get("prediction")
        if not pred or pred.get("mode") != "market":
            continue
        key = str(r["gamePk"])
        if key in log_data:
            if r.get("game_time") and not log_data[key].get("game_time"):
                log_data[key]["game_time"] = r["game_time"]
                added += 1
            continue
        log_data[key] = {
            "gamePk": r["gamePk"], "date": slate_date.isoformat(),
            "game_time": r.get("game_time"), "matchup": r["matchup"],
            "line": pred["line"], "pick": pred.get("pick"),
            "pick_side": pred.get("pick_side"),
            "pick_ml": pred.get("pick_ml"), "tier": pred["tier"],
            "p_over": pred["p_over"], "mkt_p_over": pred["mkt_p_over"],
            "value": pred.get("value"),
            "lam_home": pred["lam_home"], "lam_away": pred["lam_away"],
            "exp_total": pred["exp_total"],
            "params": _runs_params_hash(),
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "graded": False,
        }
        added += 1
    if added:
        RUNS_FORWARD_LOG.write_text(json.dumps(log_data, indent=1))
    return added


def track():
    if not RUNS_FORWARD_LOG.exists():
        sys.exit("No runs forward log yet — run `score` on a slate with "
                 "totals odds first.")
    log_data = json.loads(RUNS_FORWARD_LOG.read_text())
    cur_hash = _runs_params_hash()
    hashes = {rec.get("params") for rec in log_data.values()}
    if len(hashes) > 1:
        print("*** WARNING: verdicts in this ledger were produced by "
              "DIFFERENT model params — the trial is not clean. ***")
    if cur_hash not in hashes and hashes:
        print("*** WARNING: current runs params differ from the ones that "
              "logged these verdicts. Don't retune mid-trial. ***")

    today = date.today().isoformat()
    newly = 0
    for rec in log_data.values():
        if rec["graded"] or rec["date"] > today:
            continue
        try:
            gs = get_game_summary(rec["gamePk"])
        except Exception:
            continue
        if not gs.get("final"):
            continue
        if gs.get("innings_played", 0) < 5:
            rec.update({"graded": True, "outcome": "void", "units": 0.0,
                        "total": None})
            newly += 1; continue
        total = gs["f5_runs"]["home"] + gs["f5_runs"]["away"]
        line = rec["line"]
        outcome = ("push" if total == line else
                   "over" if total > line else "under")
        units = 0.0
        if rec.get("pick") and outcome != "push":
            units = _american_profit(rec["pick_ml"],
                                     outcome == rec["pick_side"])
        rec.update({"graded": True, "outcome": outcome, "total": total,
                    "units": round(units, 3)})
        newly += 1
    RUNS_FORWARD_LOG.write_text(json.dumps(log_data, indent=1))

    graded = [r for r in log_data.values() if r["graded"]]
    picks = [r for r in graded if r.get("pick")
             and r["outcome"] != "void"]
    pending = [r for r in log_data.values() if not r["graded"]]
    print(f"F5 TOTALS FORWARD TEST — {len(log_data)} verdicts logged, "
          f"{len(graded)} graded ({newly} just now), "
          f"{len(pending)} pending\n")

    recent = sorted(graded, key=lambda r: r["date"])[-10:]
    for r in recent:
        res = ("PUSH" if r["outcome"] == "push" else
               "VOID" if r["outcome"] == "void" else
               "WIN " if r.get("pick_side") == r["outcome"] else "LOSS")
        pick = r.get("pick") or "(no bet)"
        ml = f'{r["pick_ml"]:+.0f}' if r.get("pick_ml") else ""
        tot = f'F5={r["total"]}' if r.get("total") is not None else ""
        u = f'{r.get("units", 0):+.2f}' if r.get("pick") else ""
        print(f'  {r["date"]}  {r["matchup"]:<40} {pick:<12}{ml:>6} '
              f'{tot:>6} {res if r.get("pick") else "":<5}{u:>7}')

    def ledger(rows, label):
        dec = [r for r in rows if r["outcome"] not in ("push", "void")]
        wins = sum(1 for r in dec if r["pick_side"] == r["outcome"])
        pushes = sum(1 for r in rows if r["outcome"] == "push")
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
        dec = [r for r in picks if r["outcome"] not in ("push", "void")]
        if dec:
            claimed = mean([r["p_over"] if r["pick_side"] == "over"
                            else 1 - r["p_over"] for r in dec])
            actual = sum(1 for r in dec
                         if r["pick_side"] == r["outcome"]) / len(dec)
            print(f"\n  calibration in the wild: model claimed "
                  f"{claimed:.1%} on its picks, delivered {actual:.1%}")
    else:
        print("\n  No graded bets yet — verdicts grade automatically once "
              "games go final.")


def track_safe():
    if not RUNS_FORWARD_LOG.exists():
        print("  No verdicts logged yet — nothing to grade on day one.")
        return
    track()


# --------------------------------------------------------------------------
# SITE EXPORT — docs/data_totals.json for the First Five Totals dashboard
# --------------------------------------------------------------------------
def export_site(slate_date=None):
    slate_date = slate_date or date.today()
    site = Path("./docs")
    site.mkdir(exist_ok=True)
    log_data = (json.loads(RUNS_FORWARD_LOG.read_text())
                if RUNS_FORWARD_LOG.exists() else {})
    graded = [r for r in log_data.values() if r.get("graded")]
    picks = [r for r in graded if r.get("pick")
             and r.get("outcome") != "void"]
    dec = [r for r in picks if r["outcome"] not in ("push", "void")]
    wins = sum(1 for r in dec if r["pick_side"] == r["outcome"])
    pushes = sum(1 for r in picks if r["outcome"] == "push")
    units = sum(r.get("units", 0) for r in picks)

    by_day = {}
    for r in sorted(picks, key=lambda r: r["date"]):
        by_day[r["date"]] = by_day.get(r["date"], 0) + r.get("units", 0)
    cum, series = 0.0, []
    for d_iso, u in sorted(by_day.items()):
        cum += u
        series.append({"date": d_iso, "units": round(cum, 2)})

    todays = [r for r in log_data.values()
              if r["date"] == slate_date.isoformat()]
    bets = [{"matchup": r["matchup"], "pick": r["pick"],
             "ml": r.get("pick_ml"), "tier": r["tier"],
             "time": r.get("game_time"), "line": r["line"],
             "exp_total": r.get("exp_total"),
             "lam_home": r.get("lam_home"), "lam_away": r.get("lam_away"),
             "model_p": r["p_over"] if r["pick_side"] == "over"
                        else round(1 - r["p_over"], 3),
             "mkt_p": r["mkt_p_over"] if r["pick_side"] == "over"
                      else round(1 - r["mkt_p_over"], 3),
             "value": abs(r.get("value") or 0)}
            for r in todays if r.get("pick")]
    bets.sort(key=lambda b: (b["time"] is None, b["time"] or "",
                             -b["value"]))
    params = load_runs_params() or {}
    beta = params.get("beta", [])
    coefs = dict(zip(["intercept"] + list(params.get("features", FEATURES)),
                     beta)) if beta else {}

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "slate_date": slate_date.isoformat(),
        "ledger": {
            "bets": len(picks), "wins": wins,
            "losses": len(dec) - wins, "pushes": pushes,
            "units": round(units, 2),
            "roi": round(units / len(picks), 4) if picks else None,
            "hit": round(wins / len(dec), 4) if dec else None,
            "claimed": round(mean(
                [r["p_over"] if r["pick_side"] == "over"
                 else 1 - r["p_over"] for r in dec]), 4) if dec else None,
        },
        "series": series,
        "bets_today": bets,
        "evaluated_today": len(todays),
        "model": {
            "coefficients": {k: round(v, 4) for k, v in coefs.items()},
            "alpha": params.get("alpha"),
            "prior_games": params.get("prior_games", PRIOR_GAMES),
            "fit_season": params.get("fit_season"),
            "edge_min": params.get("edge_min", RUNS_EDGE_MIN),
            "edge_strong": params.get("edge_strong", RUNS_EDGE_STRONG),
            "batting_window_days": (fm.load_model_params() or {}).get(
                "batting_window_days", fm.BATTING_WINDOW_DAYS),
        },
    }
    (site / "data_totals.json").write_text(json.dumps(data, indent=1))

    # ---- full bet log for the "past bets" page (newest first) ----
    history = []
    for r in sorted(graded,
                    key=lambda r: (r["date"], r.get("game_time") or ""),
                    reverse=True):
        if not r.get("pick"):
            continue                      # NO VALUE verdicts aren't bets
        oc = r.get("outcome")
        result = ("push" if oc == "push" else
                  "void" if oc == "void" else
                  "win" if r.get("pick_side") == oc else "loss")
        history.append({
            "date": r["date"], "time": r.get("game_time"),
            "matchup": r["matchup"], "pick": r["pick"],
            "ml": r.get("pick_ml"), "tier": r["tier"],
            "line": r.get("line"), "exp_total": r.get("exp_total"),
            "lam_home": r.get("lam_home"), "lam_away": r.get("lam_away"),
            "total": r.get("total"),
            "model_p": r["p_over"] if r["pick_side"] == "over"
                       else round(1 - r["p_over"], 3),
            "mkt_p": r["mkt_p_over"] if r["pick_side"] == "over"
                     else round(1 - r["mkt_p_over"], 3),
            "value": abs(r.get("value") or 0),
            "result": result, "units": r.get("units", 0),
        })
    (site / "history_totals.json").write_text(json.dumps({
        "generated_at": data["generated_at"],
        "algo": "First Five Totals",
        "ledger": data["ledger"],
        "bets": history,
    }, indent=1))

    print(f"  docs/data_totals.json written — {len(bets)} bet(s) today, "
          f"ledger {wins}-{len(dec) - wins}-{pushes}, "
          f"{units:+.2f} units")
    print(f"  docs/history_totals.json written — {len(history)} graded bet(s)")


# --------------------------------------------------------------------------
# DAILY — grade, fetch totals odds, score, export the dashboard data
# --------------------------------------------------------------------------
def daily(slate_date):
    steps = [
        ("Grading yesterday's totals verdicts", lambda: track_safe()),
        ("Fetching today's F5 totals odds",
         lambda: fetch_odds_day(slate_date)),
        ("Scoring the slate (bets only)",
         lambda: score_slate(slate_date, bets_only=True)),
        ("Exporting the site data", lambda: export_site(slate_date)),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(steps)}] {label}\n{'=' * 70}")
        try:
            fn()
        except SystemExit as e:
            print(f"  (skipped: {e})")
        except Exception as e:
            print(f"  (step failed: {type(e).__name__}: {e} — continuing)")
    print(f"\n{'=' * 70}\nDone. Ledger: `python f5_runs.py track` any time."
          f"\n{'=' * 70}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="F5 runs/totals model (v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_bt = sub.add_parser("backtest",
                          help="fit the runs GLM on a season's slates")
    p_bt.add_argument("--season", type=int, default=date.today().year)
    p_ev = sub.add_parser("evaluate",
                          help="frozen-params out-of-sample season test")
    p_ev.add_argument("--season", type=int, required=True)
    p_sc = sub.add_parser("score", help="score a slate of games")
    p_sc.add_argument("--date", default=None)
    p_sc.add_argument("--bets", action="store_true",
                      help="show only totals that clear the value threshold")
    p_fo = sub.add_parser("fetch-odds",
                          help="fetch F5 totals lines to archive")
    p_fo.add_argument("--date", default=None)
    p_fo.add_argument("--historical", action="store_true")
    p_fo.add_argument("--start", default=None,
                      help="with --historical: fetch a date range")
    sub.add_parser("track", help="grade forward-test verdicts vs reality")
    p_ex = sub.add_parser("export",
                          help="write docs/data_totals.json for the site")
    p_ex.add_argument("--date", default=None)
    p_dy = sub.add_parser("daily",
                          help="one-shot: grade, fetch totals odds, "
                               "show today's bets")
    p_dy.add_argument("--date", default=None)
    args = ap.parse_args()

    d = parse_date(getattr(args, "date", None))
    if args.cmd == "backtest":
        backtest(args.season)
    elif args.cmd == "evaluate":
        evaluate(args.season)
    elif args.cmd == "score":
        score_slate(d, bets_only=args.bets)
    elif args.cmd == "fetch-odds":
        if args.historical and args.start:
            cur = parse_date(args.start)
            while cur <= d:
                fetch_odds_day(cur, historical=True)
                cur += timedelta(days=1)
        else:
            fetch_odds_day(d, historical=args.historical)
    elif args.cmd == "track":
        track()
    elif args.cmd == "export":
        export_site(d)
    elif args.cmd == "daily":
        daily(d)


if __name__ == "__main__":
    main()