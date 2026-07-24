# F5 Streakiness Model — v2

Scores MLB games for the **first 5 innings** based on recent form. Batting side:
each starter's last-14-day AVG/OBP/SLG percentile-ranked league-wide (70%) plus
team runs/game over the same window (30%). Pitching side: the starter's last 3
starts — runs allowed through 5 IP (40%), WHIP (30%), K% (20%), IP/start (10%),
all percentile-ranked against every qualified MLB starter.

## Setup
    pip install -r requirements.txt

## Daily workflow (one command)
    python f5_model.py daily
Runs the whole morning in order: grades yesterday's forward-test verdicts,
fetches today's F5 odds into the archive, and prints today's qualifying bets
(sorted by edge). Each step is failure-isolated — a network blip or missing
odds key skips that step with a note instead of killing the run.

## Manual workflow
    python f5_model.py snapshot     # once per day: builds league percentile context
    python f5_model.py score        # score today's slate (auto-builds snapshot if stale)
    python f5_model.py score --date 07/21/2026

Output: console table + `f5_scores_YYYY-MM-DD.json` with full per-batter and
per-pitcher detail.

## First run warning
The snapshot needs the game feed for every final game in the last 30 days
(~400-450 games) to compute runs-thru-5 for the league's starters. First build
takes ~10-15 min with the polite 0.25s delay. Feeds are cached in
`f5_cache/games/` forever (final games don't change), so every day after that
it only fetches the previous day's ~15 games.

## Data availability (v1.1)
No game is skipped and there is no status column — you judge confidence from
the numbers themselves:
- Each game shows `Batters available: q/t (pct%)` — a batter counts as
  available with >= 15 PA in the 14-day window (both lineups combined).
- Pitcher score prints `--` when the starter has < 3 starts in the last 30
  days (rolling; catches IL returns) or no probable pitcher is listed.
- Lineup not posted yet -> previous game's lineup, marked "lineup projected".

All knobs live in the CONFIG block at the top of `f5_model.py`.

## v2: handedness, backtest, predictions

**Platoon adjustment.** Each batter's 14-day form is multiplied by his
season-long platoon factor vs the opposing starter's hand:
(OBP+SLG vs that hand) / (overall OBP+SLG), clamped to 0.85-1.15, requiring
40+ PA vs that hand, falling back to the prior season (the April rule).
Switch hitters resolve automatically. The backtest uses PRIOR-season splits
only, so no future data leaks into historical predictions.

**Backtest + fit.**
    python f5_model.py backtest                # May 1 -> yesterday
    python f5_model.py backtest --start 06/01/2026
Reconstructs every historical slate with as-of snapshots, actual lineups and
actual starters, fits P(home leads after 5) = logistic(b0 + b1*bat_diff +
b2*pit_diff), chronological 70/30 train/validation, then sets the no-bet band
at ~70% bet volume and STRONG at the top quartile of edges. Params save to
`f5_cache/model_params.json`; per-game rows to `f5_backtest_rows_YYYY.json`.
READ THE VALIDATION LINE before trusting picks — if validation accuracy is
~50%, the model has no edge and NO BET is the correct verdict on everything.

**Weight sweep.**
    python f5_model.py sweep            # rank ~2,000 weight configs
    python f5_model.py sweep --apply    # adopt the winner for live scoring
Staged coordinate search over ALL internal weights, from stored backtest
components (no API calls): hitter blend (0.4-1.0), the four pitcher-metric
weights (0.1 simplex), and the AVG/OBP/SLG mix inside the batting form
(0.05 simplex). Alternates blend+pitcher and slash stages until no stage
improves. Each config is fit on the first 60% of games and ranked by
LOG-LOSS on the next 20% (selection set); ONLY the winner is evaluated on
the final untouched 20%. The "WINNER on untouched TEST tail" block is the number to trust; the
selection-set table is contaminated by the search itself. Also prints where
the current/default weights rank. Requires one `backtest` run on this version
first (older rows files lack the stored components).

**Market odds (v2.1).** Requires a key from the-odds-api.com in the
`ODDS_API_KEY` environment variable (PowerShell:
`$env:ODDS_API_KEY="yourkey"`).

    python f5_model.py fetch-odds        # today's F5 moneylines -> archive
    python f5_model.py import-odds x.csv # historical odds from CSV
                                         # (date,away,home,away_ml,home_ml)
    python f5_model.py fetch-odds --historical --start 05/01/2026
                                         # needs Odds API paid historical plan
    python f5_model.py market            # ROI backtest vs archived odds

Live `score` then shows the market line and a VALUE verdict: model P vs the
de-vigged market P, betting only when the model exceeds the market by
`market_edge_min` (tuned by the `market` command, default 3%). F5 moneylines
push on ties, matching the model's tie handling. Daily `fetch-odds` costs
~16 API credits (free tier = 500/month, so daily use just fits); each day you
fetch also grows your own odds archive for future ROI backtests. The `market`
report tunes the threshold on the first 80% of matched games and prints a
HOLDOUT ROI on the last 20% — that holdout line is the only number that
should touch your bankroll decisions.

**Frozen out-of-sample evaluation (the real exam).**
    python f5_model.py backtest --season 2025      # rows only; params untouched
    python f5_model.py fetch-odds --historical --start 05/01/2025 --date 09/28/2025
    python f5_model.py evaluate --season 2025      # tunes and saves NOTHING
Scores a completed season under the current frozen params: accuracy,
log-loss, calibration by confidence bucket, ROI at the frozen threshold, and
the pre-registered edge bands (0.03-0.07 expected positive, >=0.10 expected
negative). Past-season backtests deliberately skip fitting so the frozen
model can never be contaminated by the season it is being examined on.

**Batting window experiments (--bat-window).**
`backtest`, `sweep`, `market`, `evaluate`, and `crossfit` all accept
`--bat-window N` (default 14). Rows files and batter-stat caches are
window-tagged, so 7-day and 14-day universes coexist without collisions —
compare them by running the same crossfit under each flag. A winning window
is carried in the saved params (`batting_window_days`) and live `score`
honors it automatically.

**Cross-season fit (weight-stability search).**
    python f5_model.py crossfit                    # 2025 <-> 2026
    python f5_model.py crossfit --apply            # adopt pooled winner
For every weight config: fit coefficients on one season, score log-loss on
the OTHER, both directions, rank by the mean. Only cross-season signal can
win; per-season noise loses in both directions. Prints the baseline, the
current fitted params, and the winner's per-direction numbers. If the best
achievable cross log-loss sits at the 0.6931 coin-flip line, the report says
so plainly: the feature set, not the weights, is the limit. After a crossfit
both seasons are burned as holdouts — the fresh exam is season 2024
(backtest --season 2024, fetch its odds, evaluate --season 2024).

**Forward test (the live trial).** Every `score` on a slate with odds
automatically records that day's verdicts to `f5_forward_log.json` — pick,
tier, price, model and market probabilities, and a hash of the params that
produced them. First log wins: re-running score later never rewrites the bet
you'd have placed.

    python f5_model.py track

grades everything final (F5 ties = push, rain-shortened <5 innings = void),
then prints the running ledger: W-L-P, units, ROI, per-tier breakdown, and
live calibration (claimed vs delivered). It warns loudly if verdicts in the
ledger came from different params or if the current params differ from the
logged ones — don't retune mid-trial.

**Verdicts.** Once params exist, `score` adds per game:
    VERDICT: Cleveland Guardians  ·  P=58%  ·  LEAN
Tiers: STRONG / LEAN / NO BET. No verdict prints when any score is `--`.

**One-time costs on first v2 run:** the game cache schema changed (linescores
+ pitcher hands), so cached feeds refetch once (~15-20 min for a season). The
first backtest also fetches historical lineup batting stats (~2 calls/game,
resumable — safe to Ctrl-C and rerun).

## Known v1 simplifications (revisit in phase 2)
- Runners a starter leaves on base who score after his exit are charged to the
  reliever's play, not the starter (slightly kinder than official ER rules).
- Openers/bulk guys aren't detected — the first pitcher listed is "the starter."
- Doubleheaders: both games score independently; lineup fallback grabs the most
  recent final, which for game 2 of a twin bill will be game 1 (usually fine).
- Batter-vs-pitcher opponent adjustment intentionally NOT included yet.

## Testing
`test_e2e.py` runs the full pipeline against a mocked MLB API (6-team fake
league, 30 days of games) and asserts the OK / LOW CONFIDENCE / PROJECTED /
SKIPPED paths all behave. Run: `python test_e2e.py`
