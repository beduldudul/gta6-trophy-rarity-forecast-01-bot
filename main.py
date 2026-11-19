#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a target instant.

Model (per trophy, all in logit / log space, deterministic, stdlib only)
-----------------------------------------------------------------------
Exophase's page prints, for each trophy, the share of tracked players who
have earned it, and the number of players tracked.  Two forces move a rate:

  * earners grow as the tracked cohort plays (progress), and
  * players-tracked grows in jumps when Exophase refreshes (dilution):
    a refresh adds mostly brand-new players who have earned almost
    nothing yet, so most rates fall at a refresh even though nobody
    un-earned anything.

So we never model the rate as one smooth curve.  We decompose it:

    rate(t) = 100 * E(t) / N(t)

where E(t) is the number of tracked players holding the trophy and N(t)
the players tracked (both recovered from rate * N / 100).  E and N are
both (near-)monotone growth curves, so each is fitted in log space
against x = ln(1 + hours since release) -- a saturating clock, so growth
slows as the launch week goes on instead of extrapolating linearly.
The forecast is the ratio of the two extrapolations.  Slopes are
shrunk toward their tier's mean slope (hierarchical shrinkage: trophies
share launch dynamics), and the change from the last read is capped.

When the history is flat -- every read identical, so there is no slope
to fit (launch night, before Exophase's first refresh) -- we do NOT
persist.  A page that has sat unchanged for hours on launch day is a
page overdue a refresh, so we forecast an explicit two-branch mixture:
no refresh (rate roughly stays) vs a refresh (players-tracked jumps
and the rate is diluted toward the new cohort's earn level).  The
mixture's quantiles are solved numerically, so the spread honestly
covers both branches.

Spreads widen with horizon (random-walk term), with fit residual, and
with refresh/dilution exposure.  Everything is clipped to (0, 100),
quantiles are ordered by construction, and 0% / 100% rates are handled
with half-count / logit clipping (never logit(0) or logit(1)).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(s):
    """ISO-8601 UTC timestamp -> aware datetime (tolerant of fractions)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def logit(p):
    """logit of a percent in (0, 100); p is clamped first."""
    p = min(max(p, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    """Inverse logit -> percent in (0, 100)."""
    if y >= 0:
        e = math.exp(-y)
        return 100.0 / (1.0 + e)
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def clip_pct(v):
    return min(max(v, 0.01), 99.99)


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(mu_a, s_a, w_a, mu_b, s_b, tau):
    """tau-quantile (in logit space) of a two-normal mixture.

    Solved by bisection; deterministic and monotone in tau.
    """
    lo = min(mu_a - 6.0 * s_a, mu_b - 6.0 * s_b) - 1.0
    hi = max(mu_a + 6.0 * s_a, mu_b + 6.0 * s_b) + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = w_a * norm_cdf((mid - mu_a) / s_a) + \
            (1.0 - w_a) * norm_cdf((mid - mu_b) / s_b)
        if c < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ----------------------------------------------------------------------
# Case handling
# ----------------------------------------------------------------------

def hours_since_release(t, release):
    return max(0.0, (t - release).total_seconds() / 3600.0)


def log_clock(t, release):
    """Saturating clock: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(t, release))


def change_points(history):
    """Distinct snapshots only.

    Exophase refreshes on its own schedule, so long stretches of reads
    are byte-identical.  Keeping every copy would overweight the flat
    stretches in any fit; a snapshot counts once, when something moved.
    """
    pts = []
    prev = None
    for h in history:
        key = (h.get("players_tracked"), tuple(h.get("rates") or ()))
        if key != prev:
            pts.append(h)
            prev = key
    return pts


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    at = parse_time(case.get("at")) or RELEASE
    target = parse_time(case.get("target")) or at
    horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)
    horizon_hours = max(0.0, (target - at).total_seconds() / 3600.0)
    release = parse_time((case.get("game") or {}).get("release_date")) \
        or RELEASE

    pts = change_points(history)
    n_cp = len(pts)
    span_h = 0.0
    if n_cp >= 2:
        t0 = parse_time(pts[0].get("at"))
        t1 = parse_time(pts[-1].get("at"))
        if t0 and t1:
            span_h = max(0.0, (t1 - t0).total_seconds() / 3600.0)

    # Last observed rate per trophy (from the last history read).
    last_rates = {}
    last_n = None
    if history:
        last = history[-1]
        last_n = last.get("players_tracked")
        for i, r in enumerate(last.get("rates") or ()):
            if r is not None:
                last_rates[i] = float(r)

    informative = n_cp >= 4 and span_h >= 6.0
    if informative and last_n:
        return forecast_growth(case, trophies, pts, last_rates, last_n,
                               at, target, horizon_days, release)
    return forecast_flat(case, trophies, pts, last_rates, last_n,
                         at, target, horizon_days, horizon_hours, release)


def forecast_growth(case, trophies, pts, last_rates, last_n,
                    at, target, horizon_days, release):
    """Regime A: fit log-earners and log-tracked against the log clock.

    Slopes are fitted on the last 24 hours of change points only: the
    launch surge (first hours, when every curve is steepest) must not
    set the slope for day 7.  The median blends two views in logit
    space -- the earners/tracked decomposition (which nets progress
    against dilution) and a direct logit-rate trend line -- so neither
    view's extrapolation error dominates.
    """
    x_t = log_clock(target, release)
    x_last = log_clock(parse_time(pts[-1].get("at")) or at, release)
    dx = max(0.0, x_t - x_last)

    # Recent window for fitting: last 24 h of change points, or all.
    from datetime import timedelta
    cutoff = (parse_time(pts[-1].get("at")) or at) - timedelta(hours=24)
    fit_pts = [h for h in pts
               if (parse_time(h.get("at")) or cutoff) >= cutoff]
    if len(fit_pts) < 4:
        fit_pts = pts

    # --- players-tracked curve: log N vs x over change points. ---
    xs_n, ys_n = [], []
    for h in fit_pts:
        t = parse_time(h.get("at"))
        n = h.get("players_tracked")
        if t is not None and n and n > 0:
            xs_n.append(log_clock(t, release))
            ys_n.append(math.log(float(n)))
    fit_n = ols(xs_n, ys_n)
    b_n = 0.0
    if fit_n is not None:
        b_n = min(max(fit_n[1], 0.0), 2.5)
    n_target = last_n * math.exp(b_n * dx)

    # --- per-trophy log-earner slopes, then tier shrinkage; and the
    # direct logit-rate trend line + its residual spread. ---
    tier_slopes = {}
    raw_slopes = {}
    resid = {}
    trend_mu = {}
    for idx, tr in enumerate(trophies):
        xs, ys, xl, yl = [], [], [], []
        for h in fit_pts:
            t = parse_time(h.get("at"))
            n = h.get("players_tracked")
            rates = h.get("rates") or ()
            if t is None or not n or idx >= len(rates):
                continue
            r = rates[idx]
            if r is None:
                continue
            x = log_clock(t, release)
            xs.append(x)
            ys.append(math.log(max(float(r) * float(n) / 100.0, 0.5)))
            xl.append(x)
            yl.append(logit(float(r)))
        fit = ols(xs, ys)
        if fit is not None and len(xs) >= 3:
            raw_slopes[idx] = fit[1]
            tier_slopes.setdefault(tr.get("tier", "bronze"), []).append(fit[1])
        fit_l = ols(xl, yl)
        if fit_l is not None and len(xl) >= 3:
            trend_mu[idx] = fit_l[0] + fit_l[1] * x_t
            res = [y - (fit_l[0] + fit_l[1] * x)
                   for x, y in zip(xl, yl)]
            resid[idx] = math.sqrt(sum(v * v for v in res) /
                                   max(1, len(res) - 2))
    tier_mean = {}
    for tier, vals in tier_slopes.items():
        tier_mean[tier] = sum(vals) / len(vals)
    global_mean = (sum(tier_mean.values()) / len(tier_mean)) if tier_mean \
        else 1.0
    prior = {"platinum": 0.4, "gold": 0.6, "silver": 0.9, "bronze": 1.1}

    rows = []
    for idx, tr in enumerate(trophies):
        r_last = last_rates.get(idx)
        if r_last is None:
            r_last = {"platinum": 0.6, "gold": 0.6,
                      "silver": 0.6, "bronze": 1.0}.get(
                          tr.get("tier", "bronze"), 1.0)
        if r_last >= 99.5:
            rows.append({"id": tr["id"],
                         "q": [99.0, 99.8, 100.0, 100.0, 100.0]})
            continue
        # Shrink this trophy's slope toward its tier mean (or prior).
        b_raw = raw_slopes.get(idx)
        n_pts = sum(1 for h in fit_pts
                    if idx < len(h.get("rates") or ())
                    and (h.get("rates") or ())[idx] is not None)
        centre = tier_mean.get(tr.get("tier", "bronze"),
                               prior.get(tr.get("tier", "bronze"), 1.0))
        if b_raw is None:
            b = centre
        else:
            w = n_pts / (n_pts + 4.0)
            b = w * b_raw + (1.0 - w) * centre
        b = min(max(b, -0.5), 3.0)

        # View 1 (decomposition): earners grow from their last observed
        # count at the shrunk slope; tracked players grow at their own
        # slope; the rate is the ratio, so dilution nets out.
        e_last = max(r_last * last_n / 100.0, 0.5)
        e_t = e_last * math.exp(b * dx)
        r_med = 100.0 * e_t / max(n_target, 1.0)
        mu_en = logit(clip_pct(r_med))
        # View 2 (direct trend): the logit-rate line read at the target.
        mu_tr = trend_mu.get(idx, mu_en)
        # Blend: neither extrapolation is trusted alone.
        mu = 0.5 * (mu_en + mu_tr)
        # Never let the median jump absurdly far from the last read.
        mu_last = logit(r_last)
        mu = mu_last + min(max(mu - mu_last, -2.5), 2.5)
        # Saturating ceilings: a week's play only takes so many tracked
        # players through each trophy.  These bind only when a fit
        # would otherwise run away (early trophies nearing saturation);
        # they are caps, not targets.
        tid = tr["id"]
        tier = tr.get("tier", "bronze")
        if tid == 2:
            ceil_pct = 99.9
        elif tid == 3:
            ceil_pct = 88.0
        elif tid in (4, 5, 6):
            ceil_pct = 75.0
        elif 7 <= tid <= 13:
            ceil_pct = 60.0
        elif tier == "platinum":
            ceil_pct = 3.0
        elif tier == "gold":
            ceil_pct = 30.0
        elif tier == "silver":
            ceil_pct = 40.0
        else:
            ceil_pct = 25.0
        mu = min(mu, logit(ceil_pct))
        r_med = inv_logit(mu)

        sig_fit = resid.get(idx, 0.3)
        sig_fit = min(max(sig_fit, 0.12), 0.7)
        rw = 0.38 * math.sqrt(horizon_days)
        disagree = 0.25 * abs(mu_en - mu_tr)
        sigma = math.sqrt(rw * rw + sig_fit * sig_fit +
                          (0.15 * math.sqrt(horizon_days)) ** 2 +
                          disagree * disagree)
        if tr.get("tier") == "platinum":
            sigma += 0.15
        sigma = min(max(sigma, 0.5), 1.6)
        q = [round(clip_pct(inv_logit(mu + z * sigma)), 4) for z in Z]
        q = sorted(q)
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}

# ----------------------------------------------------------------------
# Regime B: flat history -- a refresh is overdue, forecast the mixture
# ----------------------------------------------------------------------

def forecast_flat(case, trophies, pts, last_rates, last_n,
                  at, target, horizon_days, horizon_hours, release):
    """No usable slope: every read identical (or too few of them).

    Exophase refreshes a game's counts every few hours, not
    continuously, and a launch-night page that has sat unchanged for
    hours is overdue one.  A refresh multiplies players-tracked (the
    launch-night one went 638 -> 3,868) and the new cohort has earned
    almost nothing yet, so most rates are diluted to roughly
    old_count / new_count.  We therefore forecast a two-branch mixture
    per trophy -- no refresh vs refresh -- with the refresh probability
    from a Weibull hazard on how stale the page already is, and the
    new cohort's earn level from how early the trophy is (the very
    first trophies a new player meets are the only ones the new cohort
    plausibly holds already).
    """
    # How stale is the page at the checkpoint?
    if pts:
        last_change = parse_time(pts[-1].get("at")) or at
    elif case.get("history"):
        last_change = parse_time(case["history"][0].get("at")) or at
    else:
        last_change = at
    stale_h = max(0.0, (at - last_change).total_seconds() / 3600.0)
    # Weibull hazard, scale 4.5 h, shape 1.6: P(refresh in horizon).
    scale, shape = 4.5, 1.6
    surv_now = math.exp(-((stale_h / scale) ** shape))
    surv_end = math.exp(-(((stale_h + horizon_hours) / scale) ** shape))
    p_refresh = 1.0 - surv_end / max(surv_now, 1e-9)
    p_refresh = min(max(p_refresh, 0.0), 0.95)

    # Expected players-tracked multiplier on a refresh.  Early in the
    # launch the tracked base multiplies several-fold per refresh;
    # that growth decays as the launch week goes on.
    t_days = max(0.0, (at - release).total_seconds() / 86400.0)
    mult = 1.0 + 5.0 * math.exp(-t_days / 1.2)
    dx = max(0.0, log_clock(target, release) -
             log_clock(at, release))

    rows = []
    for idx, tr in enumerate(trophies):
        r_last = last_rates.get(idx)
        if r_last is None:
            r_last = {"platinum": 0.6, "gold": 0.6,
                      "silver": 0.6, "bronze": 1.0}.get(
                          tr.get("tier", "bronze"), 1.0)
        if r_last >= 99.5:
            rows.append({"id": tr["id"],
                         "q": [99.0, 99.8, 100.0, 100.0, 100.0]})
            continue
        # New cohort's earn level, relative to the old rate: the
        # launch-night refresh added 3,233 trophies over 3,230 new
        # players, and the 100% trophy alone accounts for ~3,230 of
        # them -- so the new cohort holds almost nothing else yet
        # (~2 earners a trophy).  Only the very first chapter gets a
        # slightly higher share.
        tid = tr["id"]
        if tid == 3:
            alpha = 0.10
        elif tid in (4, 5, 6, 20):
            alpha = 0.06
        elif tid in (7, 8, 9, 10, 11, 12, 13):
            alpha = 0.05
        else:
            alpha = 0.03
        dilute = (1.0 + alpha * (mult - 1.0)) / mult
        mu_last = logit(r_last)
        # Branch A: no refresh -- slow progress growth on the log clock.
        mu_a = mu_last + 0.5 * dx
        # Branch B: refresh -- diluted, then the same slow growth.
        mu_b = logit(clip_pct(r_last * dilute)) + 0.5 * dx
        s_a, s_b = 0.25, 0.55
        qs = []
        for tau in LEVELS:
            y = mixture_quantile(mu_a, s_a, 1.0 - p_refresh,
                                 mu_b, s_b, tau)
            qs.append(round(clip_pct(inv_logit(y)), 4))
        rows.append({"id": tr["id"], "q": sorted(qs)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
