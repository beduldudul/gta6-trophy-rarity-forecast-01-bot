# GTA VI trophy rarity forecaster

Forecasts every Grand Theft Auto VI (PS5) trophy's Exophase earn rate
at a case's target instant, as five quantiles per trophy, for
make.fun's `gta6-trophy-rarity-forecast-01`.

## How it works

Exophase's page prints each trophy's earn rate and the players-tracked
count. Two forces move a rate: earners grow as the tracked cohort
plays, and players-tracked grows in jumps when Exophase refreshes,
diluting most rates (a refresh adds mostly brand-new players who have
earned almost nothing yet). So the program never models the rate as
one smooth curve. It decomposes it:

```
rate(t) = 100 * E(t) / N(t)
```

`E(t)` = tracked players holding the trophy, `N(t)` = players tracked,
both recovered from `rate * N / 100`. Each is fitted in log space
against the saturating clock `x = ln(1 + hours since release)`, so
growth slows over the launch week instead of extrapolating linearly.
The forecast is the ratio of the two extrapolations, blended in logit
space with a direct logit-rate trend line fitted on the last 24 hours
of change points (stale, byte-identical reads are de-duplicated first,
so flat stretches cannot overweight a fit). Slopes shrink toward their
tier's mean slope; the median's move from the last read is capped and
saturating ceilings per trophy group keep early trophies from running
away.

When the history is flat -- no slope to fit, e.g. launch night before
Exophase's first refresh -- the program does not persist. A page that
has sat unchanged for hours on launch day is overdue a refresh, so it
forecasts an explicit two-branch mixture (no refresh vs refresh, with
the refresh probability from a Weibull hazard on the page's staleness
and the new cohort's earn level from the launch-night refresh's
trophies-earned arithmetic), with quantiles solved numerically.

Spreads widen with horizon, fit residual, disagreement between the two
views, and refresh exposure. All fitting is in logit space; 0% and
100% rates are handled with clipping / half-counts.

## Run

```
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input),
                  # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Standard library only, deterministic, ~0.1 s per case.

## License

MIT. See LICENSE.
