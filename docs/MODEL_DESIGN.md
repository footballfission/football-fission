# Hierarchical model design

## Goal

Predict distributions of NFL player statistics, then convert simulated stat lines into fantasy points. Keep opportunity, efficiency, outcome uncertainty, and scoring separate so each stage can be inspected and improved.

## Current models

| Position | Model | Main structure |
| --- | --- | --- |
| WR | `HierarchicalParametric` | Negative-binomial targets, beta-binomial catches, signed per-catch gains, touchdown process, empirical rare rushing |
| RB | `RBChain` | Hurdle carries, targets coupled through a latent volume factor, conditional rushing and receiving production |
| TE | `TEChain` | Hurdle targets, conditional receptions and yardage, per-catch touchdowns, rare rushing |
| QB | `QBChain` | Hurdle passing opportunities, completions and passing production, interceptions, rushing |
| K | `KickerChain` | Hurdle field-goal attempts, distance-band allocation and make rates, extra-point attempts and makes |
| D/ST | `DefenseChain` | Count and hurdle distributions for defensive events, points allowed mapped through scoring tiers |

Each implementation is in [the model package](../src/hierarchical_football). Read the position module's docstring for its exact formulas, constraints, options, and known limitations.

## Conditional means and distribution shapes

LightGBM models estimate conditional opportunity, efficiency, and rate parameters from explicitly selected pregame features. Count dispersion, catch overdispersion, gain mixtures, and rare-event parameters supply uncertainty around those conditional predictions.

Models can estimate shapes on an internal historical calibration segment or accept a frozen shape layer estimated from earlier seasons. Keep all calibration information earlier than the evaluation cutoff. See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the fitting contract.

Signed gain distributions permit negative yardage. Binomial exposure stages connect receptions to targets and touchdowns to eligible events where the selected mode supports that constraint. The WR default separate touchdown mode has different constraints from its joint per-catch mode; inspect the option rather than assuming every mode behaves identically.

## Why a hierarchy

The hierarchy exposes volume, efficiency, and scoring assumptions separately. It can generate new stat combinations and provides an entry point for externally modeled opportunity. That flexibility also makes distributional assumptions and unrealistic tails important validation concerns.

Good individual-stat forecasts do not guarantee a good fantasy-point distribution. Relationships among catches, yards, touchdowns, and rushing must be retained. Current within-player dependence does not imply coherent teammate or opponent dependence.

## Scoring

[scoring.py](../src/hierarchical_football/scoring.py) converts named stat draws to points. For scrimmage scoring:

```text
points = alpha * receptions
       + 0.1 * (receiving_yards + rushing_yards)
       + 6 * (receiving_tds + rushing_tds)
```

Use alpha 0, 0.5, or 1 for standard, half PPR, or full PPR. Rescore the same draws without refitting. Other rules require the corresponding statistics and explicit scoring logic. Passing, kicking, and defense have separate scoring functions; these are research definitions rather than exhaustive platform support.

## Validation priorities

Evaluate MAE of predictive means, sample CRPS, interval coverage and width, zero mass, and floor/ceiling probability calibration. Use chronological evaluation and paired clustered uncertainty. Freeze a model configuration before evaluating on untouched data.

[The comparison guide](COMPARISON_GUIDE.md) explains the reference reports. Current concerns include WR ceiling inflation, TE touchdown dependence, QB volume bias, and D/ST points-allowed tier mass.

## Future extensions

Follow [TEAM_OPPORTUNITY_PLAN.md](TEAM_OPPORTUNITY_PLAN.md) for team pools, roster-wide opportunity allocation, availability mixtures, and reconciled outcomes. Begin with an explicit availability scenario and target redistribution while keeping downstream efficiency fixed. Add complexity only when an isolated evaluation establishes its value.
