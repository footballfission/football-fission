# Injury modeling requirements

This package does not currently implement injury risk, weekly availability, recovery duration, or teammate opportunity redistribution. These notes define the distinctions needed by the [team opportunity plan](TEAM_OPPORTUNITY_PLAN.md).

## Separate prediction tasks

| Task | Required target | Intended use |
| --- | --- | --- |
| Season-level injury risk | Clearly defined event during a future season | Long-horizon scenario frequency |
| Weekly availability | Participation state at a forecast cutoff | Mixture over active/inactive roster scenarios |
| Limited workload | Usage conditional on playing | Reduced-role scenarios |
| Early exit | Exposure before leaving a game | In-game redistribution |
| Recovery duration | Longitudinal absence and return episodes | Multiweek projections |

A season-level probability cannot substitute for a weekly participation probability. An Out designation is not a complete clinical event or a known absence duration.

## Data requirements

Build a roster-complete player-week panel with stable identifiers. Retain healthy, inactive, and zero-opportunity players. Record report publication/update time, roster transaction time, ingestion time where available, and prediction cutoff.

Treat injury reports as weekly observations. Missing reports are not automatically healthy labels; players with long absences may stop appearing in weekly reports. Practice participation, game designation, roster status, and realized playing time answer different questions.

Distinguish injuries from rest, suspension, healthy scratches, and other absences. Audit reporting conventions across seasons and teams. Do not infer ground-truth weeks missed from consecutive Out designations alone.

## Initial experiment

Begin with supplied pregame availability assumptions so the allocation stage can be evaluated independently. Then introduce a calibrated weekly availability model. Keep oracle diagnostic scenarios separate from operational forecasts based only on information available at the cutoff.

Expanded roles may change efficiency and uncertainty as well as volume. Evaluate both the unavailable player and affected teammates. See [INJURY_RESEARCH_BOUNDARIES.md](INJURY_RESEARCH_BOUNDARIES.md) for research rules.
