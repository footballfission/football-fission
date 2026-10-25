# Pipeline verification

Verified on October 24, 2026.

- All 22 offline pipeline tests passed. They cover cutoff boundaries, report timing and conflicts, unknown versus assumed-zero outcomes, roster coverage, team residuals, ambiguous snap mappings, duplicate records, regular-season filtering, cache reuse, failed-refresh preservation, and model-input splits.
- A cached 2024 regular-season dataset built successfully for all six positions, producing 544 team-game rows. Recorded-outcome history rows: WR 2,441; RB 1,607; TE 1,223; QB 664; K 543; D/ST 544. These counts describe this input snapshot and default missing-outcome policy, not expected constants for every refresh.
- A WR week-10 split contained 1,233 earlier training rows and 121 prediction rows. The prepared-panel example fitted with a small verification configuration and generated 10 stat draws per prediction row.

This verifies pipeline compatibility and basic execution, not predictive accuracy or historical point-in-time roster correctness. No fresh network download was performed. Ingestion behavior was tested using injected loaders; actual source availability and optional-feed schemas still require operational checks. Generated verification datasets and model draws are not bundled.
