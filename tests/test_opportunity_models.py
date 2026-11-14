"""Unit tests for opportunity pools, defense latents, and share arms."""
from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.opportunity.features import (
    build_defense_latents,
    build_skill_history,
    vacated_role_features,
)
from hierarchical_football.inference.opportunity.pools import build_opportunity_panels, extract_pool_events
from hierarchical_football.inference.opportunity.shares import fit_identity_shares, fit_latent_shares
from hierarchical_football.inference.opportunity.features import attach_model_features
from hierarchical_football.inference.opportunity.totals import fit_latent_volume


def _toy_pbp():
    rows = []
    # Two regular-season games, week 1 and 2.
    for week, game in ((1, "g1"), (2, "g2")):
        for play_id, kind, offense, defense, player, yards, success in (
            (1, "target", "KC", "BUF", "wr1", 10.0, 1.0),
            (2, "target", "KC", "BUF", "wr2", 5.0, 0.0),
            (3, "target", "KC", "BUF", "rb1", 3.0, 1.0),
            (4, "carry", "KC", "BUF", "rb1", 4.0, 1.0),
            (5, "carry", "KC", "BUF", "qb1", 2.0, 0.0),
            (6, "dropback", "KC", "BUF", "qb1", 0.0, 1.0),
            (7, "dropback", "KC", "BUF", "qb1", 0.0, 0.0),
            (8, "target", "BUF", "KC", "wr3", 8.0, 1.0),
            (9, "carry", "BUF", "KC", "rb2", 6.0, 1.0),
            (10, "dropback", "BUF", "KC", "qb2", 0.0, 1.0),
        ):
            base = {
                "season": 2023,
                "week": week,
                "game_id": game,
                "play_id": play_id + (week - 1) * 100,
                "season_type": "REG",
                "posteam": offense,
                "defteam": defense,
                "pass_attempt": 0,
                "sack": 0,
                "qb_scramble": 0,
                "play_type": "run",
                "receiver_player_id": None,
                "passer_player_id": None,
                "rush_attempt": 0,
                "qb_kneel": 0,
                "rusher_player_id": None,
                "qb_dropback": 0,
                "epa": 0.1,
                "success": success,
                "rushing_yards": 0.0,
                "passing_yards": 0.0,
                "home_team": "KC",
                "away_team": "BUF",
                "game_date": f"2023-09-0{week}",
            }
            if kind == "target":
                base.update({
                    "pass_attempt": 1,
                    "play_type": "pass",
                    "receiver_player_id": player,
                    "passer_player_id": "qb1" if offense == "KC" else "qb2",
                    "passing_yards": yards,
                })
            elif kind == "carry":
                base.update({
                    "rush_attempt": 1,
                    "rusher_player_id": player,
                    "rushing_yards": yards,
                })
            else:
                base.update({
                    "qb_dropback": 1,
                    "passer_player_id": player,
                    "pass_attempt": 1,
                    "play_type": "pass",
                })
            rows.append(base)
    return pl.DataFrame(rows)


def _toy_rosters():
    rows = []
    for week in (1, 2):
        for team, players in (
            ("KC", (("qb1", "QB"), ("rb1", "RB"), ("wr1", "WR"), ("wr2", "WR"), ("te1", "TE"))),
            ("BUF", (("qb2", "QB"), ("rb2", "RB"), ("wr3", "WR"), ("wr4", "WR"))),
        ):
            for player, position in players:
                # Week 2: wr2 is no longer active on KC (vacated role).
                if week == 2 and player == "wr2":
                    continue
                rows.append({
                    "season": 2023,
                    "week": week,
                    "team": team,
                    "gsis_id": player,
                    "position": position,
                    "status": "ACT",
                    "game_type": "REG",
                })
            rows.append({
                "season": 2023,
                "week": week,
                "team": team,
                "gsis_id": f"{team}-ol1",
                "position": "OL",
                "status": "ACT",
                "game_type": "REG",
            })
    return pl.DataFrame(rows)


class OpportunityPools(unittest.TestCase):
    def test_three_pools_conserve_and_include_qb_fb_population(self):
        panel, candidates, events, skill_players, ol_counts, audit = build_opportunity_panels(
            _toy_pbp(), _toy_rosters()
        )
        self.assertEqual(set(panel["pool"]), {"targets", "carries", "dropbacks"})
        self.assertIn("QB", set(candidates["position"]))
        # Week-2 events still credit wr2 after roster removal → OTHER mass is expected.
        self.assertGreaterEqual(audit["pools"]["targets"]["other_total"], 0)
        self.assertEqual(
            audit["pools"]["targets"]["total"],
            audit["pools"]["targets"]["candidate_total"] + audit["pools"]["targets"]["other_total"],
        )
        # Dropback candidates are QBs only.
        drop = candidates.filter(pl.col("pool") == "dropbacks")
        self.assertTrue(set(drop["position"]) <= {"QB"})
        self.assertGreater(extract_pool_events(_toy_pbp()).height, 0)
        self.assertGreater(ol_counts.height, 0)
        self.assertGreater(skill_players.height, 0)


class OpportunityFeatures(unittest.TestCase):
    def test_defense_latents_are_lagged(self):
        panel, candidates, events, skill_players, _, _ = build_opportunity_panels(
            _toy_pbp(), _toy_rosters()
        )
        defense = build_defense_latents(events, panel)
        week1 = defense.filter(pl.col("week") == 1)
        week2 = defense.filter(pl.col("week") == 2)
        # Week 1 uses only the prior (league) shrinkage; week 2 can move after week 1 games.
        self.assertTrue(week1.height > 0 and week2.height > 0)
        skills = build_skill_history(candidates, panel)
        self.assertTrue(
            skills.filter(pl.col("week") == 1)["prior_games"].max() == 0
        )
        vacated = vacated_role_features(skill_players, skills)
        # wr2 missing in week 2 for KC.
        row = vacated.filter((pl.col("week") == 2) & (pl.col("team") == "KC"))
        self.assertEqual(row.height, 1)
        self.assertGreaterEqual(row["vacated_skill_count"][0], 1)


class OpportunityFits(unittest.TestCase):
    def test_latent_and_identity_share_predict_sum_to_one(self):
        panel, candidates, events, skill_players, ol_counts, _ = build_opportunity_panels(
            _toy_pbp(), _toy_rosters()
        )
        skills = build_skill_history(candidates, panel)
        vacated = vacated_role_features(skill_players, skills)
        defense = build_defense_latents(events, panel)
        team_feats, player_feats = attach_model_features(
            panel, candidates, skills, vacated, defense, ol_counts
        )
        train_teams = team_feats.filter(pl.col("week") == 1)
        train_players = player_feats.filter(pl.col("week") == 1)
        pred_teams = team_feats.filter(pl.col("week") == 2)
        pred_players = player_feats.filter(pl.col("week") == 2)
        for pool in ("targets", "carries", "dropbacks"):
            latent = fit_latent_shares(
                train_teams.filter(pl.col("pool") == pool),
                train_players.filter(pl.col("pool") == pool),
                pool,
            )
            identity = fit_identity_shares(
                train_teams.filter(pl.col("pool") == pool),
                train_players.filter(pl.col("pool") == pool),
                pool,
                identity_ridge=10.0,
            )
            for model in (latent, identity):
                pred = model.predict(
                    pred_teams.filter(pl.col("pool") == pool),
                    pred_players.filter(pl.col("pool") == pool),
                )
                # Per team-game, candidate shares + OTHER ~= 1.
                keys = pred["team_game_keys"].with_row_index("_i")
                for row in keys.iter_rows(named=True):
                    members = pred["candidate_keys"].with_row_index("_j").filter(
                        (pl.col("game_id") == row["game_id"]) & (pl.col("team") == row["team"])
                    )["_j"].to_list()
                    total = pred["other_shares"][row["_i"]] + sum(pred["shares"][j] for j in members)
                    self.assertTrue(np.isclose(total, 1.0, atol=1e-6), msg=f"{pool} {row} {total}")
        volume = fit_latent_volume(
            train_teams.filter(pl.col("pool") == "targets"),
            train_players.filter(pl.col("pool") == "targets"),
            "targets",
        )
        means = volume.predict_mean(
            pred_teams.filter(pl.col("pool") == "targets"),
            pred_players.filter(pl.col("pool") == "targets"),
        )["means"]
        self.assertTrue(np.all(means > 0))


if __name__ == "__main__":
    unittest.main()
