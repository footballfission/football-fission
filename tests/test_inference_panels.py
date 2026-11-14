"""Offline checks for the team-game receiving panels."""
import unittest

import polars as pl

from hierarchical_football.inference.panels import (ACTIVE, CANDIDATE_POSITIONS, OTHER, available_players,
                                                    candidates, conservation, game_context, team_game_panel,
                                                    team_games)

from tests.inference_fixtures import game_frame, passing_frame, roster_frame

# One regular-season week: AAA hosts BBB and CCC hosts DDD. A fifth row is a postseason game the
# panel must not model.
WEEK_ONE = (
    {"game_id": "g1", "home_team": "AAA", "away_team": "BBB"},
    {"game_id": "g2", "home_team": "CCC", "away_team": "DDD"},
    {"game_id": "g3", "week": 19, "season_type": "POST", "home_team": "AAA", "away_team": "CCC"},
)

# AAA throws three targets: two to a receiver on its active roster and one to a lineman, who lands in
# OTHER. BBB throws two. CCC throws one to a receiver the roster feed never places on the team, and DDD
# throws none, so the panel must keep DDD's zero and CCC's OTHER target without a candidate row.
PLAYS = (
    {"game_id": "g1", "team": "AAA", "defteam": "BBB", "receiver_id": "wr1", "position": "WR"},
    {"game_id": "g1", "team": "AAA", "defteam": "BBB", "receiver_id": "wr1", "position": "WR"},
    {"game_id": "g1", "team": "AAA", "defteam": "BBB", "receiver_id": "ol1", "position": "UNKNOWN"},
    {"game_id": "g1", "team": "BBB", "defteam": "AAA", "receiver_id": "wr9", "position": "WR"},
    {"game_id": "g1", "team": "BBB", "defteam": "AAA", "receiver_id": "wr9", "position": "WR"},
    {"game_id": "g2", "team": "CCC", "defteam": "DDD", "receiver_id": "wr5", "position": "WR"},
)

ROSTER = (
    {"team": "AAA", "gsis_id": "wr1", "position": "WR", "status": "ACT"},
    {"team": "AAA", "gsis_id": "rb1", "position": "RB", "status": "ACT"},
    {"team": "AAA", "gsis_id": "te1", "position": "TE", "status": "ACT"},
    {"team": "AAA", "gsis_id": "ol1", "position": "OL", "status": "ACT"},
    {"team": "AAA", "gsis_id": "qb1", "position": "QB", "status": "ACT"},
    {"team": "AAA", "gsis_id": "wr2", "position": "WR", "status": "RES"},
    {"team": "AAA", "gsis_id": "wr3", "position": "WR", "status": "INA"},
    {"team": "BBB", "gsis_id": "wr9", "position": "WR", "status": "ACT"},
)


def panel_frame():
    """The panel and candidates for the fixture week, built the way the layer builds them."""
    return team_game_panel(passing_frame(*PLAYS), game_frame(*WEEK_ONE), roster_frame(*ROSTER))


def row(frame, **match):
    """The one row of a frame that matches every named column."""
    found = frame.filter(pl.all_horizontal([pl.col(k) == v for k, v in match.items()]))
    if found.height != 1:
        raise AssertionError(f"expected one row matching {match}, found {found.height}")
    return found.row(0, named=True)


class GameContext(unittest.TestCase):
    def test_a_regular_season_game_appears_once(self):
        games = game_context(game_frame(*WEEK_ONE))
        self.assertEqual(games.height, 2)
        self.assertEqual(sorted(games["game_id"].to_list()), ["g1", "g2"])

    def test_postseason_games_are_dropped(self):
        games = game_context(game_frame(*WEEK_ONE))
        self.assertNotIn("g3", games["game_id"].to_list())

    def test_a_game_that_appears_in_two_weeks_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            game_context(game_frame(*WEEK_ONE, {"game_id": "g1", "week": 2, "home_team": "AAA",
                                               "away_team": "BBB"}))

    def test_a_missing_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "game_context: missing columns"):
            game_context(game_frame(*WEEK_ONE).drop("home_team"))


class TeamGames(unittest.TestCase):
    def test_both_teams_of_every_game_appear(self):
        games = game_context(game_frame(*WEEK_ONE))
        panel = team_games(passing_frame(*PLAYS), games)
        self.assertEqual(panel.height, 4)
        self.assertEqual(sorted(panel["team"].to_list()), ["AAA", "BBB", "CCC", "DDD"])

    def test_home_and_opponent_are_recorded_from_the_game_row(self):
        games = game_context(game_frame(*WEEK_ONE))
        panel = team_games(passing_frame(*PLAYS), games)
        home = row(panel, team="AAA")
        away = row(panel, team="BBB")
        self.assertEqual((home["is_home"], home["opponent"]), (True, "BBB"))
        self.assertEqual((away["is_home"], away["opponent"]), (False, "AAA"))
        self.assertEqual(home["game_date"], "2024-09-08")

    def test_targets_are_the_recorded_targets_of_each_team(self):
        games = game_context(game_frame(*WEEK_ONE))
        panel = team_games(passing_frame(*PLAYS), games)
        self.assertEqual(row(panel, team="AAA")["targets"], 3)
        self.assertEqual(row(panel, team="BBB")["targets"], 2)
        self.assertEqual(row(panel, team="CCC")["targets"], 1)

    def test_a_team_game_without_a_target_keeps_a_zero(self):
        games = game_context(game_frame(*WEEK_ONE))
        panel = team_games(passing_frame(*PLAYS), games)
        ddd = row(panel, team="DDD")
        self.assertEqual(ddd["targets"], 0)
        self.assertIsNotNone(ddd["targets"])

    def test_a_passing_team_game_the_games_feed_does_not_contain_is_refused(self):
        games = game_context(game_frame(*WEEK_ONE))
        with self.assertRaisesRegex(ValueError, "games feed does not contain"):
            team_games(passing_frame(*PLAYS, {"game_id": "g9", "team": "EEE", "defteam": "AAA"}), games)

    def test_a_week_the_two_feeds_disagree_about_is_refused(self):
        games = game_context(game_frame(*WEEK_ONE))
        moved = tuple({**play, "week": 2} if play["team"] == "AAA" else play for play in PLAYS)
        with self.assertRaisesRegex(ValueError, "disagree about when"):
            team_games(passing_frame(*moved), games)

    def test_a_team_game_that_spans_two_weeks_is_refused(self):
        games = game_context(game_frame(*WEEK_ONE))
        with self.assertRaisesRegex(ValueError, "more than one week"):
            team_games(passing_frame(*PLAYS, {"game_id": "g1", "team": "AAA", "week": 2,
                                              "receiver_id": "wr1"}), games)

    def test_a_passing_frame_that_breaks_the_contract_is_refused(self):
        games = game_context(game_frame(*WEEK_ONE))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            team_games(passing_frame(*PLAYS).drop("air_yards"), games)


class AvailablePlayers(unittest.TestCase):
    def test_only_active_rows_are_candidates(self):
        players = available_players(roster_frame(*ROSTER))
        self.assertEqual(sorted(players["player_id"].to_list()), ["rb1", "te1", "wr1", "wr9"])

    def test_only_the_modelled_positions_are_kept(self):
        players = available_players(roster_frame(*ROSTER))
        self.assertEqual(sorted(players["position"].unique().to_list()), sorted(CANDIDATE_POSITIONS))

    def test_a_reserve_or_inactive_row_is_not_available(self):
        players = available_players(roster_frame(*ROSTER))
        self.assertNotIn("wr2", players["player_id"].to_list())
        self.assertNotIn("wr3", players["player_id"].to_list())

    def test_an_absent_status_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "available players: missing columns"):
            available_players(roster_frame(*ROSTER).drop("status"))


class Candidates(unittest.TestCase):
    def setUp(self):
        self.panel, self.candidates_frame = panel_frame()

    def test_one_row_per_available_candidate(self):
        aaa = self.candidates_frame.filter(pl.col("team") == "AAA")
        self.assertEqual(sorted(aaa["player_id"].to_list()), ["rb1", "te1", "wr1"])

    def test_a_candidate_without_a_target_keeps_an_observed_zero(self):
        rb1 = row(self.candidates_frame, team="AAA", player_id="rb1")
        self.assertEqual(rb1["targets"], 0)

    def test_the_index_orders_by_player_id_and_never_by_targets(self):
        aaa = self.candidates_frame.filter(pl.col("team") == "AAA").sort("candidate_index")
        self.assertEqual(aaa["player_id"].to_list(), ["rb1", "te1", "wr1"])
        self.assertEqual(aaa["targets"].to_list(), [0, 0, 2])
        self.assertEqual(aaa["candidate_index"].to_list(), [0, 1, 2])

    def test_the_index_restarts_for_every_team_game(self):
        indices = self.candidates_frame.group_by("game_id", "team").agg(
            pl.col("candidate_index").min().alias("low"), pl.col("candidate_index").max().alias("high"))
        self.assertEqual(indices.filter(pl.col("low") != 0).height, 0)

    def test_a_team_that_plays_twice_in_a_week_is_refused(self):
        twice = pl.concat([self.panel, self.panel.filter(pl.col("team") == "AAA")
                           .with_columns(pl.lit("g2").alias("game_id"))])
        with self.assertRaisesRegex(ValueError, "team games: .*duplicate key"):
            candidates(passing_frame(*PLAYS), roster_frame(*ROSTER), twice)


class Conservation(unittest.TestCase):
    def setUp(self):
        self.panel, self.candidates_frame = panel_frame()

    def test_candidates_and_other_add_up_to_the_team_total(self):
        report = conservation(self.panel, self.candidates_frame)
        self.assertEqual(report["team_games"], 4)
        self.assertEqual(report["targets"], 6)
        self.assertEqual(report["candidate_targets"], 4)
        self.assertEqual(report["other_targets"], 2)
        self.assertEqual(report["candidates"], 4)

    def test_the_other_residual_is_the_non_skill_target(self):
        aaa = row(self.panel, team="AAA")
        self.assertEqual((aaa["candidates"], aaa["candidate_targets"], aaa["other_targets"]), (3, 2, 1))

    def test_a_team_game_without_roster_coverage_puts_every_target_in_other(self):
        ddd = row(self.panel, team="DDD")
        self.assertEqual((ddd["candidates"], ddd["candidate_targets"], ddd["other_targets"]), (0, 0, 0))
        ccc = row(self.panel, team="CCC")
        self.assertEqual((ccc["candidates"], ccc["candidate_targets"], ccc["other_targets"]), (0, 0, 1))

    def test_games_with_other_targets_are_counted(self):
        self.assertEqual(conservation(self.panel, self.candidates_frame)["team_games_with_other"], 2)

    def test_a_negative_residual_is_refused(self):
        broken = self.panel.with_columns(
            pl.when(pl.col("team") == "AAA").then(pl.lit(-1)).otherwise(pl.col("other_targets"))
            .alias("other_targets"))
        with self.assertRaisesRegex(ValueError, "explain more targets than the team recorded"):
            conservation(broken, self.candidates_frame)

    def test_a_total_that_does_not_add_up_is_refused(self):
        broken = self.panel.with_columns(
            pl.when(pl.col("team") == "AAA").then(pl.lit(9)).otherwise(pl.col("targets")).alias("targets"))
        with self.assertRaisesRegex(ValueError, "do not add up to the team total"):
            conservation(broken, self.candidates_frame)

    def test_stored_candidate_counts_and_totals_must_match_candidate_rows(self):
        for column in ('candidates', 'candidate_targets'):
            broken = self.panel.with_columns((pl.col(column) + 1).alias(column))
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'stored candidate'):
                conservation(broken, self.candidates_frame)
        broken = self.panel.with_columns(pl.lit(None).alias('candidates'))
        with self.assertRaisesRegex(ValueError, 'stored candidate'):
            conservation(broken, self.candidates_frame)

    def test_orphan_candidate_groups_and_duplicate_team_games_are_refused(self):
        orphan = self.candidates_frame.head(1).with_columns(pl.lit('not_in_panel').alias('game_id'))
        with self.assertRaisesRegex(ValueError, 'no matching team game'):
            conservation(self.panel, pl.concat([self.candidates_frame, orphan]))
        with self.assertRaisesRegex(ValueError, 'duplicate key'):
            conservation(pl.concat([self.panel, self.panel.head(1)]), self.candidates_frame)


class Panel(unittest.TestCase):
    def test_the_panel_carries_the_declared_columns(self):
        panel, candidates_frame = panel_frame()
        self.assertEqual(panel.columns, [
            "season", "week", "game_date", "game_id", "team", "opponent", "is_home", "targets",
            "candidates", "candidate_targets", "other_targets"])
        self.assertEqual(candidates_frame.columns, [
            "season", "week", "game_id", "team", "candidate_index", "player_id", "position", "targets"])

    def test_the_other_label_is_never_a_candidate_row(self):
        _, candidates_frame = panel_frame()
        self.assertNotIn(OTHER, candidates_frame["player_id"].to_list())

    def test_the_panel_refuses_to_return_a_frame_that_lost_a_target(self):
        panel, candidates_frame = panel_frame()
        self.assertEqual(int(panel["targets"].sum()), 6)
        self.assertEqual(int(panel["candidate_targets"].sum()), int(candidates_frame["targets"].sum()))
        self.assertEqual(ACTIVE, "ACT")
