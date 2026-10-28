"""Offline checks for writing the canonical inference tables and their manifest."""
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from hierarchical_football.data.store import content_hash, write_frame
from hierarchical_football.inference.contracts import PASSING_CONTRACT, RUSHING_CONTRACT, validate
from hierarchical_football.inference.prepare import build

PLAY = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, season_type="REG", posteam="A",
    defteam="B", play_type="pass", two_point_attempt=0.0, aborted_play=0.0, sack=0.0,
    qb_scramble=0.0, pass_attempt=1.0, passer_player_id="qb1", receiver_player_id="wr1",
    complete_pass=1.0, air_yards=8.0, yardline_100=40.0, ydstogo=10.0, down=1.0,
    pass_location="left", rush_attempt=0.0, qb_kneel=0.0, qb_spike=0.0,
    rusher_player_id=None, rushing_yards=None, run_location=None, run_gap=None,
)

PBP_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, season_type=pl.String,
    posteam=pl.String, defteam=pl.String, play_type=pl.String, two_point_attempt=pl.Float64,
    aborted_play=pl.Float64, sack=pl.Float64, qb_scramble=pl.Float64, pass_attempt=pl.Float64,
    passer_player_id=pl.String, receiver_player_id=pl.String, complete_pass=pl.Float64,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Float64,
    pass_location=pl.String, rush_attempt=pl.Float64, qb_kneel=pl.Float64, qb_spike=pl.Float64,
    rusher_player_id=pl.String, rushing_yards=pl.Float64, run_location=pl.String,
    run_gap=pl.String,
)

ROSTER_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, team=pl.String, gsis_id=pl.String, position=pl.String,
    game_type=pl.String,
)

RUN = dict(play_type="run", pass_attempt=0.0, rush_attempt=1.0, rusher_player_id="rb1",
           rushing_yards=4.0, run_location="left", run_gap="guard")


def season_plays(season):
    rows = [{"play_id": 1}, {"play_id": 2, "complete_pass": 0.0},
            {"play_id": 3, **RUN}, {"play_id": 4, "qb_kneel": 1.0, **RUN}]
    return [{**PLAY, **row, "season": season, "game_id": f"{season}_01_A_B"} for row in rows]


def write_raw(root, seasons=(2024, 2025), with_rosters=True):
    for season in seasons:
        write_frame(root / "pbp" / f"season={season}" / "part.parquet",
                    pl.DataFrame(season_plays(season), schema_overrides=PBP_SCHEMA))
        if with_rosters:
            rosters = [{"gsis_id": "qb1", "position": "QB"}, {"gsis_id": "wr1", "position": "WR"},
                       {"gsis_id": "rb1", "position": "HB"}]
            write_frame(root / "rosters_weekly" / f"season={season}" / "part.parquet",
                        pl.DataFrame([{**{"season": season, "week": 1, "team": "A", "game_type": "REG"},
                                        **row} for row in rosters], schema_overrides=ROSTER_SCHEMA))


@contextmanager
def workspace():
    parent = Path(__file__).resolve().parent
    folder = parent / ("inference_build_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError("unexpected cleanup path")
        shutil.rmtree(folder)


class Build(unittest.TestCase):
    def build(self, root, **kwargs):
        return build(root / "raw", root / "inference", kwargs.pop("seasons", (2024, 2025)), **kwargs)

    def test_both_tables_and_the_report_are_written(self):
        with workspace() as root:
            write_raw(root / "raw")
            report = self.build(root)
            for name, contract in (("passing_targets", PASSING_CONTRACT),
                                   ("rushing_plays", RUSHING_CONTRACT)):
                table = pl.read_parquet(root / "inference" / f"{name}.parquet")
                self.assertTrue(validate(table, contract).equals(table))
                self.assertEqual(table.height, report["passing" if name == "passing_targets" else "rushing"]["preparation"]["target_rows"])
            self.assertEqual(json.loads((root / "inference" / "prepare_report.json").read_text())["seasons"],
                             [2024, 2025])

    def test_report_names_every_input_partition_with_its_hash(self):
        with workspace() as root:
            write_raw(root / "raw")
            report = self.build(root)
            self.assertEqual([entry["table_partition"] for entry in report["inputs"]], [
                "pbp/season=2024/part.parquet", "pbp/season=2025/part.parquet",
                "rosters_weekly/season=2024/part.parquet", "rosters_weekly/season=2025/part.parquet"])
            for entry in report["inputs"]:
                self.assertEqual(entry["sha256"],
                                 content_hash(root / "raw" / entry["table_partition"]))

    def test_report_carries_the_audits_of_both_tables(self):
        with workspace() as root:
            write_raw(root / "raw")
            report = self.build(root)
            self.assertEqual(report["passing"]["coverage"]["rows"], 4)
            self.assertEqual(report["passing"]["pairing_support"]["pairings"], 1)
            self.assertEqual(report["passing"]["pairing_support"]["receivers_with_one_quarterback"], 1)
            self.assertEqual(report["rushing"]["coverage"]["rows"], 2)
            self.assertEqual(report["rushing"]["preparation"]["rusher_scope"], "backfield")

    def test_written_sidecar_records_rows_and_hash(self):
        with workspace() as root:
            write_raw(root / "raw")
            self.build(root)
            table_path = root / "inference" / "passing_targets.parquet"
            sidecar = json.loads(table_path.with_suffix(".json").read_text())
            self.assertEqual(sidecar["rows"], 4)
            self.assertEqual(sidecar["sha256"], content_hash(table_path))

    def test_roster_positions_reach_the_tables(self):
        with workspace() as root:
            write_raw(root / "raw")
            self.build(root)
            table = pl.read_parquet(root / "inference" / "rushing_plays.parquet")
            self.assertEqual(table["position"].to_list(), ["RB", "RB"])

    def test_missing_play_by_play_partition_is_refused(self):
        with workspace() as root:
            write_raw(root / "raw")
            with self.assertRaises(FileNotFoundError):
                self.build(root, seasons=(2023,))

    def test_missing_rosters_are_refused_for_a_position_scope(self):
        with workspace() as root:
            write_raw(root / "raw", with_rosters=False)
            with self.assertRaisesRegex(ValueError, "needs cached rosters_weekly"):
                self.build(root)

    def test_missing_rosters_are_allowed_for_the_all_scope(self):
        with workspace() as root:
            write_raw(root / "raw", with_rosters=False)
            report = self.build(root, rusher_scope="all")
            self.assertFalse(report["roster_positions_joined"])
            self.assertEqual(report["rushing"]["preparation"]["target_rows"], 2)
            table = pl.read_parquet(root / "inference" / "rushing_plays.parquet")
            self.assertEqual(table["position"].unique().to_list(), ["UNKNOWN"])

    def test_no_seasons_is_refused(self):
        with workspace() as root:
            write_raw(root / "raw")
            with self.assertRaisesRegex(ValueError, "at least one season"):
                self.build(root, seasons=())


if __name__ == "__main__":
    unittest.main()
