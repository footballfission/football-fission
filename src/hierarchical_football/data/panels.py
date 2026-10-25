"""Roster-wide canonical rows, team targets, and model-ready position panels."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from .. import scoring
from .history import add_lagged_player_features
from .reports import latest_known_injury_status
from .schema import (BLOCK_COLUMNS, KEYS, TEAM_KEYS, PLAYER_OUTCOMES, POSITION_COLUMNS,
                     LAG_SOURCE_COLUMNS, model_features)
from .store import atomic_json, content_hash, read_table, utc_now, write_frame


def required(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def unique(frame, keys, name):
    required(frame, keys, name)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: null or duplicate keys {keys}")


def normalize(frame, *, roster=False):
    frame = frame.copy()
    aliases = {"gsis_id": "player_id", "recent_team": "team", "passing_interceptions": "interceptions"}
    for source, destination in aliases.items():
        if destination not in frame and source in frame:
            frame = frame.rename(columns={source: destination})
    for name in ("season", "week"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype("int64")
    kind = "game_type" if roster else "season_type"
    required(frame, [kind], "roster" if roster else "player statistics")
    return frame.loc[frame[kind].eq("REG")].copy()


def schedule_rows(schedules, *, prediction_time=None, as_of=None):
    required(schedules, ["season", "week", "game_type", "game_id", "home_team", "away_team",
                         "gameday", "gametime", "home_score", "away_score"], "schedules")
    schedules = schedules.loc[schedules.game_type.eq("REG")].copy()
    unique(schedules, ["game_id"], "schedules")
    # Provider gametime is America/New_York; store naive UTC for consistent NumPy history math.
    local = pd.to_datetime(schedules.gameday.astype(str) + " " + schedules.gametime.astype(str), format="%Y-%m-%d %H:%M", errors="coerce")
    schedules["kickoff"] = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    if schedules.kickoff.isna().any():
        raise ValueError("Schedules have missing/unparseable kickoff times; no fabricated time fallback")
    default_cutoff = local.dt.normalize() - pd.to_timedelta((local.dt.weekday - 2) % 7, unit="D")
    schedules["prediction_timestamp"] = default_cutoff.dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None)
    as_of = pd.Timestamp(as_of or utc_now())
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    limit = as_of.tz_convert("UTC").tz_localize(None)
    schedules["game_completed"] = (schedules.home_score.notna() & schedules.away_score.notna()
        & ((schedules.kickoff + pd.Timedelta(hours=4)) < limit))
    if prediction_time is not None:
        stamp = pd.Timestamp(prediction_time)
        if stamp.tzinfo is None:
            raise ValueError("Prediction timestamp must include a timezone")
        schedules.loc[~schedules.game_completed, "prediction_timestamp"] = stamp.tz_convert("UTC").tz_localize(None)
    if (schedules.prediction_timestamp >= schedules.kickoff).any():
        raise ValueError("Prediction timestamps must precede kickoff for every requested game")
    result = []
    for side, other in (("home", "away"), ("away", "home")):
        columns = ["season", "week", "game_id", "kickoff", "prediction_timestamp", "game_completed"]
        block = schedules[columns].copy()
        block["team"] = schedules[f"{side}_team"]
        block["opponent"] = schedules[f"{other}_team"]
        block["is_home"] = side == "home"
        block["points_allowed"] = pd.to_numeric(schedules[f"{other}_score"], errors="coerce").where(block.game_completed)
        result.append(block)
    rows = pd.concat(result, ignore_index=True)
    unique(rows, TEAM_KEYS, "schedule team-games")
    return rows


def _attach_snaps(players, snaps, ids):
    if snaps.empty:
        return players
    required(snaps, ["pfr_player_id", "season", "week", "team", "offense_snaps", "offense_pct"], "snap counts")
    required(ids, ["pfr_id", "gsis_id"], "player ID mapping")
    if "game_type" in snaps:
        snaps = snaps.loc[snaps.game_type.eq("REG")].copy()
    cross = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()
    # Ambiguous mappings remain unmatched instead of assigning another player's snaps.
    cross = cross.loc[~cross.duplicated("pfr_id", keep=False)]
    snaps = snaps.merge(cross, left_on="pfr_player_id", right_on="pfr_id", how="left", validate="many_to_one")
    snaps = snaps.rename(columns={"gsis_id": "player_id", "offense_pct": "offense_snap_pct"})
    snaps = snaps.loc[snaps.player_id.notna(), KEYS + ["offense_snaps", "offense_snap_pct"]]
    unique(snaps, KEYS, "snap counts")
    return players.merge(snaps, on=KEYS, how="left", validate="one_to_one")


def _extend_rosters(rosters, context):
    """Copy each team's latest snapshot into later scheduled weeks that lack one."""
    scheduled = context[TEAM_KEYS].drop_duplicates()
    covered = rosters[TEAM_KEYS].drop_duplicates()
    missing = scheduled.merge(covered, on=TEAM_KEYS, how="left", indicator=True)
    missing = missing.loc[missing._merge.eq("left_only"), TEAM_KEYS]
    blocks = []
    for season, week, team in missing.itertuples(index=False):
        history = rosters.loc[rosters.season.eq(season) & rosters.team.eq(team) & rosters.week.lt(week)]
        if history.empty:
            continue
        block = history.loc[history.week.eq(history.week.max())].copy()
        block["week"] = week
        block["roster_carried_forward"] = True
        blocks.append(block)
    rosters = rosters.assign(roster_carried_forward=False)
    if not blocks:
        return rosters, 0, 0
    extended = pd.concat([rosters, *blocks], ignore_index=True)
    return extended, sum(len(block) for block in blocks), len(blocks)


def assemble(schedules, stats, rosters, *, snaps=None, ids=None, zero_missing=False,
             extend_rosters=False, prediction_time=None, as_of=None):
    """Pure transformation entrypoint. Missing stats remain unknown unless policy is explicit."""
    context = schedule_rows(schedules, prediction_time=prediction_time, as_of=as_of)
    stats = normalize(stats)
    rosters = normalize(rosters, roster=True)
    carried_rows, extended_weeks = 0, 0
    if extend_rosters:
        rosters, carried_rows, extended_weeks = _extend_rosters(rosters, context)
    else:
        rosters = rosters.assign(roster_carried_forward=False)
    required(rosters, KEYS, "weekly rosters")
    # Rosters without a stable ID cannot be linked; report and retain their count separately.
    missing_roster_ids = int(rosters.player_id.isna().sum())
    rosters = rosters.loc[rosters.player_id.notna()].copy()
    required(stats, KEYS, "player statistics")
    unique(stats.loc[stats.player_id.notna()], KEYS, "identified player statistics")
    if stats[TEAM_KEYS].isna().any().any():
        raise ValueError("Player statistics have unknown team/week keys")
    unique(rosters, KEYS, "weekly rosters")
    required(stats, ["position"], "player statistics")
    required(rosters, ["position"], "weekly rosters")
    source_columns = [c for c in PLAYER_OUTCOMES if c in stats]
    for column in source_columns:
        stats[column] = pd.to_numeric(stats[column], errors="raise")
    # Verify non-null source counters rather than turning nulls into recorded zero.
    if stats[source_columns].isna().any().any():
        raise ValueError("Player statistic counters contain nulls; audit incomplete source rows")
    for column in source_columns:
        if "yards" not in column and stats[column].lt(0).any():
            raise ValueError(f"Negative count in {column}")
    if {"receptions", "targets"}.issubset(stats) and stats.receptions.gt(stats.targets).any():
        raise ValueError("Receptions exceed targets")
    metadata = [c for c in ("position", "status", "full_name", "roster_carried_forward") if c in rosters]
    roster = rosters[KEYS + metadata].rename(columns={"position": "roster_position", "status": "roster_status"})
    roster["roster_present"] = True
    selected = stats.loc[stats.player_id.notna(), KEYS + ["position"] + source_columns].copy()
    selected["stat_record_present"] = True
    players = roster.merge(selected, on=KEYS, how="outer", validate="one_to_one")
    players["position"] = players.position.combine_first(players.roster_position)
    players = players.drop(columns="roster_position")
    players["position"] = players.position.replace({"HB": "RB", "FB": "RB"})
    players["roster_present"] = players.roster_present.eq(True)
    players["stat_record_present"] = players.stat_record_present.eq(True)
    players["roster_carried_forward"] = players.roster_carried_forward.eq(True)
    unmatched = players.merge(context[TEAM_KEYS], on=TEAM_KEYS, how="left", indicator=True)
    unmatched_rows = int(unmatched._merge.eq("left_only").sum())
    players = players.merge(context.drop(columns="points_allowed"), on=TEAM_KEYS, how="inner", validate="many_to_one")
    source_teams = stats.groupby(TEAM_KEYS, as_index=False).size().rename(columns={"size": "source_player_rows"})
    players = players.merge(source_teams, on=TEAM_KEYS, how="left", validate="many_to_one")
    assumed = (~players.stat_record_present & players.game_completed & players.source_player_rows.gt(0) & zero_missing)
    players["outcome_status"] = np.select([players.game_completed & players.stat_record_present, assumed],
        ["recorded", "assumed_zero"], default="unknown")
    for column in source_columns:
        players.loc[assumed, column] = 0.0
        players.loc[~players.game_completed, column] = np.nan
    players["history_eligible"] = players.outcome_status.ne("unknown")
    if snaps is not None and not snaps.empty:
        players = _attach_snaps(players, snaps, ids if ids is not None else pd.DataFrame())
    # True targetable pool: sum ALL source players, not just fantasy positions or current roster matches.
    totals = stats.groupby(TEAM_KEYS)[[c for c in ("targets", "carries", *POSITION_COLUMNS["DST"]) if c in stats]].sum(min_count=1).reset_index()
    totals = totals.rename(columns={"targets": "team_targets", "carries": "team_carries"})
    teams = context.merge(totals, on=TEAM_KEYS, how="left", validate="one_to_one", suffixes=("", "_stat"))
    for column in [c for c in teams if c not in context.columns]:
        teams.loc[~teams.game_completed, column] = np.nan
    for quantity in ("targets", "carries"):
        total = f"team_{quantity}"
        if quantity not in players or total not in teams:
            raise ValueError(f"Missing required team opportunity counter: {quantity}")
        players = players.merge(teams[TEAM_KEYS + [total]], on=TEAM_KEYS, how="left", validate="many_to_one")
        players[f"{quantity[:-1] if quantity == 'targets' else 'carry'}_share"] = players[quantity] / players[total].replace(0, np.nan)
        covered = players.loc[players.roster_present & players.stat_record_present].groupby(TEAM_KEYS)[quantity].sum()
        coverage = covered.rename(f"roster_{quantity}").reset_index()
        teams = teams.merge(coverage, on=TEAM_KEYS, how="left", validate="one_to_one")
        # Residual is outside roster coverage, not an invented allocation to known teammates.
        teams[f"residual_{quantity}"] = teams[total] - teams[f"roster_{quantity}"].fillna(0)
    if set(BLOCK_COLUMNS).issubset(stats.columns):
        blocks = stats.groupby(TEAM_KEYS)[list(BLOCK_COLUMNS)].sum().sum(axis=1).rename("blocked_kicks").reset_index()
        teams = teams.merge(blocks, on=TEAM_KEYS, how="left", validate="one_to_one")
        teams.loc[~teams.game_completed, "blocked_kicks"] = np.nan
    report = dict(player_rows=len(players), team_rows=len(teams), missing_roster_ids=missing_roster_ids,
                  unidentified_stat_rows=int(stats.player_id.isna().sum()),
                  unmatched_schedule_rows=unmatched_rows, assumed_zero_rows=int(assumed.sum()),
                  carried_forward_roster_rows=carried_rows, extended_roster_team_weeks=extended_weeks,
                  unknown_outcome_rows=int(players.outcome_status.eq("unknown").sum()),
                  raw_asof_verified=False, missing_source_outcome_columns=sorted(set(PLAYER_OUTCOMES)-set(source_columns)))
    if "offense_snaps" in players:
        report["recorded_players_without_snap_match"] = int((players.stat_record_present & players.offense_snaps.isna()).sum())
        cross = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates()
        report["ambiguous_pfr_ids_excluded"] = int(cross.loc[cross.duplicated("pfr_id", keep=False), "pfr_id"].nunique())
    return pl.from_pandas(players), pl.from_pandas(teams), report


def position_panel(players, teams, position):
    columns = POSITION_COLUMNS[position]
    if position == "DST":
        frame = teams.with_columns(pl.col("team").alias("player_id"), pl.lit("DST").alias("position"),
                                   pl.col("game_completed").alias("history_eligible"))
    else:
        frame = players.filter(pl.col("position") == position)
    required(frame, columns, position)
    usable = pl.all_horizontal([pl.col(c).is_not_null() for c in columns])
    frame = frame.with_columns((usable & pl.col("game_completed")).alias("outcome_available"))
    if position in ("WR", "RB", "TE"):
        fn = scoring.scrimmage(1.0)
    elif position == "QB":
        fn = scoring.passing(True)
    elif position == "K":
        fn = scoring.kicking
    else:
        fn = scoring.defense
    values = frame.select(columns).to_numpy().astype(float)
    frame = frame.with_columns(pl.Series("fantasy_points", fn(values, columns)))
    frame = frame.with_columns(pl.when(pl.col("outcome_available")).then(pl.col("fantasy_points")).otherwise(None).alias("fantasy_points"))
    sources = tuple(dict.fromkeys((*columns, "offense_snaps", "offense_snap_pct",
                                   "target_share", "carry_share", "fantasy_points")))
    return add_lagged_player_features(frame, sources=sources)


def build(raw: Path, output: Path, seasons, *, positions=("WR", "RB", "TE", "QB", "K", "DST"),
          zero_missing=False, extend_rosters=False, prediction_time=None, as_of=None):
    if not seasons or not positions or set(positions) - set(POSITION_COLUMNS):
        raise ValueError("Specify seasons and supported positions")
    as_of = as_of or utc_now()
    schedules = read_table(raw, "schedules", seasons)
    stats = read_table(raw, "player_stats_week", seasons)
    rosters = read_table(raw, "rosters_weekly", seasons)
    snaps = read_table(raw, "snap_counts", seasons, required=False)
    ids = read_table(raw, "ff_playerids", seasons, required=False)
    # Static ID tables may repeat across cached seasons.
    if not ids.is_empty():
        ids = ids.drop("season", strict=False).unique()
    players, teams, report = assemble(schedules.to_pandas(), stats.to_pandas(), rosters.to_pandas(),
        snaps=snaps.to_pandas(), ids=ids.to_pandas(), zero_missing=zero_missing,
        extend_rosters=extend_rosters, prediction_time=prediction_time, as_of=as_of)
    injuries = read_table(raw, "injuries", seasons, required=False)
    players = latest_known_injury_status(players, injuries)
    report["candidate_injury_report_matches"] = int(players["candidate_report_time"].is_not_null().sum())
    team_sources = tuple(c for c in ("team_targets", "team_carries", "points_allowed") if c in teams.columns)
    team_history = teams.with_columns(pl.col("team").alias("player_id"),
        (pl.col("game_completed") & pl.col("team_targets").is_not_null()).alias("history_eligible"))
    team_history = add_lagged_player_features(team_history, sources=team_sources)
    team_features = [c for c in team_history.columns if any(c == f"{source}_{horizon}" for source in team_sources
        for horizon in ("last", "ewma_short", "ewma_med", "last16", "season_1", "season_2", "season_3"))]
    teams = team_history.drop("player_id")
    players = players.join(teams.select(*TEAM_KEYS, *team_features), on=TEAM_KEYS, how="left", validate="m:1")
    report.update(built_at=utc_now(), seasons=list(seasons), zero_missing_policy=zero_missing,
                  extend_rosters=extend_rosters, as_of=as_of, prediction_time=prediction_time,
                  positions={}, inputs=[])
    used_tables = {"schedules", "player_stats_week", "rosters_weekly", "snap_counts", "ff_playerids",
                   "injuries", "depth_charts", "team_stats_week"}
    for path in sorted(raw.rglob("part.parquet")):
        relative = path.relative_to(raw)
        if relative.parts[0] in used_tables and (any(f"season={s}" in path.parts for s in seasons)
                                                or relative.as_posix() == "ff_playerids/part.parquet"):
            report["inputs"].append({"table_partition": path.relative_to(raw).as_posix(),
                "sha256": content_hash(path)})
    report["status"] = "building"
    atomic_json(output / "build_report.json", report)
    write_frame(output / "player_games.parquet", players)
    write_frame(output / "team_games.parquet", teams)
    atomic_json(output / "team_features.json", [*team_features, "team", "opponent", "is_home", "week"])
    for position in positions:
        panel = position_panel(players, teams, position)
        features = tuple(dict.fromkeys((*model_features(panel.columns), *team_features)))
        train = panel.filter(pl.col("outcome_available"))
        prediction = panel.filter(~pl.col("game_completed"))
        write_frame(output / position / "history.parquet", train)
        write_frame(output / position / "prediction.parquet", prediction)
        atomic_json(output / position / "features.json", list(features))
        report["positions"][position] = {"history_rows": train.height, "prediction_rows": prediction.height}
    # Preserve reports/depth/team detail as facts; never silently add them to feature lists.
    for table in ("injuries", "depth_charts", "team_stats_week"):
        frame = read_table(raw, table, seasons, required=False)
        if not frame.is_empty():
            write_frame(output / "facts" / f"{table}.parquet", frame,
                        metadata={"model_feature_eligible": False, "historical_asof_verified": False})
    report["status"] = "complete"
    atomic_json(output / "build_report.json", report)
    return report
