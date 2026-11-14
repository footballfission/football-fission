"""Evidence audits for the canonical inference tables.

An audit reports how much a table can support: how many plays per season, how much of each
column is missing, and whether the passer-receiver graph lets two players be separated at all.
It reads the table and never rewrites it.
"""
from __future__ import annotations

import statistics

import polars as pl

from .contracts import PASSING_CONTRACT, validate


def _share(nulls, rows):
    return round(nulls / rows, 4) if rows else None


def table_coverage(table, contract):
    """Row counts by season and the null share of every column the table carries."""
    validate(table, contract)
    columns = {}
    for column in table.columns:
        nulls = int(table[column].null_count())
        columns[column] = {"nulls": nulls, "share": _share(nulls, table.height)}
    seasons = {}
    for season in sorted(table["season"].unique().to_list()):
        block = table.filter(pl.col("season") == season)
        seasons[str(int(season))] = {"rows": block.height}
    return {
        "table": contract.name,
        "rows": table.height,
        "seasons": seasons,
        "columns": columns,
        "present_optional_columns": [c for c in contract.optional_columns if c in table.columns],
    }


def _components(pairs):
    """Connected components of the passer-receiver graph, as node sets."""
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for passer, receiver in pairs:
        union(passer, receiver)
    grouped = {}
    for node in list(parent):
        grouped.setdefault(find(node), set()).add(node)
    return list(grouped.values())


def pairing_support(table, *, contract=PASSING_CONTRACT):
    """Exposure per player and per pairing, and the connectedness of the pairing graph.

    A passer and a receiver who only ever appear together are one unknown quantity rather than
    two effects, and the component count is what makes that visible.
    """
    validate(table, contract)
    passers = set(table["qb_id"].unique().to_list())
    receivers = set(table["receiver_id"].unique().to_list())
    pairings = table.group_by(["qb_id", "receiver_id"]).len()
    components = _components(pairings.select(["qb_id", "receiver_id"]).iter_rows())
    described = []
    for nodes in components:
        block = table.filter(pl.col("qb_id").is_in(list(nodes)) | pl.col("receiver_id").is_in(list(nodes)))
        described.append({
            "players": len(nodes),
            "quarterbacks": len(nodes & passers),
            "receivers": len(nodes & receivers),
            "pairings": int(pairings.filter(
                pl.col("qb_id").is_in(list(nodes)) & pl.col("receiver_id").is_in(list(nodes))).height),
            "rows": block.height,
        })
    described.sort(key=lambda entry: (-entry["rows"], -entry["players"]))
    largest = described[0] if described else None
    exposure = pairings.group_by("receiver_id").len()["len"].to_list()
    return {
        "table": contract.name,
        "rows": table.height,
        "quarterbacks": len(passers),
        "receivers": len(receivers),
        "pairings": pairings.height,
        "pairings_per_receiver": {
            "min": min(exposure, default=0),
            "median": statistics.median(exposure) if exposure else None,
            "max": max(exposure, default=0),
        },
        "receivers_with_one_quarterback": sum(
            1 for count in pairings.group_by("receiver_id").len()["len"].to_list() if count == 1),
        "quarterbacks_with_one_receiver": sum(
            1 for count in pairings.group_by("qb_id").len()["len"].to_list() if count == 1),
        "components": len(described),
        "single_pairing_components": sum(1 for entry in described if entry["pairings"] == 1),
        "largest_component": largest,
        "largest_component_row_share": _share(largest["rows"], table.height) if largest else None,
        "components_detail": described,
    }
