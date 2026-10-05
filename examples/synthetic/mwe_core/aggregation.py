"""Aggregate paired synthetic Monte Carlo draws without losing sparse columns.

Source adaptation
-----------------
The column-completeness fix follows the corrected study implementation in
``submission/EIAR/r1/figure_recalculation_20261001/code/aggregation/``
``reaggregate_paper_targets.py`` (``setup``, ``parse_header``, ``process``).
That implementation builds a complete HMI schema before accumulation. This
standalone example constructs the equivalent schema from the union of ALL
building frames, because synthetic buildings can omit their all-zero columns.

Material screening preserves ``_merge_material_name`` from the original
repository-relative ``stock_modelling/proof_of_concept_uncertainty/``
``variance_decomposition/processing_files.py`` with all six Figure 3 groups
enabled and ``timber_exclude_exact=True``. Unselected materials are retained in
``other``. No private files or original hard-coded data paths are imported.

Input values are material masses, normally kg. Row k must represent the SAME
Monte Carlo realisation in every building; aggregation never reorders rows.
"""

from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype


LAYERS = ("structure", "skin", "space", "services")
MATERIALS = ("brick", "concrete", "stone", "timber", "metal", "other", "glass")


def _canonical(value: object) -> str:
    """Match the original screening function's case and whitespace handling."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def classify_material(name: object) -> str:
    """Screen a TERMINAL material name using the original Figure 3 rules.

    Rule priority is timber, stone, metal, concrete, brick, glass. A name that
    is exactly ``Timber`` is excluded from the selected timber group. Bare
    ``slate`` and ``plywood`` also enter ``other``. The function is deliberately
    not a chemical taxonomy and must not receive an entire construction path.
    """
    material = _canonical(name)
    if "timber" in material and material != "timber":
        return "timber"
    if any(token in material for token in ("stone", "granite", "ballast")):
        return "stone"
    if any(token in material for token in ("metal", "zinc", "steel", "copper", "brass", "aluminium")):
        return "metal"
    if "concrete" in material:
        return "concrete"
    if "brick" in material:
        return "brick"
    if "glass" in material:
        return "glass"
    return "other"


def _path_parts(column: object) -> tuple[str, str]:
    if not isinstance(column, str):
        raise ValueError("Every material column must be a six-level string path.")
    parts = column.split("///")
    if len(parts) != 6 or any(not part.strip() for part in parts):
        raise ValueError(f"Expected six non-empty path levels separated by '///': {column!r}")
    layer = _canonical(parts[0])
    if layer not in LAYERS:
        raise ValueError(f"Unrecognised building layer {parts[0]!r} in {column!r}")
    return layer, parts[-1]


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    if len(frame) == 0:
        raise ValueError(f"{label} must contain at least one draw.")
    index = frame.index
    if (
        not index.is_unique
        or not is_integer_dtype(index.dtype)
        or is_bool_dtype(index.dtype)
        or not np.array_equal(index.to_numpy(), np.arange(len(frame)))
    ):
        raise ValueError(f"{label} draw index must be unique and ordered 0..K-1.")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} contains duplicate material columns.")
    for column in frame.columns:
        _path_parts(column)
        dtype = frame[column].dtype
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
            raise ValueError(f"{label} has non-real-numeric material masses in {column!r}.")
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite material masses.")
    if (values < 0).any():
        raise ValueError(f"{label} contains negative material masses.")


def aggregate_draws(
    buildings: dict[str, pd.DataFrame], building_to_area: dict[str, str]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Sum building draws into area and city frames using a complete column union.

    Every building must have an area assignment and the same ordered draw index
    ``0..K-1``. Additional mapping entries are harmless and ignored. Missing
    material columns in a building mean zero mass; missing/NaN cells do not.
    Material columns follow first appearance across ALL buildings, and areas
    follow first appearance in ``buildings``. All output frames share the same
    column order and draw index. Inputs are not modified.

    Returns raw six-level-path mass frames, not target summaries. Each building
    enters exactly one area and the city exactly once.
    """
    if not isinstance(buildings, Mapping) or not buildings:
        raise ValueError("buildings must be a non-empty mapping of IDs to draw frames.")
    if not isinstance(building_to_area, Mapping):
        raise TypeError("building_to_area must be a mapping.")
    missing = [building for building in buildings if building not in building_to_area]
    if missing:
        raise ValueError(f"Missing area assignments for building IDs: {missing!r}")

    columns: list[str] = []
    seen: set[str] = set()
    draw_index = None
    areas: list[str] = []
    for building, frame in buildings.items():
        if not isinstance(building, str) or not building.strip():
            raise ValueError("Every building ID must be a non-empty string.")
        area = building_to_area[building]
        if not isinstance(area, str) or not area.strip():
            raise ValueError(f"Building {building!r} needs a non-empty string area ID.")
        _validate_frame(frame, f"Building {building!r}")
        if draw_index is None:
            draw_index = frame.index.copy()
        elif not np.array_equal(frame.index.to_numpy(), draw_index.to_numpy()):
            raise ValueError(f"Building {building!r} has misaligned draw indices.")
        if area not in areas:
            areas.append(area)
        for column in frame.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)

    assert draw_index is not None
    shape = (len(draw_index), len(columns))
    sums = {area: np.zeros(shape, dtype=float) for area in areas}
    with np.errstate(over="ignore", invalid="ignore"):
        for building, frame in buildings.items():
            aligned = frame.reindex(columns=columns, fill_value=0).to_numpy(dtype=float)
            sums[building_to_area[building]] += aligned
        city_values = np.zeros(shape, dtype=float)
        for values in sums.values():
            city_values += values
    if not all(np.isfinite(values).all() for values in sums.values()) or not np.isfinite(city_values).all():
        raise ValueError("Aggregation overflowed to non-finite material masses.")
    area_frames = {
        area: pd.DataFrame(values, index=draw_index.copy(), columns=columns)
        for area, values in sums.items()
    }
    city = pd.DataFrame(city_values, index=draw_index.copy(), columns=columns)
    return area_frames, city


def summarise_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Return paired total, four layer, and seven material mass series.

    Layer comes from path level 1; material screening reads level 6 only. Each
    input mass belongs to exactly one layer and exactly one material group.
    ``total`` is computed from the raw masses once, never by summing overlapping
    total/layer/material summaries. Thus the four layer columns and, separately,
    the seven material columns each sum to ``total`` on every draw.
    """
    _validate_frame(frame, "Target input")
    values = frame.to_numpy(dtype=float)
    paths = [_path_parts(column) for column in frame.columns]
    targets: dict[str, np.ndarray] = {"total": values.sum(axis=1)}
    for layer in LAYERS:
        select = [i for i, (path_layer, _) in enumerate(paths) if path_layer == layer]
        targets[f"layer__{layer}"] = values[:, select].sum(axis=1)
    for group in MATERIALS:
        select = [i for i, (_, material) in enumerate(paths) if classify_material(material) == group]
        targets[f"material__{group}"] = values[:, select].sum(axis=1)
    result = pd.DataFrame(targets, index=frame.index.copy())
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Target summation overflowed to non-finite material masses.")
    return result
