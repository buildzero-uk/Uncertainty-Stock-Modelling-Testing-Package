"""Original age/prototype-slot sampling with file-independent imports.

The caller supplies row-conditional probabilities. The example's CSV stores
sampled ages in rows and observed ages in columns, so it is transposed on load.
"""
from typing import Sequence, Any, List
from copy import deepcopy
import numpy as np


def generate_slots_for_age_material(
    n_slots: int,
    age_list: Sequence[Any],
    material_types: Sequence[Any],
    age_confusion_matrix: Sequence[Sequence[float]],
    tree_matrix: Sequence[Sequence[Sequence[Any]]],
    seed: int,
    specific_age: Any,
    specific_material: Any,
) -> tuple[list[Any], list[Any]]:
    """
    Monte Carlo sampling for a specific (age, material type).

    Returns
    -------
    sampled_trees : list  (length n_slots)
        A flat 1D list of sampled trees, sampled according to the confusion matrix.
    """
    # --- map labels -> indices ---
    try:
        age_idx = age_list.index(specific_age)
    except ValueError:
        raise ValueError(f"specific_age {specific_age!r} not found in age_list")
    # material type none means lazy forest
    if specific_material != 'lazy':
        try:
            material_idx = material_types.index(specific_material)
        except ValueError:
            raise ValueError(f"specific_material {specific_material!r} not found in material_types")
    else:
        material_idx = 0

    # --- extract probability vector (already normalized) ---
    probs = np.asarray(age_confusion_matrix, dtype=float)[age_idx]
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(
            f"Confusion row for age {specific_age!r} must sum to 1.0, got {probs.sum()}"
        )

    n_ages = len(age_list)

    # --- RNG ---
    rng = np.random.default_rng(seed)

    # --- sample target ages ---
    sampled_target_age_indices = rng.choice(
        n_ages,
        size=n_slots,
        p=probs
    )
    # print(sampled_target_age_indices)
    # print([i for i, sub in enumerate(tree_matrix) if len(sub) == 0])
    # --- sample trees directly into final flat list ---
    sampled_trees: List[Any] = []
    for target_age_idx in sampled_target_age_indices:
        trees = tree_matrix[target_age_idx][material_idx]
        if not trees:
            raise ValueError(
                f"No trees available for (age={age_list[target_age_idx]!r}, "
                f"material={specific_material!r})."
            )
        chosen_idx = rng.integers(len(trees))
        sampled_trees.append(deepcopy(trees[chosen_idx]))

    return sampled_trees, sampled_target_age_indices

