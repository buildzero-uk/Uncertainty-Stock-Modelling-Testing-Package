"""Material-tree primitives reused from the research implementation.

See CODE_ORIGIN.md and provenance.json for source hashes and adaptation notes.
No file I/O is performed at import time.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Dict, List, Iterable, Tuple, Set
import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.children = {}
        self.mi = None  # Only set for leaf nodes
        self.parent = None

    def add_path(self, path, mi):
        node = self
        for level, name in enumerate(path):
            if name not in node.children:
                node.children[name] = TreeNode(name, level)
            node = node.children[name]
        node.mi = mi  # Set MI at leaf

    def add_child(self, child_node):
        self.children[child_node.name] = child_node
        child_node.parent = self

    def query(self, path):
        node = self
        for name in path:
            if name in node.children:
                node = node.children[name]
            else:
                return None
        return node.mi

    def to_dict(self):
        if self.mi is not None:
            return {"MI": self.mi}
        return {name: child.to_dict() for name, child in self.children.items()}


class MaterialTreeBuilder:
    def __init__(self, csv_path_or_df, age_bands):
        if isinstance(csv_path_or_df, str):
            self.df = pd.read_csv(csv_path_or_df, encoding='ISO-8859-1')
        else:
            self.df = csv_path_or_df

        self.age_bands = age_bands
        self.hierarchy = ['Layer', 'Function', 'Sub-Function', 'Technology', 'Specification', 'Material']

    def build_tree_for_age_band(self, age_band):
        if age_band not in self.age_bands:
            raise ValueError(f"Invalid age band: {age_band}")

        filtered_df = self.df[self.df[age_band] == 'YES']
        root = TreeNode("root", level=-1)

        for _, row in filtered_df.iterrows():
            path = [row[col] for col in self.hierarchy]
            mi = row['MI']
            root.add_path(path, mi)

        return root


class TreeTrimmer:
    def __init__(self):
        """
        tech_dict format:
        {
            "Wall": ["Brick Or Block Or Stone", "Concrete", "Mixed (Masonry And Metal)", "Mixed (Masonry And Timber)",
            "Other Non-Standard Or System Build", "Timber Or Wood"],
            "Roof": ["Tile Or Stone Or Slate", "Waterproof Membrane Or Concrete", "Metal"],
            "Roof_shape": ["flat", "pitched"]
        }
        """
        self.wall_route = {"Brick Or Block Or Stone": ['brick', 'stone', 'concrete'],
                           "Concrete": ['concrete'],
                           "Mixed (Masonry And Metal)": ['brick', 'stone', 'concrete', 'steel'],
                           "Mixed (Masonry And Timber)": ['brick', 'stone', 'concrete', 'timber'],
                           "Other Non-Standard Or System Build": ['brick', 'stone', 'concrete', 'timber'],
                           "Timber Or Wood": ['timber']}
        self.wall_search_space = ['non-load bearing structure', 'load bearing']
        self.roof_route = {"Tile Or Stone Or Slate": "tiles",
                           "Waterproof Membrane Or Concrete": "covering",
                           "Metal": "sheet metal"}
        self.roof_search_space = ['finish 2', 'intermediate finish 2']

    def trim_roof_tech(self, root: TreeNode, stock_info):
        """
        For sub-functions in `roof_search_space`, keep ONLY the target technology
        (derived from roof_route[stock_info['roof_material']]) if that target exists.
        If the target doesn't exist under a sub-function, do nothing for that sub-function.
        """

        def norm(s): return str(s).strip().lower()
        search_space_norm = {norm(x) for x in self.roof_search_space}

        # Resolve target technology name (normalized)
        roof_material_key = stock_info.get("roof_material")
        if roof_material_key is None:
            return 0  # nothing to do

        target_raw = self.roof_route.get(roof_material_key)
        if target_raw is None:
            return 0  # mapping missing, bail safely

        target_norm = norm(target_raw)

        trimmed_count = 0

        # Find all Roof function nodes across layers (e.g., Skin → Roof, etc.)
        roof_function_nodes = []
        for layer_node in getattr(root, "children", {}).values():
            fn = getattr(layer_node, "children", {}).get("Roof")
            if fn is not None:
                roof_function_nodes.append(fn)

        for roof_fn in roof_function_nodes:
            # Iterate sub-functions under Roof (e.g., Covering, Finish 2, etc.)
            for sub_fn in list(getattr(roof_fn, "children", {}).values()):
                if norm(getattr(sub_fn, "name", "")) not in search_space_norm:
                    continue

                # Build normalized name map for technologies under this sub-function
                # keys are actual dict keys; vals are normalized tech names
                tech_items = list(getattr(sub_fn, "children", {}).items())
                tech_norm_by_key = {
                    key: norm(getattr(node, "name", key))
                    for key, node in tech_items
                }

                # Check if target exists here
                has_target = any(n == target_norm for n in tech_norm_by_key.values())
                if not has_target:
                    continue  # leave as-is

                # Delete all technologies EXCEPT the target
                for key, node in tech_items:
                    tech_name = getattr(node, "name", key)
                    if tech_norm_by_key[key] != target_norm:
                        del sub_fn.children[key]
                        print(f"Deleted roof technology '{tech_name}' under Roof → {sub_fn.name}, kept '{target_raw}'")
                        trimmed_count += 1

        # print('process roof tiles')
        skin_node = root.children.get("Skin")
        if not skin_node:
            return

        roof_node = skin_node.children.get("Roof")
        if not roof_node:
            return

        if stock_info.get('roof_shape') == 'flat':
            target_techs4delete = ['sheet metal', 'tiles']

        elif stock_info.get('roof_shape') == 'pitched':
            target_techs4delete = ['screed', 'ballast']

        # Iterate through each sub-function under Roof
        for subfunc_node in roof_node.children.values():
            for tech_name in list(subfunc_node.children.keys()):
                if str(tech_name).strip().lower() in target_techs4delete:
                    del subfunc_node.children[tech_name]
                    print(f"Deleted technology node '{tech_name}' under Skin → Roof → {subfunc_node.name}")
        # print('finish trim roof tech')

    def trim_wall_tech(self, root: TreeNode, stock_info) -> int:
        """
        For sub-functions in `self.wall_search_space` under any Wall function:
          - Build allowed tokens from self.wall_route[stock_info['wall_material']]
          - If at least one technology's name contains any allowed token, delete all *other* techs.
          - If none match, do nothing for that sub-function.
        Prints deletions. Returns the number of deletions.
        """

        def norm(s: str) -> str:
            return str(s).strip().lower()

        def children(node):
            return getattr(node, "children", {})

        def nname(node, fallback):
            return getattr(node, "name", fallback)

        wall_material_key = stock_info.get("wall_material")
        if not wall_material_key:
            return 0
        allowed_raw = self.wall_route.get(wall_material_key)
        if not allowed_raw:
            return 0

        allowed_tokens = {norm(x) for x in allowed_raw}  # e.g., {"concrete", "brick", "stone"}
        search_space = {norm(x) for x in getattr(self, "wall_search_space", [])}
        trimmed = 0

        # Find Wall nodes (prefer Skin → Wall; else scan all layers)
        wall_nodes = []
        skin = children(root).get("Skin")
        if skin and children(skin).get("Wall"):
            wall_nodes.append(children(skin).get("Wall"))
        else:
            for layer in children(root).values():
                wn = children(layer).get("Wall")
                if wn is not None:
                    wall_nodes.append(wn)

        # Helper: does this tech match any allowed token?
        def is_allowed_tech(tech_label_norm: str) -> bool:
            return any(tok in tech_label_norm for tok in allowed_tokens)

        for wall_fn in wall_nodes:
            for sub_key, sub_fn in list(children(wall_fn).items()):
                sub_norm = norm(nname(sub_fn, sub_key))
                if sub_norm not in search_space:
                    continue

                tech_items = list(children(sub_fn).items())
                tech_norm_by_key = {tkey: norm(nname(tnode, tkey)) for tkey, tnode in tech_items}

                # Proceed only if at least one allowed tech is present (by substring)
                has_any_allowed = any(is_allowed_tech(tn) for tn in tech_norm_by_key.values())
                if not has_any_allowed:
                    continue

                # Delete everything not matching the allowed tokens
                for tkey, tnode in tech_items:
                    tech_label = nname(tnode, tkey)
                    tech_label_norm = tech_norm_by_key[tkey]
                    if not is_allowed_tech(tech_label_norm):
                        del children(sub_fn)[tkey]
                        trimmed += 1
                        print(
                            f"Deleted wall technology '{tech_label}' under Wall → {nname(sub_fn, sub_key)}; "
                            f"allowed tokens: {sorted(allowed_tokens)}"
                        )

        return trimmed

    def run_trimmer(self, root: TreeNode, stock_info):
        self.trim_roof_tech(root, stock_info)
        self.trim_wall_tech(root, stock_info)


class UncertaintyForestBuilder:
    def __init__(self, age_bands: list, age_matrix: np.ndarray | None):
        self.age_bands = age_bands
        self.age_confusion = age_matrix

    @staticmethod
    def _trim_tree_one_sample(root: Any, rng: np.random.Generator) -> Any:
        """
            Keep full hierarchy but enforce:
              A) Each Subfunction keeps exactly one Technology (uniform random if >1).
              B) That Technology keeps exactly one Specification (uniform random if >1).
            All Materials under the kept Specification are preserved.
        """
        root_copy = deepcopy(root)

        for layer_node in list(root_copy.children.values()):
            for func_node in list(layer_node.children.values()):
                for subfunc_node in list(func_node.children.values()):
                    tech_items = list(subfunc_node.children.items())
                    if not tech_items:
                        continue

                    # Rule A
                    if len(tech_items) > 1:
                        idx = rng.integers(0, len(tech_items))
                        keep_name, keep_node = tech_items[idx]
                        subfunc_node.children = {keep_name: keep_node}
                    else:
                        keep_name, keep_node = tech_items[0]

                    # Rule B
                    spec_items = list(keep_node.children.items())
                    if len(spec_items) > 1:
                        sidx = rng.integers(0, len(spec_items))
                        skeep_name, skeep_node = spec_items[sidx]
                        keep_node.children = {skeep_name: skeep_node}
                    # If 0 or 1, nothing further to trim (materials remain)

        return root_copy

    def run_one_sample(self,
                       trimmed_tree,
                       n_samples: int,
                       random_state: int | np.random.Generator | None = None):
        rng = random_state if isinstance(random_state, np.random.Generator) \
            else np.random.default_rng(random_state)
        return [self._trim_tree_one_sample(trimmed_tree, rng) for _ in range(n_samples)]

