"""Generate small, wholly synthetic fixtures; no external data are read.

Run ``python generate_data.py`` from this directory, or specify ``--output``.
Coefficients below are illustrative assumptions, not calibrated UK estimates.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

AGES = ["<1920", "1960-1980", "2002-2006"]
DEFAULT_SEED = 20261005


def buildings() -> pd.DataFrame:
    """Two fictional areas with twelve houses each and no real coordinates."""
    rows = []
    for i in range(24):
        width = 6.0 + (i % 5) * 0.65
        depth = 7.5 + (i % 4) * 0.7
        floor = [1, 2, 2, 3][i % 4]
        flat = i % 4 == 3
        footprint = width * depth
        rows.append({
            "building_id": f"synthetic_{i + 1:03d}",
            "area_id": "synthetic_area_A" if i < 12 else "synthetic_area_B",
            "age": AGES[i % 3], "footprint_m2": footprint,
            "perimeter_m": 2 * (width + depth), "storeys": floor,
            "height_m": 2.7 * floor,
            "roof_area_m2": footprint if flat else footprint * 1.18,
            "wall_material": ["Brick Or Block Or Stone", "Concrete", "Timber Or Wood"][(i // 3) % 3],
            "roof_material": "Waterproof Membrane Or Concrete" if flat else "Tile Or Stone Or Slate",
            "roof_shape": "flat" if flat else "pitched",
            "width_m": width, "depth_m": depth,
        })
    return pd.DataFrame(rows)


def training(seed: int = DEFAULT_SEED, n: int = 300) -> pd.DataFrame:
    """Illustrative conditional count/lognormal mechanism with genuine noise."""
    rng = np.random.default_rng(seed)
    width = rng.uniform(5, 10, n)
    depth = rng.uniform(6, 12, n)
    floors = rng.choice([1, 2, 3], size=n, p=[0.25, 0.60, 0.15])
    area = width * depth * floors
    perimeter = 2 * (width + depth)
    az = (area - 130) / 60
    pz = (perimeter - 33) / 5
    fz = floors - 2
    rooms = rng.poisson(np.exp(1.80 + 0.23 * az + 0.09 * pz + 0.10 * fz))
    rz = (rooms - 6) / 3
    windows = rng.poisson(np.exp(2.00 + 0.12 * az + 0.09 * pz + 0.10 * fz + 0.10 * rz))
    doors = rng.poisson(np.exp(1.65 + 0.12 * az + 0.06 * fz + 0.16 * rz))
    ln_length = 3.20 + 0.19 * az + 0.07 * pz + 0.12 * fz + 0.10 * rz + 0.04 * (doors - 5)
    length = np.exp(ln_length + rng.normal(0, 0.22, n))
    return pd.DataFrame({"T": area, "P": perimeter, "F": floors,
                         "R": rooms, "W": windows, "D": doors, "L": length})


def age_probabilities() -> pd.DataFrame:
    """Rows are sampled ages; columns are observed ages; columns sum to one."""
    return pd.DataFrame([[0.80, 0.12, 0.05],
                         [0.15, 0.76, 0.15],
                         [0.05, 0.12, 0.80]], index=pd.Index(AGES, name="sampled_age"), columns=AGES)


def hmi() -> pd.DataFrame:
    """Illustrative hierarchy with alternative technologies/specifications.

    Materials within one selected specification are jointly retained. Age
    eligibility is an assumption encoded explicitly using YES/NO columns.
    """
    rows = []

    def option(layer, function, subfunction, technology, specification,
               materials, ages=AGES, unit="kg/m2"):
        for material, mi in materials:
            row = dict(zip(["Layer", "Function", "Sub-Function", "Technology", "Specification", "Material"],
                           [layer, function, subfunction, technology, specification, material]))
            row.update({"MI": float(mi), "unit": unit})
            row.update({age: "YES" if age in ages else "NO" for age in AGES})
            rows.append(row)

    # Load-bearing walls: alternatives selected by the material tree sampler.
    option("Structure", "Wall", "Load bearing", "Brick masonry", "Illustrative solid wall", [("Brick", 260), ("Mortar", 30)])
    option("Structure", "Wall", "Load bearing", "Brick masonry", "Illustrative later wall", [("Brick", 190), ("Mortar", 24)], AGES[1:])
    option("Structure", "Wall", "Load bearing", "Stone masonry", "Illustrative rubble wall", [("Sandstone", 390), ("Mortar", 40)], AGES[:2])
    option("Structure", "Wall", "Load bearing", "Concrete", "Illustrative cast wall", [("Concrete", 300), ("Steel", 4)])
    option("Structure", "Wall", "Load bearing", "Timber framing", "Illustrative frame", [("Timber studding", 24), ("Plywood", 7)])
    option("Structure", "Floor", "Load bearing", "Concrete slab", "Illustrative slab", [("Concrete", 220), ("Steel", 5)])
    option("Structure", "Floor", "Load bearing", "Timber joists", "Illustrative joists", [("Timber structure", 26), ("Chipboard", 10)])
    option("Structure", "Roof", "Load bearing", "Timber rafters", "Illustrative rafters", [("Timber structure", 22), ("Steel", 1)])
    option("Structure", "Roof", "Load bearing", "Concrete deck", "Illustrative deck", [("Concrete", 150), ("Steel", 4)], AGES[1:])

    # Exact subfunction and technology labels enable the original trimmer.
    option("Skin", "Wall", "Non-load bearing structure", "Brick cladding", "Illustrative facing", [("Brick", 90), ("Mortar", 12)])
    option("Skin", "Wall", "Non-load bearing structure", "Concrete panel", "Illustrative panel", [("Concrete", 110)])
    option("Skin", "Wall", "Non-load bearing structure", "Timber cladding", "Illustrative boards", [("Timber decking", 13), ("Steel", 0.5)])
    option("Skin", "Wall", "Insulation", "Mineral insulation", "Illustrative insulation", [("Mineral wool", 5)])
    option("Skin", "Wall", "Insulation", "Foam insulation", "Illustrative insulation", [("Polymer foam", 2)], AGES[1:])
    option("Skin", "Roof", "Finish 2", "Tiles", "Illustrative clay tiles", [("Clay tile", 40), ("Timber battens", 4)])
    option("Skin", "Roof", "Finish 2", "Tiles", "Illustrative slate tiles", [("Slate", 32), ("Timber battens", 4)], AGES[:2])
    option("Skin", "Roof", "Finish 2", "Covering", "Illustrative membrane", [("Bitumen membrane", 5)])
    option("Skin", "Roof", "Finish 1", "Ballast", "Illustrative ballast", [("Ballast (gravel)", 25)])
    option("Skin", "Roof", "Finish 1", "Timber support", "Illustrative support", [("Timber decking", 8)])
    option("Skin", "Opening", "Window", "Metal frame", "Illustrative window", [("Aluminium", 8), ("Glass", 18)], unit="kg/item")
    option("Skin", "Opening", "Window", "Polymer frame", "Illustrative window", [("PVC", 7), ("Glass", 15)], unit="kg/item")
    option("Skin", "Opening", "Door", "Timber door", "Illustrative door", [("Timber", 25), ("Brass", 0.6)], unit="kg/item")

    option("Space", "Wall", "Partition", "Brick partition", "Illustrative partition", [("Brick", 80), ("Plaster", 10)])
    option("Space", "Wall", "Partition", "Timber partition", "Illustrative partition", [("Timber studding", 8), ("Plasterboard", 12)])
    option("Space", "Floor", "Finish", "Timber flooring", "Illustrative floor", [("Timber hardwood flooring", 12)])
    option("Space", "Floor", "Finish", "Sheet flooring", "Illustrative floor", [("Chipboard", 10), ("PVC", 2)])
    option("Space", "Opening", "Door", "Timber door", "Illustrative internal door", [("Timber", 15), ("Steel", 0.4)], unit="kg/item")

    option("Services", "Electrical", "Wiring", "Copper cable", "Illustrative cable A", [("Copper", 0.10), ("Plastic", 0.04)], unit="kg/m")
    option("Services", "Electrical", "Wiring", "Copper cable", "Illustrative cable B", [("Copper", 0.14), ("Plastic", 0.06)], AGES[1:], unit="kg/m")
    option("Services", "Ventilation", "Extraction", "Extract fan", "Illustrative fan", [("Steel", 1.5), ("Plastic", 0.8)], unit="kg/item")
    return pd.DataFrame(rows)


def generate(output: Path, seed: int = DEFAULT_SEED) -> None:
    output.mkdir(parents=True, exist_ok=True)
    buildings().to_csv(output / "buildings.csv", index=False, float_format="%.10g")
    training(seed).to_csv(output / "training.csv", index=False, float_format="%.10g")
    age_probabilities().to_csv(output / "age_probabilities.csv", float_format="%.10g")
    hmi().to_csv(output / "hmi.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output, args.seed)
    print(f"Synthetic fixtures written to {args.output.resolve()}")
