"""Research component-dimension conversion for numeric synthetic inputs.

The apply_dimensions function is copied unchanged; the numeric-only helper
avoids evaluating expression strings in this self-contained example.
"""
import math


def evaluate_expression(expr, area_value):
    if not isinstance(expr, (int, float)):
        raise ValueError("The synthetic example accepts numeric dimensions only")
    return expr


def apply_dimensions(node, ngd_dict, inferred_dict, area_value=1,
                     layer=None, function=None, subfunction=None, floor_count=1):
    """
    Recursively apply dimension values to each leaf node in the tree.

    For each MI leaf, convert:
        MI → [MI, Dimension, MI * Dimension]

    Parameters
    ----------
    node : TreeNode
        The root node or subtree node.
    ngd_dict : dict
        Nested dictionary containing dimension multipliers from NGD by layer/function/subfunction.
    area_value : float
        Base area multiplier.
    layer, function, subfunction : str
        Track current hierarchy for recursion.
    floor_count : int
        The number of floors of a building
    inferred_dict:
        Nested dictionary containing dimension multipliers from mc sampled by layer/function/subfunction.
    """
    if math.isnan(floor_count):
        floor_count = 1
    component_dims = {
        'Structure': {
            'Wall': ngd_dict['perimeter'] * ngd_dict['wall_height'],
            'Roof': ngd_dict['roof_area'],
            'Floor': ngd_dict['area'],
        },
        'Skin': {
            'Wall': ngd_dict['perimeter'] * ngd_dict['wall_height'],
            'Roof': ngd_dict['roof_area'],
            'Opening': {
                'Door': 2,
                'Window': inferred_dict['window']
            },
            'Floor': ngd_dict['area']
        },
        'Space': {
            'Wall': inferred_dict['interior_wall'] * ngd_dict['wall_height'] / ngd_dict['floor'],
            'Floor': (ngd_dict['floor'] - 1) * ngd_dict['area'],
            'Opening': {
                'Door': inferred_dict['interior_door']
            }
        },
        'Services': {
            'Heating ': {
                'Transportation': inferred_dict['room'] * (0.25 * ngd_dict['perimeter'] + 2 * ngd_dict['floor']),
                # pipe
                'Emitter': {'Electric Radiator': inferred_dict['room'],
                            'Hot Water Radiator': inferred_dict['room'],
                            'Underfloor heating': ngd_dict['area'] * ngd_dict['floor']},  # this includes floor heating
                'Storage': 0.4,
                'Heat Source': 1
            },
            'Electrical': {
                'Wiring': inferred_dict['room'] * (0.25 * ngd_dict['perimeter'] + 2 * ngd_dict['floor'])
            },
            'Ventilation': {
                'Extraction': 2,  # no wiring?
                'Wiring': inferred_dict['room'] * (0.25 * ngd_dict['perimeter'] + 2 * ngd_dict['floor'])
            }
        }
    }

    if node.level == 0:
        layer = node.name
    elif node.level == 1:
        function = node.name
    elif node.level == 2:
        subfunction = node.name

    if not node.children:
        # Leaf node: apply conversion rule
        dimension = area_value  # default fallback

        try:
            layer_dict = component_dims.get(layer, {})

            # Direct value (e.g., Services.Heat Source = 1)
            if isinstance(layer_dict, (int, float)):
                dimension = layer_dict

            elif isinstance(layer_dict, dict):
                f_val = layer_dict.get(function)

                if f_val is None:
                    dimension = area_value
                elif isinstance(f_val, (int, float, str)):
                    dimension = f_val
                elif isinstance(f_val, dict):
                    if subfunction in f_val:
                        dimension = f_val[subfunction]
                    else:
                        print(f'the dimension of {f_val} has been set to area')
                        dimension = area_value
        except Exception:
            dimension = area_value

        dimension = evaluate_expression(dimension, area_value)

        if node.mi is not None:
            try:
                node.mi = [node.mi, dimension, node.mi * dimension]
            except:
                node.mi = [node.mi, dimension, None]

    else:
        for child in node.children.values():
            apply_dimensions(child, ngd_dict, inferred_dict, area_value, layer, function, subfunction, floor_count)

