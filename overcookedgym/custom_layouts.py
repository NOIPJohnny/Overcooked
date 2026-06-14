from copy import deepcopy


CUSTOM_LAYOUTS = {
    "chicane_bottleneck_hard": {
        "grid": [
            "XXXXXXOXXXXXXX",
            "X 1      D   X",
            "X XXXX XXXX  X",
            "X    X X     X",
            "XXXX X X XXXXX",
            "XS   X X  2  X",
            "XPP        XXX",
            "XXXXXXXXXXXXXX",
        ],
        "start_order_list": None,
        "cook_time": 20,
        "num_items_for_soup": 3,
        "delivery_reward": 20,
        "rew_shaping_params": None,
    },
    "asymmetric_corridor_hard": {
        "grid": [
            "XXXXXOXXXXDXXX",
            "X  1    X    X",
            "X XXXX  X XX X",
            "X    X     X X",
            "XXX  XXXXX X X",
            "X    X   2  XX",
            "XSXXXX XXXXPPX",
            "XXXXXXXXXXXXXX",
        ],
        "start_order_list": None,
        "cook_time": 20,
        "num_items_for_soup": 3,
        "delivery_reward": 20,
        "rew_shaping_params": None,
    },
    "double_pot_maze_hard": {
        "grid": [
            "XXXXOXXXXXDXXXX",
            "X 1    X      X",
            "X XXXX XXX XX X",
            "X    X     X  X",
            "XXX  XXXXX X XX",
            "X    X   2   PX",
            "XSXXXX XXXXPXXX",
            "XXXXXXXXXXXXXXX",
        ],
        "start_order_list": None,
        "cook_time": 20,
        "num_items_for_soup": 3,
        "delivery_reward": 20,
        "rew_shaping_params": None,
    },
}

CUSTOM_LAYOUT_NAMES = tuple(CUSTOM_LAYOUTS.keys())


def custom_layout_params(layout_name):
    params = deepcopy(CUSTOM_LAYOUTS[layout_name])
    params["layout_name"] = layout_name
    return params
