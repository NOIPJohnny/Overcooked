import numpy as np


TERRAIN_COLORS = {
    " ": (236, 238, 241),
    "X": (118, 124, 132),
    "O": (210, 170, 90),
    "T": (204, 95, 82),
    "D": (230, 230, 240),
    "S": (96, 150, 105),
    "P": (76, 93, 112),
}

OBJECT_COLORS = {
    "onion": (230, 204, 79),
    "tomato": (210, 73, 67),
    "dish": (245, 245, 250),
    "soup": (224, 134, 64),
}

PLAYER_COLORS = [(58, 124, 196), (213, 92, 77), (78, 164, 111), (155, 92, 184)]


def render_state(mdp, state, tile_size=48):
    from PIL import Image, ImageDraw, ImageFont

    width = mdp.width * tile_size
    height = mdp.height * tile_size
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    for y, row in enumerate(mdp.terrain_mtx):
        for x, terrain in enumerate(row):
            _draw_tile(draw, x, y, terrain, tile_size)

    for obj in state.objects.values():
        _draw_object(draw, obj, tile_size)

    for idx, player in enumerate(state.players):
        _draw_player(draw, idx, player, tile_size)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(11, tile_size // 5))
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 4), f"Orders: {state.curr_order}", fill=(25, 28, 32), font=font)
    return np.asarray(image)


def _draw_tile(draw, x, y, terrain, tile_size):
    x0, y0 = x * tile_size, y * tile_size
    x1, y1 = x0 + tile_size, y0 + tile_size
    fill = TERRAIN_COLORS.get(terrain, (220, 220, 220))
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(210, 214, 218))
    label = {"O": "O", "T": "T", "D": "D", "S": "S", "P": "P"}.get(terrain)
    if label is not None:
        draw.text((x0 + 5, y0 + 4), label, fill=(32, 36, 40))


def _draw_object(draw, obj, tile_size):
    x, y = obj.position
    cx = x * tile_size + tile_size // 2
    cy = y * tile_size + tile_size // 2
    radius = tile_size // 5
    fill = OBJECT_COLORS.get(obj.name, (255, 255, 255))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill=fill, outline=(35, 35, 35), width=2)
    if obj.name == "soup" and obj.state is not None:
        _, num_items, cook_time = obj.state
        text = str(cook_time if num_items == 3 else num_items)
        draw.text((cx - radius // 2, cy - radius // 2), text, fill=(30, 30, 30))


def _draw_player(draw, idx, player, tile_size):
    x, y = player.position
    pad = tile_size // 8
    x0, y0 = x * tile_size + pad, y * tile_size + pad
    x1, y1 = (x + 1) * tile_size - pad, (y + 1) * tile_size - pad
    fill = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
    draw.rounded_rectangle([x0, y0, x1, y1], radius=tile_size // 6,
                           fill=fill, outline=(20, 24, 28), width=2)
    draw.text((x0 + 4, y0 + 3), str(idx), fill=(255, 255, 255))
    _draw_orientation(draw, player, tile_size)
    if player.held_object is not None:
        held = player.held_object.deepcopy()
        held.position = (x + 0.28, y + 0.72)
        _draw_object(draw, held, tile_size)


def _draw_orientation(draw, player, tile_size):
    x, y = player.position
    cx = x * tile_size + tile_size // 2
    cy = y * tile_size + tile_size // 2
    dx, dy = player.orientation
    tip = (cx + dx * tile_size // 4, cy + dy * tile_size // 4)
    left = (cx - dy * tile_size // 8, cy + dx * tile_size // 8)
    right = (cx + dy * tile_size // 8, cy - dx * tile_size // 8)
    draw.polygon([tip, left, right], fill=(255, 255, 255))
