"""Design tokens and layout math for the Grid World renderer.

Pure module: no rendering-library import of any kind, no surfaces, no
fonts -- only numbers, RGB tuples, strings and functions over them. Every
colour, spacing value, font size and on-screen string used anywhere in the
``ui`` package is read from here, so a pygame-free interpreter can import
this module and every layout formula, palette entry, luminance branch and
copy string can be asserted correct without a display.
"""

from __future__ import annotations

# --- Window and board layout -------------------------------------------------

WINDOW_SIZE = (960, 640)
BOARD_SIZE = 640
HUD_WIDTH = 320
BOARD_PADDING = 32
MIN_CELL = 24

# --- Spacing scale (multiples of 4, with declared stroke-weight exceptions) -

SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 16
SPACING_LG = 24
SPACING_XL = 32
SPACING_2XL = 48
SPACING_3XL = 64

GRID_LINE_WIDTH = 1
SELECTION_OUTLINE_WIDTH = 3

# --- Typography ---------------------------------------------------------------

FONT_BODY = 16
FONT_LABEL = 13
FONT_HEADING = 28
FONT_DISPLAY = 40


# --- Color ---------------------------------------------------------------------

def _rgb(hex_value: str) -> tuple[int, int, int]:
    """Convert a six-digit hex string (no leading ``#``) to an RGB tuple."""
    return (
        int(hex_value[0:2], 16),
        int(hex_value[2:4], 16),
        int(hex_value[4:6], 16),
    )


COLOR_BOARD_BG = (255, 255, 255)
COLOR_HUD_BG = _rgb("F2F2F2")
COLOR_GRID_LINE = _rgb("D9D9D9")
COLOR_OBSTACLE = _rgb("1A1A1A")
COLOR_TEXT = _rgb("1A1A1A")
COLOR_TEXT_MUTED = _rgb("5C5C5C")
COLOR_DESTRUCTIVE = _rgb("D55E00")
COLOR_SCRIM = _rgb("1A1A1A")

SCRIM_ALPHA = round(255 * 0.60)
PARKED_FILL_ALPHA = round(255 * 0.25)
FLASH_ALPHA = round(255 * 0.30)

FLASH_MS = 150

HUD_WARNING_BLOCK_HEIGHT = 64
"""Vertical space reserved for the Phase 3 unwinnable warning block, so
adding it later never reflows the HUD panel stack."""

CAR_PALETTE: tuple[tuple[int, int, int], ...] = (
    _rgb("E69F00"),  # 1 orange
    _rgb("56B4E9"),  # 2 sky blue
    _rgb("009E73"),  # 3 bluish green
    _rgb("F0E442"),  # 4 yellow
    _rgb("0072B2"),  # 5 blue
    _rgb("D55E00"),  # 6 vermillion
    _rgb("CC79A7"),  # 7 reddish purple
    _rgb("8C6D31"),  # 8 brown
)

# --- Copy -----------------------------------------------------------------------

WINDOW_TITLE = "Grid World" + " " + "—" + " " + "TP1"
LABEL_MOVES = "Moves"
LABEL_CAR = "Car"
LABEL_CONTROLS = "Controls"
NO_CAR_SELECTED = "—"

CONTROL_LEGEND_ARROW_ROW = 0
"""Index of the legend row whose arrows are drawn rather than typeset.

``01-UI-SPEC.md`` specifies the row as ``← ↑ → ↓  move``, but pygame's
bundled ``freesansbold.ttf`` carries no glyph for U+2190-U+2193: all four
render the identical ``.notdef`` box. Substituting a system font that does
carry them (DejaVu Sans) would only move the failure to whichever machine
lacks it. So the arrows are drawn with ``pygame.draw`` primitives by
``sprites.arrow_row_surface`` -- the same no-asset, no-font-dependency rule
the car and flag sprites already follow. The SPEC's visual intent is
preserved exactly; only the mechanism differs.
"""

CONTROL_LEGEND: tuple[str, str, str, str] = (
    "move",
    "1-9" + "  " + "select car",
    "R" + "  " + "reset",
    "Esc" + "  " + "quit",
)

WIN_BODY = "R to play again" + " · " + "Esc to quit"


# --- Layout math ----------------------------------------------------------------

def cell_size(cols: int, rows: int) -> int:
    """Return the square cell size for a ``cols`` by ``rows`` grid.

    ``max(MIN_CELL, floor((BOARD_SIZE - 2 * BOARD_PADDING) / max(cols, rows)))``.
    Always integer floor division -- never float arithmetic, never ``round``.
    """
    return max(MIN_CELL, (BOARD_SIZE - 2 * BOARD_PADDING) // max(cols, rows))


def grid_origin(cols: int, rows: int) -> tuple[int, int]:
    """Return the top-left pixel ``(x, y)`` of the grid inside the board area.

    The grid is centered inside the ``BOARD_SIZE`` square board area when it
    fits. When the ``MIN_CELL`` floor makes the grid larger than the board
    area, the origin clamps to the board's own top-left so the grid is drawn
    from there and clipped at the board edge rather than being centered
    off-screen in both directions.
    """
    cell = cell_size(cols, rows)
    grid_width = cols * cell
    grid_height = rows * cell
    origin_x = max(0, (BOARD_SIZE - grid_width) // 2)
    origin_y = max(0, (BOARD_SIZE - grid_height) // 2)
    return origin_x, origin_y


def numeral_size(cell: int, digits: int) -> int:
    """Return the numeral point size for a sprite drawn at ``cell`` pixels.

    ``max(12, floor(cell * 0.42))`` for one digit, ``max(12, floor(cell *
    0.30))`` for two or more digits, so car counts above nine keep numeric
    labels with no letter fallback.
    """
    if digits <= 1:
        return max(12, (cell * 42) // 100)
    return max(12, (cell * 30) // 100)


def car_hue(car: int) -> tuple[int, int, int]:
    """Return the palette colour for 1-based car number ``car``.

    Cycles through ``CAR_PALETTE`` for car numbers beyond the palette length.
    """
    return CAR_PALETTE[(car - 1) % len(CAR_PALETTE)]


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Return the sRGB relative luminance of ``rgb`` on a 0.0-1.0 scale."""

    def channel(value: int) -> float:
        c = value / 255.0
        if c <= 0.03928:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def numeral_color(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return the numeral colour for a sprite filled with ``fill``.

    Derived from ``fill``'s relative luminance, never hardcoded:
    luminance > 0.5 -> ``COLOR_TEXT``, otherwise white.
    """
    if relative_luminance(fill) > 0.5:
        return COLOR_TEXT
    return (255, 255, 255)


def win_heading(moves: int) -> str:
    """Return the win overlay heading for a solve taking ``moves`` moves.

    Singular at exactly one move, plural otherwise.
    """
    if moves == 1:
        return "Solved in 1 move"
    return f"Solved in {moves} moves"
