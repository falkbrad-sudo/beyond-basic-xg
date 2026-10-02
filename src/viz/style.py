"""Shared visual style for every figure in this project.

One source of truth for surfaces, ink and colour ramps, so the figures read as
one system rather than as a pile of separate matplotlib defaults.

Colour assignment follows the job the colour does, not taste:

- **Shot quality (xG) and point density are magnitude**, so they get a single
  hue, light to dark (`XG_CMAP` / `DENSITY_CMAP`). Never a rainbow: a
  multi-hue ramp invents ordering that isn't in the data and stops being
  readable for colourblind viewers.
- **Goal vs. miss is identity, not magnitude**, so it is carried by marker
  *shape*, never by colour alone (see `shot_maps.py`). That keeps the colour
  channel free for xG and keeps the outcome legible in greyscale and under
  any form of colour-vision deficiency.
- Reference lines, grids and axes are chrome, so they wear ink colours and sit
  one shade off the surface rather than competing with the data.

Contrast of every step used for a discrete mark is asserted in
tests/test_viz_style.py rather than eyeballed.
"""
from __future__ import annotations

import textwrap

from matplotlib.colors import LinearSegmentedColormap

# --- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- single-hue sequential ramp (blue), light -> dark ----------------------
# Steps 100-700. The full range is for continuous/sequential encoding, where
# the lightest step legitimately means "near zero" and may recede toward the
# surface.
BLUE_RAMP = {
    100: "#cde2fb",
    150: "#b7d3f6",
    200: "#9ec5f4",
    250: "#86b6ef",
    300: "#6da7ec",
    350: "#5598e7",
    400: "#3987e5",
    450: "#2a78d6",
    500: "#256abf",
    550: "#1c5cab",
    600: "#184f95",
    650: "#104281",
    700: "#0d366b",
}

# The lightest step allowed for a DISCRETE mark on the light surface; lighter
# steps fall below the 2:1 floor and vanish into the background.
ORDINAL_LIGHTEST = BLUE_RAMP[250]

# Primary accent for a single series (categorical slot 1).
SERIES_1 = "#2a78d6"

# Continuous ramps. Sequential encoding, so the full light->dark range is in
# play; both are the same hue because they answer the same kind of question.
XG_CMAP = LinearSegmentedColormap.from_list(
    "bbxg_blue",
    [BLUE_RAMP[s] for s in (100, 200, 300, 400, 500, 600, 700)],
)

# Starts at the ordinal floor rather than the lightest step: in a density plot
# drawn with mincnt=1, the single-count bins are discrete marks that carry the
# outliers, so they have to clear 2:1 against the surface instead of fading
# into it the way a genuine near-zero magnitude is allowed to.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "bbxg_blue_density",
    [BLUE_RAMP[s] for s in (250, 350, 450, 550, 650, 700)],
)

# --- diverging ramp for pitch control --------------------------------------
# Pitch control is a POLARITY, not a magnitude: 0.5 means "contested" and the
# two directions mean opposite things (attack controls / defence controls).
# So it gets a diverging scale -- two hues that read as opposite, with a
# neutral grey midpoint that reads as "neither". Blue is the attack (matching
# the sequential xG hue), red the defence. A single-hue ramp here would imply
# that 0.5 is "half as much" of something rather than a boundary.
RED_RAMP = {200: "#f6bfbe", 350: "#ef8d8c", 500: "#e34948", 650: "#a82423"}
NEUTRAL_MID = "#f0efec"

CONTROL_CMAP = LinearSegmentedColormap.from_list(
    "bbxg_control_diverging",
    [
        RED_RAMP[650], RED_RAMP[500], RED_RAMP[350], RED_RAMP[200],
        NEUTRAL_MID,
        BLUE_RAMP[200], BLUE_RAMP[350], BLUE_RAMP[500], BLUE_RAMP[650],
    ],
)

# Team marker colours, matching the two poles of CONTROL_CMAP so the players
# and the surface they generate read as the same encoding.
ATTACK_COLOR = BLUE_RAMP[650]
DEFEND_COLOR = RED_RAMP[650]
BALL_COLOR = "#0b0b0b"

# Marker shapes carry outcome, so outcome never depends on colour.
GOAL_MARKER = "*"
MISS_MARKER = "o"

FIGURE_DPI = 160


def _srgb_to_linear(channel: float) -> float:
    """Undo the sRGB transfer function for one 0-1 channel."""
    return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a hex colour.

    Parameters
    ----------
    hex_color : str
        '#rrggbb'.

    Returns
    -------
    float
        Luminance in [0, 1].
    """
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (
        0.2126 * _srgb_to_linear(r)
        + 0.7152 * _srgb_to_linear(g)
        + 0.0722 * _srgb_to_linear(b)
    )


def contrast_ratio(foreground: str, background: str = SURFACE) -> float:
    """WCAG contrast ratio between two hex colours.

    Used to check colour choices by computation rather than by eye; see
    tests/test_viz_style.py.

    Parameters
    ----------
    foreground, background : str
        '#rrggbb'. Background defaults to this project's chart surface.

    Returns
    -------
    float
        Ratio in [1, 21].

    Examples
    --------
    >>> round(contrast_ratio("#000000", "#ffffff"), 1)
    21.0
    >>> round(contrast_ratio("#ffffff", "#ffffff"), 1)
    1.0
    """
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def apply_base_style(ax, *, grid_axis: str | None = "both") -> None:
    """Strip matplotlib's default chrome down to recessive hairlines.

    Removes the top/right spines, puts the remaining ones in the baseline
    colour, and draws a solid hairline grid one shade off the surface (never
    dashed, since dashing reads as "projection" when it is only a grid).
    """
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=GRIDLINE,
            linewidth=1.0,
            linestyle="-",
            zorder=0,
        )
        ax.set_axisbelow(True)


def add_titles(
    fig,
    title: str | None,
    subtitle: str | None = None,
    *,
    width: int = 68,
    subtitle_width: int = 92,
) -> None:
    """Place a wrapped title/subtitle block at the top-left of a figure.

    Wrapping is done here rather than left to the caller because an overlong
    headline silently clips at the figure edge or collides with a colorbar,
    both of which make a figure harder to read and easy to misread.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    title, subtitle : str, optional
    width, subtitle_width : int
        Wrap widths in characters; the subtitle is smaller so it fits more.
    """
    y = 0.975
    if title:
        wrapped = textwrap.fill(title, width=width)
        fig.text(
            0.012, y, wrapped,
            color=INK_PRIMARY, fontsize=13, va="top", ha="left", linespacing=1.35,
        )
        y -= 0.056 * (wrapped.count("\n") + 1) + 0.012
    if subtitle:
        fig.text(
            0.012, y, textwrap.fill(subtitle, width=subtitle_width),
            color=INK_SECONDARY, fontsize=9.5, va="top", ha="left", linespacing=1.4,
        )
