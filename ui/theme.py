"""Design tokens + Flet theme. Linear-style: dark, low-contrast, tight."""

from __future__ import annotations

import flet as ft


class T:
    """Design tokens. Reach into these directly from anywhere in ui/."""

    # Surfaces
    BG = "#0E0F11"          # page background
    SURFACE = "#16181B"     # cards / panels
    SURFACE_2 = "#1C1F23"   # raised / hover
    SIDEBAR = "#101113"

    # Strokes
    BORDER = "#23262B"
    BORDER_FAINT = "#1A1C1F"

    # Text
    TEXT = "#E6E8EB"
    TEXT_DIM = "#8A8F98"
    TEXT_FAINT = "#5C6068"

    # Accent (Linear purple)
    ACCENT = "#5E6AD2"
    ACCENT_HOVER = "#6E7AD8"
    ACCENT_DIM = "#3A3F7A"

    # Status
    SUCCESS = "#4CB782"
    WARNING = "#E2A73D"
    ERROR = "#EB5757"

    # Spacing rhythm (4px grid)
    S1 = 4
    S2 = 8
    S3 = 12
    S4 = 16
    S5 = 20
    S6 = 24

    # Radii
    R_SM = 4
    R = 6
    R_LG = 8


def linear_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=T.ACCENT,
            on_primary="#FFFFFF",
            secondary=T.TEXT_DIM,
            on_secondary=T.TEXT,
            surface=T.SURFACE,
            on_surface=T.TEXT,
            outline=T.BORDER,
            outline_variant=T.BORDER_FAINT,
            error=T.ERROR,
        ),
        font_family="Inter",
        visual_density=ft.VisualDensity.COMPACT,
    )


def hairline(width: int = 1) -> ft.BorderSide:
    return ft.BorderSide(width, T.BORDER)
