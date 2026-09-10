"""Single source of truth for key bindings.

Each entry: (keys, action_id, help text). The HUD controls section and the
viewer's dispatch both derive from this table — no undocumented keys.
"""

# action ids handled by the browser panel
BROWSER_KEYS: list[tuple[tuple[str, ...], str, str]] = [
    (("right",), "slice_next", "Change Slice"),
    (("left",), "slice_prev", "Change Slice"),
    (("down", "n", "pagedown"), "case_next", "Next Patient"),
    (("up", "p", "pageup"), "case_prev", "Previous Patient"),
    (("m",), "mask_toggle", "Toggle Mask Overlay"),
    (("1", "2", "3", "4"), "class_toggle", "Toggle Class 1-4"),
    (("e",), "export_png", "Export PNG"),
]

# action ids handled by the viewer root (search/quit)
VIEWER_KEYS: list[tuple[tuple[str, ...], str, str]] = [
    (("/",), "search_start", "Go To Case By Number"),
    (("q", "Q", "escape"), "quit", "Exit"),
]


def help_lines() -> list[str]:
    """HUD controls section built from the tables above."""
    lines = []
    for keys, _, help_text in BROWSER_KEYS:
        lines.append(f"  [{' / '.join(keys)}] : {help_text}")
    for keys, _, help_text in VIEWER_KEYS:
        lines.append(f"  [{' / '.join(keys)}] : {help_text}")
    return lines


def find_action(key: str, table) -> str | None:
    """Return the action_id bound to `key`, or None."""
    for keys, action, _ in table:
        if key in keys:
            return action
    return None
