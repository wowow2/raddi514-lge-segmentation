"""Dark radiology theme. Every color/font/size literal lives here — nowhere else."""

from dataclasses import dataclass

from matplotlib.colors import ListedColormap


@dataclass(frozen=True)
class Theme:
    # Canvas
    figure_face: str = "#0b0e14"
    axes_face: str = "#0b0e14"
    panel_face: str = "#151a24"
    panel_edge: str = "#2b3346"
    # Text
    text: str = "#e6e9f0"
    dim_text: str = "#8b93a7"
    # HUD
    hud_font: str = "monospace"
    hud_size: float = 9.0
    title_size: float = 12.0
    # Overlay classes: (name, rgba) for labels 1..4
    classes: tuple = (
        ("Cavity", (0.25, 0.62, 1.00, 0.55)),
        ("Myo", (0.20, 0.90, 0.35, 0.55)),
        ("Scar", (1.00, 0.25, 0.25, 0.80)),
        ("MVO", (1.00, 0.90, 0.10, 0.95)),
    )
    # Filmstrip
    strip_current_edge: str = "#5bc8ff"

    @property
    def overlay_cmap(self) -> ListedColormap:
        return ListedColormap([(0, 0, 0, 0)] + [c[1] for c in self.classes])

    @property
    def legend_colors(self) -> list[tuple]:
        return [c[1] for c in self.classes]


DARK = Theme()
