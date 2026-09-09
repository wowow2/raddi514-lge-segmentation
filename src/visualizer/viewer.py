import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Colormap: 0: BG, 1: Cavity (Blue), 2: Myo (Green), 3: Scar (Red), 4: MVO (Yellow)
CMAP = ListedColormap([
    (0.0, 0.0, 0.0, 0.0),
    (0.1, 0.45, 0.95, 0.55),
    (0.1, 0.85, 0.25, 0.55),
    (0.95, 0.15, 0.15, 0.80),
    (1.0, 0.92, 0.05, 0.95)
])

class DatasetViewer:
    def __init__(self, cases):
        self.cases = cases
        self.case_idx = 0
        self.slice_idx = 0
        self.show_mask = True
        
        self.fig = plt.figure(figsize=(12, 7.5))
        self.fig.canvas.manager.set_window_title("EMIDEC Explorer - Imaging & Clinical Metadata")
        
        # Grid: 70% Image, 30% Metadata Panel
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.15)
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_info.axis("off")
        
        # Load initial
        case = self.cases[self.case_idx]
        img, gt = case.load()
        self.slice_idx = img.shape[2] // 2
        
        # Fast Artists
        self.im_artist = self.ax_img.imshow(
            img[:, :, self.slice_idx].T, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1
        )
        self.mask_artist = self.ax_img.imshow(
            gt[:, :, self.slice_idx].T, cmap=CMAP, origin="lower", aspect="equal", vmin=0, vmax=4, interpolation="none"
        )
        self.ax_img.axis("off")
        
        # Static Legend
        legend_elements = [
            Patch(facecolor=(0.1, 0.45, 0.95, 0.6), edgecolor='none', label='1: Cavity'),
            Patch(facecolor=(0.1, 0.85, 0.25, 0.6), edgecolor='none', label='2: Myo'),
            Patch(facecolor=(0.95, 0.15, 0.15, 0.8), edgecolor='none', label='3: Infarct'),
            Patch(facecolor=(1.0, 0.92, 0.05, 0.95), edgecolor='none', label='4: MVO')
        ]
        self.ax_img.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, frameon=True)
        
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.render()

    def render(self):
        case = self.cases[self.case_idx]
        img, gt = case.load()
        num_slices = img.shape[2]
        
        # 1. Update Image & Mask
        self.im_artist.set_data(img[:, :, self.slice_idx].T)
        if self.show_mask:
            self.mask_artist.set_data(gt[:, :, self.slice_idx].T)
            self.mask_artist.set_visible(True)
        else:
            self.mask_artist.set_visible(False)
            
        self.ax_img.set_title(
            f"{case.case_id} ({case.group}) — Slice {self.slice_idx + 1}/{num_slices}",
            fontsize=12, fontweight="bold", pad=10
        )
        
        # 2. Render Formatted Clinical & Technical HUD
        self.ax_info.clear()
        self.ax_info.axis("off")
        
        meta = case.clinical_meta
        vox = case.voxel_counts
        
        info_lines = [
            r"$\bf{PATIENT\ INFORMATION}$",
            f"  • Case: {case.case_id}  ({case.group})",
            f"  • Age / Sex: {meta.get('Age', 'N/A')} yrs  |  {meta.get('Sex', 'N/A')}",
            f"  • Overweight: {meta.get('Overweight', 'N/A')}  |  Tobacco: {meta.get('Tobacco', 'N/A')}",
            f"  • Hypertension: {meta.get('Arterial hypertension', 'N/A')}  |  Diabetes: {meta.get('Diabetes', 'N/A')}",
            "",
            r"$\bf{CARDIAC\ BIOMARKERS}$",
            f"  • LVEF (FEVG): {meta.get('FEVG', 'N/A')} %",
            f"  • Troponin: {meta.get('Troponin', 'N/A')}",
            f"  • NT-proBNP: {meta.get('NTProBNP', 'N/A')}",
            f"  • Killip Max: {meta.get('Killip Max', 'N/A')}",
            f"  • ECG (ST+): {meta.get('ECG (ST +)', 'N/A')}",
            "",
            r"$\bf{ACQUISITION\ GEOMETRY}$",
            f"  • Matrix Size: {case.shape[0]} × {case.shape[1]}",
            f"  • Slices (Z): {case.shape[2]}",
            f"  • In-Plane Spacing: {case.zooms[0]} × {case.zooms[1]} mm",
            f"  • Slice Thickness: {case.zooms[2]} mm",
            "",
            r"$\bf{PATHOLOGY\ QUANTIFICATION}$",
            f"  • Infarction (Scar): {'YES' if case.has_scar else 'NO'} ({vox[3]:,d} voxels)",
            f"  • MVO (No-Reflow): {'YES' if case.has_mvo else 'NO'} ({vox[4]:,d} voxels)",
            f"  • LV Cavity: {vox[1]:,d} voxels",
            f"  • Healthy Myo: {vox[2]:,d} voxels",
            "",
            "─" * 40,
            r"$\bf{CONTROLS}$",
            "  [← / →]  : Change Slice",
            "  [↑ / ↓]  : Previous / Next Patient",
            "  [M]      : Toggle Mask Overlay",
            "  [Q / ESC]: Exit"
        ]
        
        self.ax_info.text(
            0.05, 0.98, "\n".join(info_lines),
            transform=self.ax_info.transAxes,
            fontsize=9.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9)
        )
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        case = self.cases[self.case_idx]
        img, _ = case.load()
        num_slices = img.shape[2]
        
        if event.key == "right":
            self.slice_idx = min(self.slice_idx + 1, num_slices - 1)
            self.render()
        elif event.key == "left":
            self.slice_idx = max(self.slice_idx - 1, 0)
            self.render()
        elif event.key in ["down", "n", "pageup"]:
            self.case_idx = (self.case_idx + 1) % len(self.cases)
            new_case = self.cases[self.case_idx]
            new_img, _ = new_case.load()
            self.slice_idx = new_img.shape[2] // 2
            self.render()
        elif event.key in ["up", "p", "pagedown"]:
            self.case_idx = (self.case_idx - 1) % len(self.cases)
            new_case = self.cases[self.case_idx]
            new_img, _ = new_case.load()
            self.slice_idx = new_img.shape[2] // 2
            self.render()
        elif event.key in ["m", "M"]:
            self.show_mask = not self.show_mask
            self.render()
        elif event.key in ["q", "Q", "escape"]:
            plt.close(self.fig)

    def show(self):
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.show()
