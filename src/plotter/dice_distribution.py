"""Per-case Dice distribution figure: panels per label, violins per run."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import LABEL_NAMES
from .cv_data import CVData


class DiceDistributionPlot:
    """Violin + group-colored strip plot of per-case Dice, one panel per label."""

    def __init__(self, cv_data: CVData, labels: tuple[int, ...] = (1, 2, 3, 4),
                 awful_below: float = 0.2):
        self.cv = cv_data
        self.labels = labels
        self.awful_below = awful_below
        self._names = list(cv_data.run_names)
        self._rng = np.random.default_rng(0)
        self._data = None  # {run: {label: [(dice, group, case_id)]}}

    # ------------------------------------------------------------------ data
    def collect(self):
        data = {name: {lbl: [] for lbl in self.labels} for name in self._names}
        for fi, nid, case in self.cv.iter_val():
            gt = self.cv.gt_array(case)
            for name in self._names:
                pred = self.cv.pred_array(name, fi, nid)
                if pred is None:
                    continue
                dd = self.cv.case_dice(pred, gt)
                for lbl in self.labels:
                    data[name][lbl].append((dd[lbl], case.group, case.case_id))
        self._data = data
        return data

    def load_results(self, tables: dict) -> dict:
        """Fill from saved metrics instead of loading prediction volumes.

        Args:
            tables: ``{run_name: ResultsTable}`` in display order.
        """
        data = {name: {lbl: [] for lbl in self.labels} for name in tables}
        for name, table in tables.items():
            for row in table.rows:
                lbl = row.get("class_id")
                dice = row.get("dice")
                if lbl not in data[name] or not row.get("dice_defined") or dice is None:
                    continue
                data[name][lbl].append(
                    (float(dice), row.get("group"), row["case_id"]))
        self._names = list(tables)
        self._data = data
        return data

    # ------------------------------------------------------------------ figure
    COLORS = {"Pathological": "#4aa3ff", "Normal": "#ff5a5a",
              "Unknown": "#aaaaaa"}

    def _panel(self, ax, name, lbl):
        """One violin + strip for a single run and label."""
        vals = [v for v, _, _ in self._data[name][lbl]]
        if not vals:  # run with no predictions on disk: leave panel empty
            ax.text(0, 0.5, "no data", color="white", ha="center")
            return
        parts = ax.violinplot([vals], positions=[0], showmeans=True, widths=0.6)
        for pc in parts["bodies"]:
            pc.set_facecolor("#666666")
            pc.set_alpha(0.6)
        zero_ids, low = [], []
        for v, grp, cid in self._data[name][lbl]:
            ax.plot(self._rng.uniform(-0.12, 0.12), v, ".",
                    color=self.COLORS.get(grp, "#aaaaaa"), alpha=0.8, ms=7)
            if v == 0.0:
                zero_ids.append(cid.replace("Case_", ""))
            elif v < self.awful_below:
                low.append((v, cid.replace("Case_", "")))
        for v, cid in sorted(low):
            ax.text(0.16, v, cid, color="white", fontsize=7, va="center")
        if zero_ids:
            ax.text(0.16, 0.0, f"0.0 x{len(zero_ids)}",
                    color="white", fontsize=7, va="center")
            print(f"  {name} {LABEL_NAMES[lbl]}: zero -> "
                  f"{' '.join(sorted(zero_ids))}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])

    @staticmethod
    def _style(fig, axes, title):
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axes:
            ax.set_facecolor("#252525")
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Dice", color="white")
            ax.tick_params(colors="white")
            ax.grid(axis="y", alpha=0.2)
        handles = [plt.Line2D([], [], marker=".", color=c, linestyle="", label=g)
                   for g, c in
                   DiceDistributionPlot.COLORS.items() if g != "Unknown"]
        fig.legend(handles=handles, loc="upper right", facecolor="#2e2e2e",
                   edgecolor="#555555", labelcolor="white")
        fig.suptitle(title, color="white", fontweight="bold")
        fig.tight_layout()

    def save(self, path: str):
        """All runs side by side, one panel per label."""
        if self._data is None:
            self.collect()
        n = len(self.labels)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                                 sharey=True)
        axes = list(np.ravel(np.array(axes, dtype=object)))
        for ax, lbl in zip(axes, self.labels):
            ax.set_facecolor("#252525")
            ax.set_title(f"{LABEL_NAMES[lbl]} Dice", color="white",
                         fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Dice", color="white")
            ax.tick_params(colors="white")
            for xi, name in enumerate(self._names):
                vals = [v for v, _, _ in self._data[name][lbl]]
                if not vals:
                    continue
                parts = ax.violinplot([vals], positions=[xi], showmeans=True,
                                      widths=0.6)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#666666")
                    pc.set_alpha(0.6)
                zero_ids, low = [], []
                for v, grp, cid in self._data[name][lbl]:
                    ax.plot(xi + self._rng.uniform(-0.12, 0.12), v, ".",
                            color=self.COLORS.get(grp, "#aaaaaa"),
                            alpha=0.8, ms=7)
                    if v == 0.0:
                        zero_ids.append(cid.replace("Case_", ""))
                    elif v < self.awful_below:
                        low.append((v, cid.replace("Case_", "")))
                for v, cid in sorted(low):
                    ax.text(xi + 0.16, v, cid, color="white", fontsize=7,
                            va="center")
                if zero_ids:
                    ax.text(xi + 0.16, 0.0, f"0.0 x{len(zero_ids)}",
                            color="white", fontsize=7, va="center")
                    print(f"  {name} {LABEL_NAMES[lbl]}: zero -> "
                          f"{' '.join(sorted(zero_ids))}")
            ax.set_xticks(range(len(self._names)))
            ax.set_xticklabels(self._names, color="white")
            ax.grid(axis="y", alpha=0.2)
        for ax in axes[len(self.labels):]:
            ax.axis("off")
        self._style(fig, [a for a in axes if a.axison],
                    "Per-case Dice by run (post-processed; awful < "
                    f"{self.awful_below} counted)")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
        print(f"saved -> {path}")
        plt.close(fig)

    def save_by_run(self, out_dir: str, prefix: str = "dice"):
        """One figure per run with a panel per label."""
        if self._data is None:
            self.collect()
        n = len(self.labels)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        os.makedirs(out_dir, exist_ok=True)
        for name in self._names:
            if not any(self._data[name][lbl] for lbl in self.labels):
                print(f"skipped -> {prefix}_{name}.png (no predictions on disk)")
                continue
            fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows),
                                     sharey=True)
            axes = list(np.ravel(np.array(axes, dtype=object)))
            for ax, lbl in zip(axes, self.labels):
                ax.set_title(f"{LABEL_NAMES[lbl]} Dice", color="white",
                             fontweight="bold")
                self._panel(ax, name, lbl)
            for ax in axes[len(self.labels):]:
                ax.axis("off")
            self._style(fig, [a for a in axes if a.axison],
                        f"Per-case Dice: {name} (post-processed)")
            path = os.path.join(out_dir, f"{prefix}_{name}.png")
            fig.savefig(path, facecolor=fig.get_facecolor(), dpi=150)
            print(f"saved -> {path}")
            plt.close(fig)
