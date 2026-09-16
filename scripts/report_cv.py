"""Full CV analysis across runs (calls src.plotter).

Writes the per-case Dice distribution figure and the bad-performers table.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plotter import BadPerformersTable, CVData, DiceDistributionPlot

GT_ROOT = "var/emidec_dataset/train"
SPLITS = "var/nnUNet_validation/splits_final.json"
RUNS = [
    ("plain", "var/nnUNet_validation_pp"),
    ("resenc", "var/nnUNet_validation_500resenc_pp"),
    ("regions", "var/nnUNet_validation_501_pp"),
]
FIGURE_OUT = "var/results/dice_distributions.png"
TABLE_OUT = "var/results/bad_performers.md"


def main():
    cv = CVData(GT_ROOT, SPLITS, RUNS)
    plot = DiceDistributionPlot(cv)
    plot.save(FIGURE_OUT)
    plot.save_by_run("var/results", prefix="dice")
    BadPerformersTable(cv).save(TABLE_OUT)


if __name__ == "__main__":
    main()
