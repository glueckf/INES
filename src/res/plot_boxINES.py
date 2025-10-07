#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original Author: Meran
https://github.com/PMeran101/INEv/blob/main/INEv/res/multi_variate_PP_bar.py
Modified by Nur Ali
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os


def plot_percentile_bars(
    input_files,
    y_column,
    x_column,
    labels,
    output_file,
    colors=None,
    hatches=None,
    styles=None,
    x_label=None,
    y_label=None,
    legend_title="Data Sets",
):
    if len(input_files) != len(labels):
        raise ValueError("Number of input files and labels must match.")

    if colors and len(colors) != len(labels):
        raise ValueError("Number of colors must match number of labels.")

    if hatches and len(hatches) != len(labels):
        raise ValueError("Number of hatches must match number of labels.")

    if styles and len(styles) != len(labels):
        raise ValueError("Number of styles must match number of labels.")

    # Defaults
    if not colors:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 2
    if not hatches:
        hatches = [""] * len(labels)
    if not styles:
        styles = ["-"] * len(labels)

    # Load data
    data_dict = {}
    for idx, file in enumerate(input_files):
        label = labels[idx]
        df = pd.read_csv(file)
        grouped = df.groupby(x_column)[y_column]
        for x_val, group in grouped:
            key = (x_val, label)
            data_dict[key] = {
                "p10": np.percentile(group, 10),
                "p90": np.percentile(group, 90),
                "min": group.min(),
                "max": group.max(),
            }

    # Setup
    x_points = sorted(set(k[0] for k in data_dict))
    positions = np.arange(len(x_points))
    n_labels = len(labels)
    total_width = 0.8
    bar_width = total_width / n_labels
    capwidth = bar_width * 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.rcParams.update({"font.size": 17})

    legend_elements = []

    for i, label in enumerate(labels):
        x_offsets = positions - total_width / 2 + bar_width / 2 + i * bar_width
        color = colors[i]
        hatch = hatches[i]
        linestyle = styles[i]

        heights, bottoms, err_lower, err_upper = [], [], [], []

        for x in x_points:
            key = (x, label)
            stats = data_dict.get(key)
            if stats:
                p10, p90 = stats["p10"], stats["p90"]
                h = max(p90 - p10, 0.015)
                b = p10 if h > 0.015 else p10 - h / 2
                heights.append(h)
                bottoms.append(b)
                err_lower.append(p10 - stats["min"] if p10 > stats["min"] else None)
                err_upper.append(stats["max"] - p90 if p90 < stats["max"] else None)
            else:
                heights.append(np.nan)
                bottoms.append(np.nan)
                err_lower.append(None)
                err_upper.append(None)

        # Bars
        ax.bar(
            x_offsets,
            heights,
            width=bar_width,
            bottom=bottoms,
            color=color,
            hatch=hatch,
            edgecolor="black",
            align="center",
            zorder=2,
        )

        # Whiskers
        for j, x_pos in enumerate(x_offsets):
            if np.isnan(bottoms[j]):
                continue

            # Lower
            if err_lower[j] is not None:
                ax.vlines(
                    x_pos,
                    bottoms[j] - err_lower[j],
                    bottoms[j],
                    color="black",
                    linestyle=linestyle,
                    zorder=3,
                )
                ax.hlines(
                    bottoms[j] - err_lower[j],
                    x_pos - capwidth / 2,
                    x_pos + capwidth / 2,
                    color="black",
                    zorder=3,
                )

            # Upper
            if err_upper[j] is not None:
                ax.vlines(
                    x_pos,
                    bottoms[j] + heights[j],
                    bottoms[j] + heights[j] + err_upper[j],
                    color="black",
                    linestyle=linestyle,
                    zorder=3,
                )
                ax.hlines(
                    bottoms[j] + heights[j] + err_upper[j],
                    x_pos - capwidth / 2,
                    x_pos + capwidth / 2,
                    color="black",
                    zorder=3,
                )

        # Legend patch
        legend_elements.append(
            Patch(facecolor=color, edgecolor="black", hatch=hatch, label=label)
        )

    # Axis
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in x_points], fontsize=23)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 11)], fontsize=23)

    if x_label:
        ax.set_xlabel(x_label, fontsize=25)
    if y_label:
        ax.set_ylabel(y_label, fontsize=25)

    ax.set_ylim(0, 1.05)
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format="png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Percentile bar plot with color, hatch and line style."
    )
    parser.add_argument("-i", "--input_files", nargs="+", required=True)
    parser.add_argument("-y", "--y_column", required=True)
    parser.add_argument("-x", "--x_column", required=True)
    parser.add_argument("-l", "--labels", nargs="+", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-c", "--colors", nargs="+")
    parser.add_argument("-t", "--hatches", nargs="+")
    parser.add_argument(
        "-s", "--styles", nargs="+", help="Line styles for whiskers (e.g. '-', '--')"
    )
    parser.add_argument("--x_label")
    parser.add_argument("--y_label")
    parser.add_argument("--legend_title", default="Legend")

    args = parser.parse_args()

    plot_percentile_bars(
        args.input_files,
        args.y_column,
        args.x_column,
        args.labels,
        args.output_file,
        colors=args.colors,
        hatches=args.hatches,
        styles=args.styles,
        x_label=args.x_label,
        y_label=args.y_label,
        legend_title=args.legend_title,
    )


if __name__ == "__main__":
    main()
