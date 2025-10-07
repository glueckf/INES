#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:43:03 2021
@author: samira
Modified by Assistant
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_percentile_bars(
    input_files,
    y_columns,
    x_column,
    labels,
    output_file,
    colors=None,
    x_label=None,
    y_label=None,
    legend_title="Data Sets",
):
    if len(labels) != len(input_files) * len(y_columns):
        raise ValueError(
            "Number of labels must match the number of (input file × y_column) combinations."
        )

    n_combinations = len(input_files) * len(y_columns)
    if colors and len(colors) != n_combinations:
        raise ValueError(
            "The number of colors must match the number of (input file, Y column) combinations."
        )

    if not colors:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] * (
            n_combinations // len(plt.rcParams["axes.prop_cycle"].by_key()["color"]) + 1
        )

    data_dict = {}

    for idx, file in enumerate(input_files):
        df = pd.read_csv(file)
        for y_col in y_columns:
            label = f"{labels[idx]} - {y_col}"
            grouped = df.groupby(x_column)[y_col]
            for x_point, group in grouped:
                key = (x_point, label)
                p10 = np.percentile(group, 10)
                p90 = np.percentile(group, 90)
                min_value = group.min()
                max_value = group.max()
                data_dict[key] = {
                    "p10": p10,
                    "p90": p90,
                    "min": min_value,
                    "max": max_value,
                }

    x_points = sorted(set([key[0] for key in data_dict.keys()]))
    positions = np.arange(len(x_points))
    total_width = 0.8
    bar_width = total_width / n_combinations

    bar_positions = []
    bar_heights = []
    bar_bottoms = []
    lower_whisker = []
    upper_whisker = []
    legend_labels = []

    plt.rcParams.update({"font.size": 17})

    for i in range(n_combinations):
        idx_file = i // len(y_columns)
        idx_y = i % len(y_columns)
        label = f"{labels[idx_file]} - {y_columns[idx_y]}"
        legend_labels.append(label)

        positions_i = positions - total_width / 2 + bar_width / 2 + i * bar_width
        bar_positions.append(positions_i)

        heights, bottoms, err_lower, err_upper = [], [], [], []

        for x_point in x_points:
            key = (x_point, label)
            if key in data_dict:
                p10 = data_dict[key]["p10"]
                p90 = data_dict[key]["p90"]
                min_value = data_dict[key]["min"]
                max_value = data_dict[key]["max"]
                bar_height = p90 - p10
                minimal_height = 0.015
                if round(bar_height, 2) == 0.0:
                    heights.append(minimal_height)
                    bottoms.append(p10 - minimal_height / 2)
                else:
                    heights.append(bar_height)
                    bottoms.append(p10)
                err_lower_value = p10 - min_value
                err_upper_value = max_value - p90
                err_lower.append(err_lower_value if err_lower_value != 0 else None)
                err_upper.append(err_upper_value if err_upper_value != 0 else None)
            else:
                heights.append(np.nan)
                bottoms.append(np.nan)
                err_lower.append(None)
                err_upper.append(None)

        bar_heights.append(heights)
        bar_bottoms.append(bottoms)
        lower_whisker.append(err_lower)
        upper_whisker.append(err_upper)

    fig, ax = plt.subplots(figsize=(14, 7))
    capwidth = bar_width * 0.4

    for i in range(n_combinations):
        x_pos = bar_positions[i]
        heights = bar_heights[i]
        bottoms = bar_bottoms[i]
        lower = lower_whisker[i]
        upper = upper_whisker[i]

        ax.bar(
            x_pos,
            heights,
            width=bar_width,
            bottom=bottoms,
            color=colors[i],
            label=legend_labels[i],
            zorder=2,
        )

        for j in range(len(x_pos)):
            if not np.isnan(bottoms[j]):
                if lower[j] is not None:
                    ax.vlines(
                        x_pos[j],
                        bottoms[j] - lower[j],
                        bottoms[j],
                        color="black",
                        zorder=3,
                    )
                    ax.hlines(
                        bottoms[j] - lower[j],
                        x_pos[j] - capwidth / 2,
                        x_pos[j] + capwidth / 2,
                        color="black",
                        zorder=3,
                    )
                if upper[j] is not None:
                    ax.vlines(
                        x_pos[j],
                        bottoms[j] + heights[j],
                        bottoms[j] + heights[j] + upper[j],
                        color="black",
                        zorder=3,
                    )
                    ax.hlines(
                        bottoms[j] + heights[j] + upper[j],
                        x_pos[j] - capwidth / 2,
                        x_pos[j] + capwidth / 2,
                        color="black",
                        zorder=3,
                    )

    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 1, 11)], fontsize=23)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(x) for x in x_points], fontsize=23)
    plt.xlabel(x_label if x_label else x_column, fontsize=25)
    plt.ylabel(y_label if y_label else ", ".join(y_columns), fontsize=25)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=16,
        title=legend_title,
    )
    plt.tight_layout()
    plt.savefig(output_file, format="png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a bar graph with 80% range bars and outliers."
    )
    parser.add_argument(
        "-i", "--input_files", nargs="+", required=True, help="Input CSV files."
    )
    parser.add_argument(
        "-y", "--y_column", nargs="+", required=True, help="Y-axis column name(s)."
    )
    parser.add_argument("-x", "--x_column", required=True, help="X-axis column name.")
    parser.add_argument(
        "-l", "--labels", nargs="+", required=True, help="Labels for the input files."
    )
    parser.add_argument(
        "-o", "--output_file", required=True, help="Output file for the plot."
    )
    parser.add_argument(
        "-c", "--colors", nargs="+", help="Colors for the bars (optional)."
    )
    parser.add_argument("--x_label", help="Custom x-axis label (optional).")
    parser.add_argument("--y_label", help="Custom y-axis label (optional).")
    parser.add_argument(
        "--legend_title", default="Legend", help="Title for the legend (optional)."
    )

    args = parser.parse_args()

    plot_percentile_bars(
        args.input_files,
        args.y_column,
        args.x_column,
        args.labels,
        args.output_file,
        colors=args.colors,
        x_label=args.x_label,
        y_label=args.y_label,
        legend_title=args.legend_title,
    )


if __name__ == "__main__":
    main()
