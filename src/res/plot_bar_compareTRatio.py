#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot bar graph from CSV files with multiple Y values.

Original Author: samira
Modified by Assistant to support multiple Y columns and custom legend tags.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to generate bar plot
def plot_bar(files, y_columns, labels, output_file, x_label='Max Parents', y_label='Mean Values', legend_tags=None):
    # Check that the number of input files and labels match
    if len(files) != len(labels):
        raise ValueError("The number of input files and labels must be the same.")

    # If legend tags are provided, check that their count matches y_columns
    if legend_tags and len(legend_tags) != len(y_columns):
        raise ValueError("The number of legend tags must match the number of y_columns.")

    # Initialize data structure for the bar plot
    data_means = []

    # Collect the mean of each y_column from each file
    for file in files:
        data = pd.read_csv(file)
        means = []
        for y_col in y_columns:
            if y_col not in data.columns:
                raise ValueError(f"Column '{y_col}' not found in file '{file}'.")
            mean_value = data[y_col].mean()
            means.append(mean_value)
        data_means.append(means)

    # Number of groups and bars
    n_groups = len(labels)
    n_bars = len(y_columns)

    # Create positions for the bars
    total_width = 0.8
    bar_width = total_width / n_bars
    positions = np.arange(n_groups)

    # Adjust labels if needed (e.g., extract numeric part from 'MaxParents_2')
    x_labels = [int(label.split('_')[1]) if '_' in label else label for label in labels]

    # Set up the plot
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 18})

    # Colors for the bars
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if n_bars > len(colors):
        colors *= (n_bars // len(colors) + 1)

    # Plot each set of bars
    for i in range(n_bars):
        offset = (i - n_bars / 2) * bar_width + bar_width / 2
        bar_positions = positions + offset
        means = [data_means[j][i] for j in range(n_groups)]
        legend_label = legend_tags[i] if legend_tags else y_columns[i]

        plt.bar(bar_positions, means, width=bar_width, label=legend_label, color=colors[i % len(colors)], align='center')
        # plt.plot(bar_positions, means, marker='o', linestyle='-', color='red')

    plt.xlabel(x_label,fontsize=25)
    plt.ylabel(y_label,fontsize=25)
    
    # Set x-axis labels
    plt.xticks(positions, x_labels, ha='right',fontsize=23)

    # Set y-axis ticks
    plt.yticks(np.linspace(0, 1, num=11),fontsize=23)

    # Add legend with custom tags if provided
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure v a file
    plt.savefig(output_file, format='png', bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot bar graph from CSV files with multiple Y values.")
    parser.add_argument('-i', '--input_files', nargs='+', required=True, help="List of input CSV files.")
    parser.add_argument('-y', '--y_columns', nargs='+', required=True, help="Column names for y-axis (e.g., TransmissionRatio TransmissionRatioINEv)")
    parser.add_argument('-l', '--labels', nargs='+', required=True, help="Labels for the files (e.g., MaxParents_2 MaxParents_3)")
    parser.add_argument('-o', '--output_file', required=True, help="Output file for the bar plot")
    parser.add_argument('--x_label', default='Max Parents', help="Label for the x-axis")
    parser.add_argument('--y_label', default='Mean Values', help="Label for the y-axis")
    parser.add_argument('--legend_tags', nargs='+', help="Custom legend tags for each y-column")

    args = parser.parse_args()

    # Call the plotting function with legend tags if provided
    plot_bar(args.input_files, args.y_columns, args.labels, args.output_file, args.x_label, args.y_label, args.legend_tags)

if __name__ == "__main__":
    main()