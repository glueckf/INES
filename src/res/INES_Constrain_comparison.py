

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

def extract_value(label: str) -> int:
    """Extracts, e.g., 2 from 'MaxParents_2.csv'."""
    digits = ''.join(filter(str.isdigit, label))
    if not digits:
        raise ValueError(f"No digits found in label '{label}'")
    return int(digits)

def load_means(files, column):
    means = []
    labels = []
    for f in files:
        df = pd.read_csv(f)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' is missing in file {f}")
        means.append(df[column].mean())
        labels.append(extract_value(os.path.basename(f)))
    return labels, means

def main():
    parser = argparse.ArgumentParser(
        description="Compare Transmission Ratio between MaxParents and Constrain_MaxParents CSV files"
    )
    parser.add_argument('--MNP', nargs='+', required=True, help="CSV files without constraints")
    parser.add_argument('--MNPHybrid', nargs='+', required=True, help="CSV files with constraints")
    parser.add_argument('-y', '--y_column', default="TransmissionRatio",
                        help="Column for the Y-axis (default: TransmissionRatio)")
    parser.add_argument('-o', '--output', default="comparison.png",
                        help="Output filename (extension determines format)")
    args = parser.parse_args()

    labels1, values1 = load_means(args.MNP, args.y_column)
    labels2, values2 = load_means(args.MNPHybrid, args.y_column)

    if labels1 != labels2:
        raise ValueError("The MaxParents values in both groups must be identical (based on filenames).")

    x = np.arange(len(labels1))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, values1, width, label='MNP')
    plt.bar(x + width/2, values2, width, label='MNP Hybrid')

    plt.xlabel("Max Parents", fontsize=26)
    plt.ylabel("Transmission Ratio", fontsize=26)
    plt.xticks(x, labels1, fontsize=21)
    plt.yticks(fontsize=21)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)  # format inferred from file extension

if __name__ == "__main__":
    main()