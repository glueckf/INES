import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to generate bar plot
def plot_bar(files, y_column, labels, output_file, x_label="MaxParents", y_label=None, legend_title="Trend"):
    """
    Generates a bar plot with an optional trend line, custom axis labels, and legend title.

    Parameters:
    - files: List of CSV file paths.
    - y_column: Column name for the y-axis.
    - labels: List of labels for each file.
    - output_file: Path to save the output plot.
    - x_label: Custom label for the x-axis (default is "MaxParents").
    - y_label: Custom label for the y-axis (defaults to y_column if None).
    - legend_title: Title for the legend (default is "Trend").
    """
    # Initialize data for the bar plot
    transmission_ratios = []
    
    # Collect the transmission ratio from each file
    for file in files:
        data = pd.read_csv(file)
        transmission_ratio_mean = data[y_column].mean()  # Calculate the mean of y_column
        transmission_ratios.append(transmission_ratio_mean)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 17})

    # Create positions for the bars
    positions = np.arange(2, len(labels) + 2)
    labels = [int(label.split('_')[1]) for label in labels]

    # Plot bars with the positions aligned with labels
    plt.bar(positions, transmission_ratios, align='center', label=legend_title)
    # Plot the trend line connecting the tops of the bars
    # plt.plot(positions, transmission_ratios, marker='o', linestyle='-', color='red')

    # Set axis labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label if y_label else y_column)
    #plt.title(f'{y_column} for Different {x_label}')

    # Set x-axis labels and ensure alignment
    plt.xticks(positions, labels, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1))

    #plt.legend(title=legend_title)
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig(output_file,format='png')
    plt.close()
    print(f"Plot saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot bar graph from CSV files.")
    parser.add_argument('-i', '--input_files', nargs='+', required=True, help="List of input CSV files.")
    parser.add_argument('-y', '--y_column', required=True, help="Column name for y-axis (e.g., TransmissionRatio)")
    parser.add_argument('-l', '--labels', nargs='+', required=True, help="Labels for the files (e.g., MaxParents_2 MaxParents_3)")
    parser.add_argument('-o', '--output_file', required=True, help="Output file for the bar plot")
    parser.add_argument('--x_label', default="MaxParents", help="Custom label for the x-axis")
    parser.add_argument('--y_label', help="Custom label for the y-axis (defaults to y_column)")
    parser.add_argument('--legend_title', default="Trend", help="Title for the legend")

    args = parser.parse_args()

    # Call the plotting function with customized labels and legend title
    plot_bar(
        args.input_files,
        args.y_column,
        args.labels,
        args.output_file,
        x_label=args.x_label,
        y_label=args.y_label,
        legend_title=args.legend_title
    )

if __name__ == "__main__":
    main()