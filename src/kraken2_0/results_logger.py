"""
Results Logger for Detailed Placement Decision Tracking

This module handles CSV file operations for logging every placement decision
considered by the solver. When enabled, it provides detailed forensic data
for algorithm analysis and debugging.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List

# CSV Header Definition
CSV_HEADER = [
    "run_id",
    "strategy_name",
    "projection_index",
    "projection",
    "candidate_node",
    "communication_strategy",
    "individual_cost",
    "transmission_latency",
    "processing_latency",
    "cumulative_latency_so_far",
    "is_pruned",
    "is_part_of_final_solution",
]

# Output Configuration
OUTPUT_DIRECTORY = "result"
OUTPUT_FILENAME = "detailed_run_log.csv"


def initialize_detailed_csv() -> None:
    """
    Initialize the CSV file with headers if it doesn't already exist.

    Creates the output directory if needed and writes the CSV header row.
    This operation is idempotent - it will not overwrite an existing file.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    output_path = Path(OUTPUT_DIRECTORY)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file_path = output_path / OUTPUT_FILENAME

    if not csv_file_path.exists():
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()


def write_detailed_log(run_id: str, detailed_log_data: List[Dict[str, Any]]) -> None:
    """
    Write detailed log entries to CSV file in append mode.

    Args:
        run_id: Unique identifier for this solver run.
        detailed_log_data: List of log entry dictionaries, each containing
            the fields defined in CSV_HEADER.

    Raises:
        OSError: If file writing fails.
        ValueError: If log entries are missing required fields.
    """
    csv_file_path = Path(OUTPUT_DIRECTORY) / OUTPUT_FILENAME

    with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(detailed_log_data)
