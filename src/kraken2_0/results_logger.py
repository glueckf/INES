"""
Results Logger for Detailed Placement Decision Tracking

This module handles Parquet file operations for logging placement decisions
and run results. It provides high-performance, scalable storage for algorithm
analysis and debugging using the Apache Parquet columnar format.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Dict, List

# Header Definitions (maintain exact column names for compatibility)
DETAILED_LOG_HEADER = [
    "run_id",
    "strategy_name",
    "workload",
    "processing_order",
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

RUN_RESULTS_HEADER = [
    "run_id",
    "strategy_name",
    "workload",
    "total_cost",
    "workload_cost",
    "max_latency",
    "num_placements",
    "placements_at_cloud",
    "average_cost_per_placement",
    "execution_time_seconds",
    "status",
]

OUTPUT_DIRECTORY = "kraken2_0/result"
RUN_RESULTS_DIR = "run_results.parquet"
DETAILED_LOG_DIR = "detailed_run_log.parquet"


def initialize_logging() -> None:
    """
    Ensure base directories for Parquet datasets exist.

    Creates the output directory structure for both run results
    and detailed logs if they don't already exist. This function
    should be called once at application startup.

    Raises:
        OSError: If directory creation fails.
    """
    base_path = Path(OUTPUT_DIRECTORY)
    (base_path / RUN_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    (base_path / DETAILED_LOG_DIR).mkdir(parents=True, exist_ok=True)


def write_detailed_log(run_id: str, detailed_log_data: List[Dict[str, Any]]) -> None:
    """
    Write detailed log entries for a single run to a dedicated Parquet file.

    Each run gets its own Parquet file named by run_id for easy identification
    and efficient querying.

    Args:
        run_id: Unique identifier for this solver run.
        detailed_log_data: List of log entry dictionaries containing
            the fields defined in DETAILED_LOG_HEADER.

    Raises:
        OSError: If file writing fails.
        ValueError: If log entries are missing required fields.
    """
    if not detailed_log_data:
        return

    df = pd.DataFrame(detailed_log_data)
    df = df[DETAILED_LOG_HEADER]

    # Convert projection objects to strings for Parquet compatibility
    df["projection"] = df["projection"].astype(str)

    file_path = Path(OUTPUT_DIRECTORY) / DETAILED_LOG_DIR / f"{run_id}.parquet"

    df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)


def write_run_results(results_data: List[Dict[str, Any]]) -> None:
    """
    Append one or more run results to the run_results Parquet dataset.

    Results are written as a new Parquet file in the dataset directory,
    enabling efficient append operations without rewriting existing data.

    Args:
        results_data: List of result dictionaries containing the fields
            defined in RUN_RESULTS_HEADER.

    Raises:
        OSError: If file writing fails.
        ValueError: If result entries are missing required fields.
    """
    if not results_data:
        return

    df = pd.DataFrame(results_data)
    df = df[RUN_RESULTS_HEADER]

    table = pa.Table.from_pandas(df, preserve_index=False)

    output_dir = Path(OUTPUT_DIRECTORY) / RUN_RESULTS_DIR

    pq.write_to_dataset(table, root_path=str(output_dir))
