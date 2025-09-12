import math
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import traceback
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from functools import lru_cache

from INES import INES, SimulationConfig, SimulationMode


def write_final_results(
        integrated_operator_placement_results: Dict[str, Any],
        ines_results: List[Any],
        config: SimulationConfig,
        graph_density: float
) -> None:
    """
    Write final combined simulation results to CSV file.
    Moved from operator_placement_legacy_hook.py for consolidated I/O.
    """
    columns_to_copy = [
        # ID
        "ines_simulation_id",
        "kraken_simulation_id",
        # Configuration parameters
        "network_size",
        "event_skew",
        "node_event_ratio",
        "num_event_types",
        "max_parents",
        "workload_size",
        "query_length",
        "simulation_mode",
        "median_selectivity",
        # Metadata
        "total_projections_placed",
        "placement_difference_to_ines_count",
        "placements_at_cloud",
        "graph_density",
        # Computation times
        "combigen_time_seconds",
        "ines_placement_time_seconds",
        "ines_push_pull_time_seconds",
        "ines_total_time_seconds",
        "kraken_execution_time_seconds",
        # Placement costs
        "all_push_central_cost",
        "inev_cost",
        "ines_cost",
        "kraken_cost",
        # Latency
        "all_push_central_latency",
        "ines_latency",
        "kraken_latency",
    ]

    kraken_metadata = integrated_operator_placement_results["formatted_results"]["metadata"]
    kraken_summary = integrated_operator_placement_results["formatted_results"]["summary"]

    # Extract IDs
    ines_simulation_id = ines_results[0]
    kraken_simulation_id = integrated_operator_placement_results["kraken_simulation_id"]

    # Extract config parameters
    network_size = config.network_size
    event_skew = config.event_skew
    node_event_ratio = config.node_event_ratio
    num_event_types = config.num_event_types
    max_parents = config.max_parents
    workload_size = config.query_size
    query_length = config.query_length
    simulation_mode = config.mode.value
    median_selectivity = ines_results[11]

    # Extract metadata
    total_projections_placed = kraken_summary.get("successful_placements", 0)
    placement_difference_to_ines_count = kraken_summary.get("placement_difference_count", 0)
    placements_at_cloud = _calculate_placements_at_cloud(integrated_operator_placement_results)
    graph_density_value = graph_density

    # Computation times
    combigen_time_seconds = ines_results[12]
    ines_placement_time_seconds = ines_results[14]
    ines_push_pull_time_seconds = ines_results[22]
    ines_total_time_seconds = float(ines_placement_time_seconds) + float(ines_push_pull_time_seconds)
    kraken_execution_time_seconds = kraken_metadata.get("execution_time_seconds", 0)

    # Placement costs
    all_push_central_cost = ines_results[2]
    inev_cost = ines_results[3]
    ines_cost = ines_results[21]
    kraken_cost = kraken_metadata.get("push_pull_plan_cost_sum", 0)

    # Latency
    all_push_central_latency = ines_results[15]
    ines_latency = ines_results[23]
    kraken_latency = kraken_metadata.get("push_pull_plan_latency", 0)

    # Compile all data into a single row
    row_data = {
        # IDs
        "ines_simulation_id": ines_simulation_id,
        "kraken_simulation_id": kraken_simulation_id,
        # Configuration parameters
        "network_size": network_size,
        "event_skew": event_skew,
        "node_event_ratio": node_event_ratio,
        "num_event_types": num_event_types,
        "max_parents": max_parents,
        "workload_size": workload_size,
        "query_length": query_length,
        "simulation_mode": simulation_mode,
        "median_selectivity": median_selectivity,
        # Metadata
        "total_projections_placed": total_projections_placed,
        "placement_difference_to_ines_count": placement_difference_to_ines_count,
        "placements_at_cloud": placements_at_cloud,
        "graph_density": graph_density_value,
        # Computation times
        "combigen_time_seconds": combigen_time_seconds,
        "ines_placement_time_seconds": ines_placement_time_seconds,
        "ines_push_pull_time_seconds": ines_push_pull_time_seconds,
        "ines_total_time_seconds": ines_total_time_seconds,
        "kraken_execution_time_seconds": kraken_execution_time_seconds,
        # Placement costs
        "all_push_central_cost": all_push_central_cost,
        "inev_cost": inev_cost,
        "ines_cost": ines_cost,
        "kraken_cost": kraken_cost,
        # Latency
        "all_push_central_latency": all_push_central_latency,
        "ines_latency": ines_latency,
        "kraken_latency": kraken_latency,
    }

    # Ensure result directory exists
    result_dir = "./kraken/result"
    os.makedirs(result_dir, exist_ok=True)
    csv_file_path = os.path.join(result_dir, "run_results.csv")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file_path)

    # Write to CSV file (append mode)
    with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write headers only if file is new
        if not file_exists:
            writer.writerow(columns_to_copy)

        # Write data row using the order defined in columns_to_copy
        row_values = [row_data[col] for col in columns_to_copy]
        writer.writerow(row_values)

    print(f"[I/O] Appended combined simulation results to {csv_file_path}")
    print(f"[I/O] INES ID: {ines_simulation_id}, Kraken ID: {kraken_simulation_id}")


def _calculate_placements_at_cloud(integrated_operator_placement_results: Dict[str, Any]) -> int:
    """
    Calculate the number of projections placed at the cloud node (node 0).
    Helper function moved from operator_placement_legacy_hook.py.
    """
    placed_at_cloud = 0
    placement_decisions = integrated_operator_placement_results.get(
        "integrated_placement_decision_by_projection", {}
    )
    for projection, decision in placement_decisions.items():
        if hasattr(decision, "node") and decision.node == 0:
            placed_at_cloud += 1
    return placed_at_cloud


@dataclass
class SimulationJob:
    """Represents a single simulation job with all necessary data."""
    job_id: int
    config: SimulationConfig
    parameter_set_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job for multiprocessing."""
        return {
            "job_id": self.job_id,
            "config": self.config,
            "parameter_set_id": self.parameter_set_id
        }


class OptimizedSimulationManager:
    """
    Manages optimized parallel simulation execution with persistent process pool.
    """

    def __init__(
            self,
            enable_parallel: bool = True,
            max_workers: Optional[int] = None,
            batch_size: int = 1
    ):
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers or (multiprocessing.cpu_count() - 1)
        self.batch_size = batch_size

        # Force single worker for sequential mode
        if not enable_parallel:
            self.max_workers = 1

        self.successful_jobs = 0
        self.failed_jobs = 0
        self.start_time = None

        # Batched results for optional batching
        self.result_batch = []

    def run_simulations(self, jobs: List[SimulationJob]) -> None:
        """
        Execute all simulation jobs using single persistent process pool.
        
        Args:
            jobs: List of SimulationJob objects to execute
        """
        total_jobs = len(jobs)
        print(f"[SIMULATION] Starting {total_jobs} jobs with {self.max_workers} workers...")
        print(f"[SIMULATION] Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")

        self.start_time = time.time()

        if self.enable_parallel and self.max_workers > 1:
            self._run_parallel(jobs)
        else:
            self._run_sequential(jobs)

        # Flush any remaining batched results
        self._flush_results()

        # Print final statistics
        total_time = time.time() - self.start_time
        avg_time_per_job = total_time / total_jobs if total_jobs > 0 else 0

        print(
            f"\n[RESULTS] All {total_jobs} jobs completed in {total_time:.1f}s (avg: {avg_time_per_job:.1f}s per job)")
        print(f"[RESULTS] Successful: {self.successful_jobs}, Failed: {self.failed_jobs}")

        if self.failed_jobs > 0:
            print(f"[RESULTS] Warning: {self.failed_jobs} job(s) failed")

    def _run_parallel(self, jobs: List[SimulationJob]) -> None:
        """
        Execute jobs in parallel using single persistent ProcessPoolExecutor.
        """
        # Serialize jobs for multiprocessing
        job_data_list = [job.to_dict() for job in jobs]

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs at once for optimal distribution
            future_to_job_id = {
                executor.submit(run_simulation_worker, job_data): job_data["job_id"]
                for job_data in job_data_list
            }

            # Process results as they complete
            for future in as_completed(future_to_job_id):
                job_id = future_to_job_id[future]
                try:
                    result = future.result()
                    self._process_completed_job(result)

                    # Progress reporting every 100 jobs
                    completed_jobs = self.successful_jobs + self.failed_jobs
                    if completed_jobs % 100 == 0:
                        elapsed = time.time() - self.start_time
                        print(f"[PROGRESS] {completed_jobs}/{len(jobs)} jobs completed - {elapsed:.1f}s elapsed")

                except Exception as e:
                    print(f"[ERROR] Job {job_id} execution failed: {str(e)}")
                    self.failed_jobs += 1

    def _run_sequential(self, jobs: List[SimulationJob]) -> None:
        """
        Execute jobs sequentially for debugging.
        """
        for i, job in enumerate(jobs):
            try:
                print(f"[SEQUENTIAL] Processing job {i + 1}/{len(jobs)} ({job.parameter_set_id})")
                result = run_simulation_worker(job.to_dict())
                self._process_completed_job(result)

            except Exception as e:
                print(f"[ERROR] Job {job.job_id} failed: {str(e)}")
                self.failed_jobs += 1

    def _process_completed_job(self, result: Dict[str, Any]) -> None:
        """
        Process a completed job result and handle I/O.
        """
        if result["success"]:
            # Write results immediately (or batch if batch_size > 1)
            if self.batch_size > 1:
                self.result_batch.append(result)
                if len(self.result_batch) >= self.batch_size:
                    self._flush_results()
            else:
                # Write immediately
                write_final_results(
                    result["integrated_results"],
                    result["ines_results"],
                    result["config"],
                    result["graph_density"]
                )

            self.successful_jobs += 1
        else:
            print(f"[ERROR] Job {result['job_id']} ({result['parameter_set_id']}) failed: {result['error_msg']}")
            self.failed_jobs += 1

    def _flush_results(self) -> None:
        """
        Write any batched results to files.
        """
        if self.result_batch:
            print(f"[I/O] Writing batch of {len(self.result_batch)} results...")
            for result in self.result_batch:
                write_final_results(
                    result["integrated_results"],
                    result["ines_results"],
                    result["config"],
                    result["graph_density"]
                )
            self.result_batch.clear()


def run_simulation_worker(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel simulation execution.
    
    Args:
        job_data: Dictionary containing serialized job data
        
    Returns:
        Dictionary with job results including all data needed for write_final_results
    """
    try:
        job_id = job_data["job_id"]
        config = job_data["config"]
        parameter_set_id = job_data["parameter_set_id"]

        print(f"[WORKER] Starting simulation job {job_id + 1} ({parameter_set_id})")

        # Execute INES simulation
        simulation = INES(config)

        # Calculate graph density
        from INES import calculate_graph_density
        graph_density = calculate_graph_density(simulation.graph)

        # Return complete result data
        return {
            "job_id": job_id,
            "ines_results": simulation.results,
            "integrated_results": simulation.integrated_operator_placement_results,
            "config": config,
            "graph_density": graph_density,
            "success": True,
            "error_msg": None,
            "parameter_set_id": parameter_set_id
        }

    except Exception as e:
        error_message = f"Exception in job {job_data.get('job_id', 'unknown')}: {str(e)}"
        print(f"[WORKER ERROR] {error_message}")
        return {
            "job_id": job_data.get("job_id", -1),
            "ines_results": None,
            "integrated_results": None,
            "config": job_data.get("config"),
            "graph_density": None,
            "success": False,
            "error_msg": error_message,
            "parameter_set_id": job_data.get("parameter_set_id", "unknown")
        }


# @dataclass
# class SimulationRunnerLegacy:
#     """Configuration and execution class for running multiple INES simulations."""
#
#     config: SimulationConfig
#     num_runs: int = 1
#     output_dir: Path = Path("./kraken/result")
#     max_workers: Optional[int] = None
#     batch_size: int = 10
#     enable_parallel: bool = True
#
#     def __post_init__(self):
#         """Initialize output directory and parallel processing parameters."""
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         if self.max_workers is None:
#             self.max_workers = min(multiprocessing.cpu_count() - 1, self.num_runs)
#         if not self.enable_parallel:
#             self.max_workers = 1
#
#     def run_single_simulation(self, run_id: int) -> Optional[pd.DataFrame]:
#         """
#         Execute a single simulation run.
#
#         Args:
#             run_id: Unique identifier for this simulation run
#
#         Returns:
#             DataFrame containing simulation results, or None if simulation failed
#         """
#         try:
#             print(f"[SIMULATION] Run {run_id + 1}/{self.num_runs} started")
#             simulation = INES(self.config)
#             return pd.DataFrame([simulation.results], columns=simulation.schema)
#
#         except Exception as e:
#             error_message = f"Exception in run {run_id}: {str(e)}\n{traceback.format_exc()}"
#             print(f"[ERROR] {error_message}")
#             return None
#
#     def run_simulation_batch(self) -> None:
#         """
#         Execute multiple simulation runs with parallel processing and append results to CSV files.
#
#         Uses multiprocessing to run simulations in parallel for better performance.
#         Results are appended to ines_results.csv as they complete.
#         """
#         ines_filepath = self.output_dir / "ines_results.csv"
#         start_time = time.time()
#
#         print(f"[INES] Starting {self.num_runs} simulation runs with {self.max_workers} workers...")
#         print(f"[CONFIG] Mode: {self.config.mode.value}")
#         print(f"[CONFIG] Network size: {self.config.network_size}")
#         print(f"[CONFIG] Query parameters: size={self.config.query_size}, length={self.config.query_length}")
#         print(f"[CONFIG] Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")
#
#         if self.enable_parallel and self.max_workers > 1:
#             self._run_parallel_simulations(ines_filepath, start_time)
#         else:
#             self._run_sequential_simulations(ines_filepath, start_time)
#
#     def _run_parallel_simulations(self, ines_filepath: Path, start_time: float) -> None:
#         """Execute simulations in parallel using ProcessPoolExecutor."""
#         successful_runs = 0
#         failed_runs = 0
#
#         # Prepare arguments for worker processes
#         worker_args = [(self.config, run_id) for run_id in range(self.num_runs)]
#
#         # Process simulations in batches to manage memory usage
#         for batch_start in range(0, self.num_runs, self.batch_size):
#             batch_end = min(batch_start + self.batch_size, self.num_runs)
#             batch_args = worker_args[batch_start:batch_end]
#
#             print(f"[BATCH] Processing runs {batch_start + 1}-{batch_end} ({len(batch_args)} simulations)")
#
#             with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
#                 # Submit all tasks in this batch
#                 future_to_run = {executor.submit(run_simulation_worker, args): args[1]
#                                  for args in batch_args}
#
#                 # Process completed tasks as they finish
#                 for future in as_completed(future_to_run):
#                     run_id, result_df, success, error_msg = future.result()
#
#                     if success and result_df is not None:
#                         self._append_ines_result(result_df, ines_filepath)
#                         successful_runs += 1
#                         elapsed = time.time() - start_time
#                         print(
#                             f"[PROGRESS] Run {run_id + 1} completed ({successful_runs}/{self.num_runs}) - {elapsed:.1f}s elapsed")
#                     else:
#                         failed_runs += 1
#                         print(f"[ERROR] Run {run_id + 1} failed: {error_msg}")
#
#         self._print_batch_summary(successful_runs, failed_runs, ines_filepath, start_time)
#
#     def _run_sequential_simulations(self, ines_filepath: Path, start_time: float) -> None:
#         """Execute simulations sequentially (fallback mode)."""
#         successful_runs = 0
#         failed_runs = 0
#
#         for run_id in range(self.num_runs):
#             result = self.run_single_simulation(run_id)
#
#             if result is not None:
#                 self._append_ines_result(result, ines_filepath)
#                 successful_runs += 1
#                 elapsed = time.time() - start_time
#                 print(
#                     f"[PROGRESS] Run {run_id + 1} completed ({successful_runs}/{self.num_runs}) - {elapsed:.1f}s elapsed")
#             else:
#                 failed_runs += 1
#
#         self._print_batch_summary(successful_runs, failed_runs, ines_filepath, start_time)
#
#     def _print_batch_summary(self, successful_runs: int, failed_runs: int,
#                              ines_filepath: Path, start_time: float) -> None:
#         """Print summary of batch execution results."""
#         total_time = time.time() - start_time
#         avg_time_per_run = total_time / self.num_runs if self.num_runs > 0 else 0
#
#         print(f"\n[RESULTS] Batch completed in {total_time:.1f}s (avg: {avg_time_per_run:.1f}s per run)")
#         print(f"[RESULTS] {successful_runs} successful runs appended to: {ines_filepath.name}")
#
#         if failed_runs > 0:
#             print(f"[RESULTS] Warning: {failed_runs} run(s) failed")
#
#     def _append_ines_result(self, result_df: pd.DataFrame, filepath: Path) -> None:
#         """
#         Append a single simulation result to the INES results CSV file.
#
#         Args:
#             result_df: DataFrame containing a single simulation result
#             filepath: Path to the INES results CSV file
#         """
#         # Add simulation configuration parameters to the result
#         enhanced_result = result_df.copy()
#
#         # Append to CSV file (create with headers if it doesn't exist)
#         if filepath.exists():
#             enhanced_result.to_csv(filepath, mode='a', header=False, index=False)
#         else:
#             enhanced_result.to_csv(filepath, mode='w', header=True, index=False)


def run_single_simulation(
        network_size: int = 12,
        node_event_ratio: float = 0.5,
        num_event_types: int = 6,
        event_skew: float = 0.3,
        max_parents: int = 1,
        workload_size: int = 3,
        query_length: int = 5,
        num_runs: int = 50,
        mode: SimulationMode = SimulationMode.RANDOM,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None
) -> None:
    """
    Run a single simulation configuration multiple times.
    
    This function creates jobs for the same configuration and executes them
    with the optimized simulation manager.
    
    Args:
        network_size: Number of nodes in the network topology
        node_event_ratio: Ratio of nodes that generate events
        num_event_types: Number of different event types
        event_skew: Skewness parameter for event distribution
        max_parents: Maximum number of parent nodes per node
        workload_size: Number of queries in the workload
        query_length: Average length of each query
        num_runs: Number of simulation runs to execute
        mode: Simulation mode determining what components are fixed/random
        enable_parallel: Whether to enable parallel processing
        max_workers: Maximum number of parallel workers (auto-detected if None)
    """
    print(f"[SINGLE] Starting single simulation configuration with {num_runs} runs")
    print(f"[CONFIG] Network size: {network_size}, Workload: {workload_size}, Query length: {query_length}")
    print(f"[CONFIG] Mode: {mode.value}, Max parents: {max_parents}")

    # Force single worker for debugging when parallel is disabled
    if not enable_parallel:
        max_workers = 1

    # Create simulation configuration
    config = SimulationConfig(
        network_size=network_size,
        node_event_ratio=node_event_ratio,
        num_event_types=num_event_types,
        event_skew=event_skew,
        max_parents=max_parents,
        query_size=workload_size,
        query_length=query_length,
        mode=mode
    )

    # Generate jobs for the same configuration
    parameter_set_id = f"single_n{network_size}_w{workload_size}_q{query_length}_p{max_parents}"
    jobs = []

    for run_id in range(num_runs):
        job = SimulationJob(
            job_id=run_id,
            config=config,
            parameter_set_id=f"{parameter_set_id}_run{run_id + 1}"
        )
        jobs.append(job)

    # Execute with optimized manager
    manager = OptimizedSimulationManager(
        enable_parallel=enable_parallel,
        max_workers=max_workers,
        batch_size=1  # Immediate writing for single simulations
    )

    manager.run_simulations(jobs)
    print(f"[SINGLE] Completed {len(jobs)} runs for single configuration")


def run_parameter_study(
        network_sizes: List[int] = [12],
        workload_sizes: List[int] = [5],
        parent_factors: List[float] = [1.8],
        query_lengths: List[int] = [5],
        runs_per_combination: int = 5,
        node_event_ratio: float = 0.5,
        num_event_types: int = 6,
        event_skew: float = 2.0,
        mode: SimulationMode = SimulationMode.FULLY_DETERMINISTIC,
        enable_parallel: bool = False,
        max_workers: int = 1
) -> None:
    """
    Run a full parameter study with all combinations.
    
    This function generates all parameter combination jobs upfront and executes
    them with a single persistent process pool for maximum efficiency.
    
    Args:
        network_sizes: List of network sizes to test
        workload_sizes: List of workload sizes to test
        parent_factors: List of parent factors to test
        query_lengths: List of query lengths to test
        runs_per_combination: Number of runs per parameter combination
        node_event_ratio: Fixed node event ratio
        num_event_types: Fixed number of event types
        event_skew: Fixed event skew parameter
        mode: Simulation mode
        enable_parallel: Whether to enable parallel processing
        max_workers: Maximum number of parallel workers
    """
    print("[PARAMETER_STUDY] Starting full parameter study")
    print(f"[PARAMETER_STUDY] Network sizes: {network_sizes}")
    print(f"[PARAMETER_STUDY] Workload sizes: {workload_sizes}")
    print(f"[PARAMETER_STUDY] Parent factors: {parent_factors}")
    print(f"[PARAMETER_STUDY] Query lengths: {query_lengths}")
    print(f"[PARAMETER_STUDY] Runs per combination: {runs_per_combination}")

    # Generate ALL jobs upfront
    jobs = []
    job_id = 0

    for parent_factor in parent_factors:
        for workload_size in workload_sizes:
            for network_size in network_sizes:
                for query_length in query_lengths:
                    # Calculate max_parents using same formula as original
                    max_parents = int(parent_factor * math.ceil(math.log2(network_size)))

                    # Create parameter set identifier
                    parameter_set_id = f"n{network_size}_w{workload_size}_q{query_length}_pf{parent_factor}_p{max_parents}"

                    # Create configuration for this parameter combination
                    config = SimulationConfig(
                        network_size=network_size,
                        node_event_ratio=node_event_ratio,
                        num_event_types=num_event_types,
                        event_skew=event_skew,
                        max_parents=max_parents,
                        query_size=workload_size,
                        query_length=query_length,
                        mode=mode
                    )

                    # Generate multiple runs for this combination
                    for run_num in range(runs_per_combination):
                        job = SimulationJob(
                            job_id=job_id,
                            config=config,
                            parameter_set_id=f"{parameter_set_id}_run{run_num + 1}"
                        )
                        jobs.append(job)
                        job_id += 1

    total_jobs = len(jobs)
    expected_jobs = len(network_sizes) * len(workload_sizes) * len(parent_factors) * len(
        query_lengths) * runs_per_combination

    print(f"[PARAMETER_STUDY] Generated {total_jobs} jobs (expected: {expected_jobs})")
    assert total_jobs == expected_jobs, f"Job count mismatch: generated {total_jobs}, expected {expected_jobs}"

    # Execute all jobs with single persistent process pool
    manager = OptimizedSimulationManager(
        enable_parallel=enable_parallel,
        max_workers=max_workers,
        batch_size=10  # Small batching for better progress tracking
    )

    manager.run_simulations(jobs)
    print(f"[PARAMETER_STUDY] Completed all {total_jobs} jobs across all parameter combinations")


# def create_simulation_runner_legacy(
#         network_size: int = 12,
#         node_event_ratio: float = 0.5,
#         num_event_types: int = 6,
#         event_skew: float = 0.3,
#         max_parents: int = 10,
#         workload_size: int = 3,
#         query_length: int = 5,
#         num_runs: int = 1,
#         mode: SimulationMode = SimulationMode.FULLY_DETERMINISTIC,
#         max_workers: Optional[int] = None,
#         batch_size: int = 10,
#         enable_parallel: bool = True
# ) -> SimulationRunnerLegacy:
#     """
#     Create a SimulationRunner with the specified parameters.
#
#     Args:
#         network_size: Number of nodes in the network topology
#         node_event_ratio: Ratio of nodes that generate events
#         num_event_types: Number of different event types
#         event_skew: Skewness parameter for event distribution
#         max_parents: Maximum number of parent nodes per node
#         workload_size: Number of queries in the workload
#         query_length: Average length of each query
#         num_runs: Number of simulation runs to execute
#         mode: Simulation mode determining what components are fixed/random
#         max_workers: Maximum number of parallel workers (auto-detected if None)
#         batch_size: Number of simulations to process per batch
#         enable_parallel: Whether to enable parallel processing
#
#     Returns:
#         Configured SimulationRunner instance
#     """
#     config = SimulationConfig(
#         network_size=network_size,
#         node_event_ratio=node_event_ratio,
#         num_event_types=num_event_types,
#         event_skew=event_skew,
#         max_parents=max_parents,
#         query_size=workload_size,
#         query_length=query_length,
#         mode=mode
#     )
#
#     return SimulationRunnerLegacy(
#         config=config,
#         num_runs=num_runs,
#         max_workers=max_workers,
#         batch_size=batch_size,
#         enable_parallel=enable_parallel
#     )


def main() -> None:
    """
    Main entry point with execution options.
    
    Choose between single simulation debugging or full parameter study.
    """

    # Option 1: Single simulation debugging (commented by default)
    # Uncomment for debugging single configurations
    # run_single_simulation(
    #     network_size=12,
    #     mode=SimulationMode.FULLY_DETERMINISTIC,
    #     enable_parallel=False,
    #     num_runs=1
    # )

    # Option 2: Full parameter study (active by default)
    run_parameter_study(
        network_sizes=[10, 30, 50, 100, 200],
        workload_sizes=[3, 5, 8, 10],
        parent_factors=[0.8, 1.2, 1.8, 2.0],
        query_lengths=[3, 5, 8, 10],
        runs_per_combination=50,
        node_event_ratio=0.5,
        num_event_types=6,
        event_skew=2.0,
        mode=SimulationMode.RANDOM,
        enable_parallel=False,
        max_workers=14
    )


# def main_original() -> None:
#     """
#     Original main function preserved for comparison.
#     This demonstrates the old approach with 320 separate ProcessPoolExecutors.
#     """
#     # Parent factors from p.meran bachelor thesis.
#     OPTIMAL_PARENT_FACTOR = 1.8
#     parent_factors = [0.8, 1.2, 1.8, 2]
#
#     workload_sizes = [3, 5, 8, 10]
#     query_lengths = [3, 5, 8, 10]
#     network_sizes = [10, 30, 50, 100, 200]
#     runs_per_network_size_and_query_length = 50
#     total_runs = 0
#     for parent_factor in parent_factors:
#         for workload_size in workload_sizes:
#             for network_size in network_sizes:
#                 for query_length in query_lengths:
#                     number_of_levels = math.ceil(math.log2(network_size))
#
#                     # First Experiment
#                     max_parents = int(parent_factor * number_of_levels)
#
#                     # Second Experiment
#                     # max_parents = round(OPTIMAL_PARENT_FACTOR * number_of_levels)
#
#                     runner = create_simulation_runner_legacy(
#                         network_size=network_size,
#                         node_event_ratio=0.5,
#                         num_event_types=6,
#                         event_skew=2.0,
#                         max_parents=max_parents,
#                         workload_size=workload_size,
#                         query_length=query_length,
#                         num_runs=runs_per_network_size_and_query_length,
#                         mode=SimulationMode.RANDOM,
#                         max_workers=14,
#                         batch_size=50,
#                         enable_parallel=True
#                     )
#                     runner.run_simulation_batch()
#                     total_runs += runs_per_network_size_and_query_length
#                     print(f"[TOTAL] Completed {total_runs} total simulation runs so far.")
#
#     # runner = create_simulation_runner_legacy(
#     #     network_size=12,
#     #     node_event_ratio=0.5,
#     #     num_event_types=6,
#     #     event_skew=0.3,
#     #     max_parents=10,
#     #     workload_size=3,
#     #     query_length=5,
#     #     num_runs=1,
#     #     mode=SimulationMode.FULLY_DETERMINISTIC,
#     #     max_workers=1,
#     #     batch_size=1,
#     #     enable_parallel=False
#     # )
#     #
#     # runner.run_simulation_batch()


if __name__ == "__main__":
    main()
