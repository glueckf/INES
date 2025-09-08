from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import traceback
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
from functools import lru_cache

from INES import INES, SimulationConfig, SimulationMode


def run_simulation_worker(args):
    """
    Worker function for parallel simulation execution.
    
    This function is designed to be pickled and run in separate processes.
    
    Args:
        args: Tuple containing (config, run_id)
        
    Returns:
        Tuple of (run_id, result_df, success_flag, error_msg)
    """
    config, run_id = args
    try:
        print(f"[WORKER] Starting simulation run {run_id + 1}")
        simulation = INES(config)
        result_df = pd.DataFrame([simulation.results], columns=simulation.schema)
        return (run_id, result_df, True, None)
    except Exception as e:
        error_message = f"Exception in run {run_id}: {str(e)}"
        print(f"[WORKER ERROR] {error_message}")
        return (run_id, None, False, error_message)


@dataclass
class SimulationRunner:
    """Configuration and execution class for running multiple INES simulations."""

    config: SimulationConfig
    num_runs: int = 1
    output_dir: Path = Path("./kraken/result")
    max_workers: Optional[int] = None
    batch_size: int = 10
    enable_parallel: bool = True

    def __post_init__(self):
        """Initialize output directory and parallel processing parameters."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.max_workers is None:
            self.max_workers = min(multiprocessing.cpu_count() - 1, self.num_runs)
        if not self.enable_parallel:
            self.max_workers = 1

    def run_single_simulation(self, run_id: int) -> Optional[pd.DataFrame]:
        """
        Execute a single simulation run.
        
        Args:
            run_id: Unique identifier for this simulation run
            
        Returns:
            DataFrame containing simulation results, or None if simulation failed
        """
        try:
            print(f"[SIMULATION] Run {run_id + 1}/{self.num_runs} started")
            simulation = INES(self.config)
            return pd.DataFrame([simulation.results], columns=simulation.schema)

        except Exception as e:
            error_message = f"Exception in run {run_id}: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_message}")
            return None

    def run_simulation_batch(self) -> None:
        """
        Execute multiple simulation runs with parallel processing and append results to CSV files.
        
        Uses multiprocessing to run simulations in parallel for better performance.
        Results are appended to ines_results.csv as they complete.
        """
        ines_filepath = self.output_dir / "ines_results.csv"
        start_time = time.time()

        print(f"[INES] Starting {self.num_runs} simulation runs with {self.max_workers} workers...")
        print(f"[CONFIG] Mode: {self.config.mode.value}")
        print(f"[CONFIG] Network size: {self.config.network_size}")
        print(f"[CONFIG] Query parameters: size={self.config.query_size}, length={self.config.query_length}")
        print(f"[CONFIG] Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")

        if self.enable_parallel and self.max_workers > 1:
            self._run_parallel_simulations(ines_filepath, start_time)
        else:
            self._run_sequential_simulations(ines_filepath, start_time)

    def _run_parallel_simulations(self, ines_filepath: Path, start_time: float) -> None:
        """Execute simulations in parallel using ProcessPoolExecutor."""
        successful_runs = 0
        failed_runs = 0

        # Prepare arguments for worker processes
        worker_args = [(self.config, run_id) for run_id in range(self.num_runs)]

        # Process simulations in batches to manage memory usage
        for batch_start in range(0, self.num_runs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_runs)
            batch_args = worker_args[batch_start:batch_end]

            print(f"[BATCH] Processing runs {batch_start + 1}-{batch_end} ({len(batch_args)} simulations)")

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks in this batch
                future_to_run = {executor.submit(run_simulation_worker, args): args[1]
                                 for args in batch_args}

                # Process completed tasks as they finish
                for future in as_completed(future_to_run):
                    run_id, result_df, success, error_msg = future.result()

                    if success and result_df is not None:
                        self._append_ines_result(result_df, ines_filepath)
                        successful_runs += 1
                        elapsed = time.time() - start_time
                        print(
                            f"[PROGRESS] Run {run_id + 1} completed ({successful_runs}/{self.num_runs}) - {elapsed:.1f}s elapsed")
                    else:
                        failed_runs += 1
                        print(f"[ERROR] Run {run_id + 1} failed: {error_msg}")

        self._print_batch_summary(successful_runs, failed_runs, ines_filepath, start_time)

    def _run_sequential_simulations(self, ines_filepath: Path, start_time: float) -> None:
        """Execute simulations sequentially (fallback mode)."""
        successful_runs = 0
        failed_runs = 0

        for run_id in range(self.num_runs):
            result = self.run_single_simulation(run_id)

            if result is not None:
                self._append_ines_result(result, ines_filepath)
                successful_runs += 1
                elapsed = time.time() - start_time
                print(
                    f"[PROGRESS] Run {run_id + 1} completed ({successful_runs}/{self.num_runs}) - {elapsed:.1f}s elapsed")
            else:
                failed_runs += 1

        self._print_batch_summary(successful_runs, failed_runs, ines_filepath, start_time)

    def _print_batch_summary(self, successful_runs: int, failed_runs: int,
                             ines_filepath: Path, start_time: float) -> None:
        """Print summary of batch execution results."""
        total_time = time.time() - start_time
        avg_time_per_run = total_time / self.num_runs if self.num_runs > 0 else 0

        print(f"\n[RESULTS] Batch completed in {total_time:.1f}s (avg: {avg_time_per_run:.1f}s per run)")
        print(f"[RESULTS] {successful_runs} successful runs appended to: {ines_filepath.name}")

        if failed_runs > 0:
            print(f"[RESULTS] Warning: {failed_runs} run(s) failed")

    def _append_ines_result(self, result_df: pd.DataFrame, filepath: Path) -> None:
        """
        Append a single simulation result to the INES results CSV file.
        
        Args:
            result_df: DataFrame containing a single simulation result
            filepath: Path to the INES results CSV file
        """
        # Add simulation configuration parameters to the result
        enhanced_result = result_df.copy()

        # Append to CSV file (create with headers if it doesn't exist)
        if filepath.exists():
            enhanced_result.to_csv(filepath, mode='a', header=False, index=False)
        else:
            enhanced_result.to_csv(filepath, mode='w', header=True, index=False)


def create_simulation_runner(
        network_size: int = 12,
        node_event_ratio: float = 0.5,
        num_event_types: int = 6,
        event_skew: float = 0.3,
        max_parents: int = 10,
        workload_size: int = 3,
        query_length: int = 5,
        num_runs: int = 1,
        mode: SimulationMode = SimulationMode.FULLY_DETERMINISTIC,
        max_workers: Optional[int] = None,
        batch_size: int = 10,
        enable_parallel: bool = True
) -> SimulationRunner:
    """
    Create a SimulationRunner with the specified parameters.
    
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
        max_workers: Maximum number of parallel workers (auto-detected if None)
        batch_size: Number of simulations to process per batch
        enable_parallel: Whether to enable parallel processing
        
    Returns:
        Configured SimulationRunner instance
    """
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

    return SimulationRunner(
        config=config,
        num_runs=num_runs,
        max_workers=max_workers,
        batch_size=batch_size,
        enable_parallel=enable_parallel
    )


def main() -> None:
    """Main entry point for simulation execution."""

    network_sizes = [10, 30, 50, 100, 200, 500]
    query_lengths = [3, 5, 8, 10]
    runs_per_network_size_and_query_length = 50
    total_runs = 0

    for size in network_sizes:
        for query_length in query_lengths:
            runner = create_simulation_runner(
                network_size=size,
                node_event_ratio=0.5,
                num_event_types=6,
                event_skew=2.0,
                max_parents=5,
                query_size=5,
                query_length=query_length,
                num_runs=runs_per_network_size_and_query_length,
                mode=SimulationMode.RANDOM,
                max_workers=14,
                batch_size=20,
                enable_parallel=True
            )
            runner.run_simulation_batch()
            total_runs += runs_per_network_size_and_query_length
            print(f"[TOTAL] Completed {total_runs} total simulation runs so far.")

if __name__ == "__main__":
    main()
