import math
import csv
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import logging
import threading

from INES import INES, SimulationConfig, SimulationMode

logger = logging.getLogger(__name__)


def _safe_float_convert(value: Any) -> float:
    """Safely convert value to float, handling numpy types and their string representations."""
    if value is None:
        return 0.0

    # If it's already a number, convert directly
    if isinstance(value, (int, float)):
        return float(value)

    # If it's a numpy type, get the item value
    if hasattr(value, 'item'):
        return float(value.item())

    # If it's a string representation of numpy type like "np.int64(647)"
    if isinstance(value, str) and 'np.' in value:
        import re
        match = re.search(r'\(([^)]+)\)', value)
        if match:
            return float(match.group(1))

    # Last resort: direct conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} (type: {type(value)}) to float, returning 0.0")
        return 0.0


# Global worker state for persistent initialization
_worker_state = {}
_worker_lock = threading.Lock()


def _setup_worker_path(parent_dir: str = None) -> None:
    """Set up sys.path for worker processes BEFORE any unpickling."""
    import sys
    if parent_dir is None:
        # Fallback: compute from __file__ if available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"[WORKER_PATH_SETUP] Added {parent_dir} to sys.path", flush=True)


def _initialize_worker() -> None:
    """
    Initialize worker process with expensive one-time setup.
    This runs once per worker process and caches expensive operations.
    """
    global _worker_state

    # Ensure sys.path is set up (should already be done by initializer, but double-check)
    _setup_worker_path()

    with _worker_lock:
        if "initialized" not in _worker_state:
            logger.info("[WORKER_INIT] Initializing worker process")

            # Pre-import heavy modules to avoid repeated imports
            try:
                import networkx as nx
                import numpy as np
                from INES import INES, calculate_graph_density

                _worker_state.update({
                    "initialized": True,
                    "networkx": nx,
                    "numpy": np,
                    "INES": INES,
                    "calculate_graph_density": calculate_graph_density,
                    "job_count": 0,
                })

                logger.info("[WORKER_INIT] Worker initialization completed")
            except ImportError as e:
                logger.error(f"[WORKER_INIT] Failed to import modules: {e}")
                raise


def _log_detailed_placement_decisions(
    integrated_operator_placement_results: Dict[str, Any],
    config: SimulationConfig,
    kraken_latency: float,
    ines_latency: float,
    graph_data: Optional[Any] = None,
) -> None:
    """
    Log detailed placement decisions for both INES and Kraken when
    Kraken latency is significantly higher than INES latency.

    Args:
        integrated_operator_placement_results: Kraken placement results
        config: Simulation configuration
        kraken_latency: Kraken total latency
        ines_latency: INES total latency
    """
    try:
        # Extract placements from new structure
        best_solution = integrated_operator_placement_results.get("best_solution", {})
        solution = best_solution.get("solution")

        if solution is None or not hasattr(solution, "placements"):
            logger.warning("[PLACEMENT_ANALYSIS] No placements available in results")
            return

        placements = solution.placements

        logger.info(
            f"[PLACEMENT_ANALYSIS] Detailed placement decisions (Kraken latency {kraken_latency:.3f} >= 1.5x INES {ines_latency:.3f}):"
        )
        logger.info(
            f"[PLACEMENT_ANALYSIS] Network size: {config.network_size}, Workload size: {config.query_size}"
        )

        # Log summary statistics
        total_placements = len(placements)
        cloud_placements = sum(
            1 for placement_info in placements.values()
            if placement_info.node == 0
        )
        fog_placements = total_placements - cloud_placements

        logger.info(f"[PLACEMENT_ANALYSIS] Total placements: {total_placements}")
        logger.info(
            f"[PLACEMENT_ANALYSIS] Cloud placements (node 0): {cloud_placements}"
        )
        logger.info(f"[PLACEMENT_ANALYSIS] Fog placements: {fog_placements}")

        # Log each placement decision in detail
        for i, (projection, placement_info) in enumerate(placements.items(), 1):
            node = placement_info.node
            cost = placement_info.individual_cost
            strategy = placement_info.strategy

            # Calculate hops from cloud (node 0)
            hops_from_cloud = "unknown"
            if isinstance(node, int):
                if node == 0:
                    hops_from_cloud = 0  # Cloud node
                else:
                    # Try to calculate actual hops if graph data is available
                    if graph_data is not None:
                        try:
                            import networkx as nx

                            if hasattr(graph_data, "shortest_path_length"):
                                hops_from_cloud = nx.shortest_path_length(
                                    graph_data, source=0, target=node
                                )
                            else:
                                hops_from_cloud = (
                                    1  # Fallback: assume direct connection to fog
                                )
                        except (ImportError, Exception):
                            hops_from_cloud = 1  # Fallback
                    else:
                        hops_from_cloud = (
                            1  # Simplified: cloud vs fog approximation
                        )

            logger.info(
                f"[PLACEMENT_DETAIL] {i:2d}. Projection: {str(projection)[:50]}..."
            )
            logger.info(
                f"    Node: {node}, Strategy: {strategy}, Cost: {cost:.3f}"
            )
            logger.info(f"    Hops from cloud: {hops_from_cloud}")

    except Exception as e:
        logger.error(
            f"[PLACEMENT_ANALYSIS] Error logging detailed placement decisions: {e}"
        )


class OptimizedCSVWriter:
    """
    Thread-safe CSV writer with batched operations for better I/O performance.
    """

    def __init__(self, csv_file_path: str, batch_size: int = 20):
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.write_count = 0
        self.batch_buffer = []

        # Ensure result directory exists
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        # Write headers if file doesn't exist
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        """Ensure CSV file has proper headers."""
        if not os.path.exists(self.csv_file_path):
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
                "parent_factor",
                "workload_size",
                "query_length",
                "simulation_mode",
                "median_selectivity",
                "xi",
                "latency_threshold",
                # Metadata
                "total_projections_placed",
                "placements_at_cloud",
                "graph_density",
                "strategy_name",
                "strategy_status",
                # Computation times
                "combigen_time_seconds",
                "ines_placement_time_seconds",
                "ines_push_pull_time_seconds",
                "ines_total_time_seconds",
                "kraken_total_execution_time_seconds",
                "kraken_strategy_execution_time_seconds",
                # Placement costs
                "all_push_central_cost",
                "inev_cost",
                "ines_cost",
                "kraken_total_cost",
                "kraken_workload_cost",
                "kraken_average_cost_per_placement",
                # Latency
                "all_push_central_latency",
                "ines_latency",
                "kraken_max_latency",
            ]

            with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns_to_copy)

    def write_result(
        self,
        integrated_operator_placement_results: Dict[str, Any],
        ines_results: List[Any],
        config: SimulationConfig,
        graph_density: float,
        ines_object: Any,
    ) -> None:
        """Write a single result with batching for better performance."""
        with self.lock:
            # Prepare the row data
            row_data = self._prepare_row_data(
                integrated_operator_placement_results,
                ines_results,
                config,
                graph_density,
                ines_object,
            )

            self.batch_buffer.append(row_data)

            # Flush if buffer is full
            if len(self.batch_buffer) >= self.batch_size:
                self._flush_buffer()

    def flush(self) -> None:
        """Flush any remaining buffered data."""
        with self.lock:
            if self.batch_buffer:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Internal method to write buffered data to CSV."""
        if not self.batch_buffer:
            return

        with open(self.csv_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row_data in self.batch_buffer:
                writer.writerow(row_data)

        self.write_count += len(self.batch_buffer)
        logger.debug(
            f"[CSV_WRITER] Wrote batch of {len(self.batch_buffer)} results "
            f"(total: {self.write_count})"
        )
        self.batch_buffer.clear()

    def _prepare_row_data(
        self,
        integrated_operator_placement_results: Dict[str, Any],
        ines_results: List[Any],
        config: SimulationConfig,
        graph_density: float,
        ines_object: Any,
    ) -> List[Any]:
        """Prepare row data for CSV writing."""
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
            "parent_factor",
            "workload_size",
            "query_length",
            "simulation_mode",
            "median_selectivity",
            "xi",
            "latency_threshold",
            # Metadata
            "total_projections_placed",
            "placements_at_cloud",
            "graph_density",
            "strategy_name",
            "strategy_status",
            # Computation times
            "combigen_time_seconds",
            "ines_placement_time_seconds",
            "ines_push_pull_time_seconds",
            "ines_total_time_seconds",
            "kraken_total_execution_time_seconds",
            "kraken_strategy_execution_time_seconds",
            # Placement costs
            "all_push_central_cost",
            "inev_cost",
            "ines_cost",
            "kraken_total_cost",
            "kraken_workload_cost",
            "kraken_average_cost_per_placement",
            # Latency
            "all_push_central_latency",
            "ines_latency",
            "kraken_max_latency",
        ]

        # Extract all data from new Kraken structure
        row_data_dict = self._extract_row_data_dict(
            integrated_operator_placement_results,
            ines_results,
            config,
            graph_density,
        )

        # Return row values in correct order
        return [row_data_dict[col] for col in columns_to_copy]

    def _extract_row_data_dict(
        self,
        integrated_operator_placement_results,
        ines_results,
        config,
        graph_density,
    ) -> Dict[str, Any]:
        """Extract row data into dictionary format using new Kraken 2.0 structure."""
        # Debug logging to see what we're receiving
        logger.info(f"[CSV_EXTRACT] Keys in integrated_results: {list(integrated_operator_placement_results.keys())}")
        logger.info(f"[CSV_EXTRACT] Has best_solution: {'best_solution' in integrated_operator_placement_results}")
        if 'best_solution' in integrated_operator_placement_results:
            bs = integrated_operator_placement_results.get('best_solution')
            logger.info(f"[CSV_EXTRACT] best_solution type: {type(bs)}, value: {bs is not None}")
            if bs:
                logger.info(f"[CSV_EXTRACT] best_solution keys: {list(bs.keys()) if isinstance(bs, dict) else 'NOT A DICT'}")
                logger.info(f"[CSV_EXTRACT] best_solution metrics: {bs.get('metrics', {}) if isinstance(bs, dict) else 'N/A'}")

        # Extract IDs
        ines_simulation_id = ines_results[0]
        kraken_simulation_id = integrated_operator_placement_results.get("run_id", "")

        # Extract config parameters
        network_size = config.network_size
        event_skew = config.event_skew
        node_event_ratio = config.node_event_ratio
        num_event_types = config.num_event_types
        max_parents = config.max_parents
        parent_factor = config.parent_factor
        workload_size = config.query_size
        query_length = config.query_length
        simulation_mode = config.mode.value
        median_selectivity = _safe_float_convert(ines_results[11])
        xi = config.xi
        latency_threshold = config.latency_threshold

        # Extract from new Kraken 2.0 structure
        best_solution = integrated_operator_placement_results.get("best_solution")
        strategies = integrated_operator_placement_results.get("strategies", {})
        problem_info = integrated_operator_placement_results.get("problem_info", {})

        # Handle case where best_solution might be None
        if best_solution and isinstance(best_solution, dict):
            best_metrics = best_solution.get("metrics", {})
            strategy_name = best_solution.get("strategy_name", "greedy")
            strategy_data = strategies.get(strategy_name, {})
            strategy_status = strategy_data.get("status", "unknown")

            # Extract metadata
            total_projections_placed = best_metrics.get("num_placements", 0)
            placements_at_cloud = best_metrics.get("placements_at_cloud", 0)

            # Kraken costs and latency
            kraken_total_cost = best_metrics.get("total_cost", 0)
            kraken_workload_cost = best_metrics.get("workload_cost", 0)
            kraken_average_cost_per_placement = best_metrics.get("average_cost_per_placement", 0)
            kraken_max_latency = best_metrics.get("max_latency", 0)

            # Strategy execution time
            kraken_strategy_execution_time_seconds = strategy_data.get("execution_time_seconds", 0)
        else:
            # No valid solution from Kraken
            best_metrics = {}
            strategy_name = ""
            strategy_status = ""
            total_projections_placed = 0
            placements_at_cloud = 0
            kraken_total_cost = None
            kraken_workload_cost = None
            kraken_average_cost_per_placement = None
            kraken_max_latency = None
            kraken_strategy_execution_time_seconds = 0

        graph_density_value = graph_density

        # Computation times (convert numpy types to Python types)
        combigen_time_seconds = _safe_float_convert(ines_results[12])
        ines_placement_time_seconds = _safe_float_convert(ines_results[14])
        ines_push_pull_time_seconds = _safe_float_convert(ines_results[22])
        ines_total_time_seconds = ines_placement_time_seconds + ines_push_pull_time_seconds
        kraken_total_execution_time_seconds = integrated_operator_placement_results.get(
            "total_execution_time_seconds", 0
        )

        # Placement costs (convert numpy types to Python types)
        all_push_central_cost = _safe_float_convert(ines_results[2])
        inev_cost = _safe_float_convert(ines_results[3])
        ines_cost = _safe_float_convert(ines_results[21])

        # Latency (convert numpy types to Python types)
        all_push_central_latency = _safe_float_convert(ines_results[15])
        ines_latency = _safe_float_convert(ines_results[23])

        return {
            # IDs
            "ines_simulation_id": ines_simulation_id,
            "kraken_simulation_id": kraken_simulation_id,
            # Configuration parameters
            "network_size": network_size,
            "event_skew": event_skew,
            "node_event_ratio": node_event_ratio,
            "num_event_types": num_event_types,
            "max_parents": max_parents,
            "parent_factor": parent_factor,
            "workload_size": workload_size,
            "query_length": query_length,
            "simulation_mode": simulation_mode,
            "median_selectivity": median_selectivity,
            "xi": xi,
            "latency_threshold": latency_threshold,
            # Metadata
            "total_projections_placed": total_projections_placed,
            "placements_at_cloud": placements_at_cloud,
            "graph_density": graph_density_value,
            "strategy_name": strategy_name,
            "strategy_status": strategy_status,
            # Computation times
            "combigen_time_seconds": combigen_time_seconds,
            "ines_placement_time_seconds": ines_placement_time_seconds,
            "ines_push_pull_time_seconds": ines_push_pull_time_seconds,
            "ines_total_time_seconds": ines_total_time_seconds,
            "kraken_total_execution_time_seconds": kraken_total_execution_time_seconds,
            "kraken_strategy_execution_time_seconds": kraken_strategy_execution_time_seconds,
            # Placement costs
            "all_push_central_cost": all_push_central_cost,
            "inev_cost": inev_cost,
            "ines_cost": ines_cost,
            "kraken_total_cost": kraken_total_cost,
            "kraken_workload_cost": kraken_workload_cost,
            "kraken_average_cost_per_placement": kraken_average_cost_per_placement,
            # Latency
            "all_push_central_latency": all_push_central_latency,
            "ines_latency": ines_latency,
            "kraken_max_latency": kraken_max_latency,
        }


# Global CSV writer instance
_csv_writer = None
_csv_writer_lock = threading.Lock()


def get_csv_writer() -> OptimizedCSVWriter:
    """Get or create the global CSV writer instance."""
    global _csv_writer
    with _csv_writer_lock:
        if _csv_writer is None:
            # Use absolute path based on script location to ensure it works
            # regardless of where the script is run from
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result_dir = os.path.join(script_dir, "kraken2_0", "result")
            csv_file_path = os.path.join(result_dir, "run_results.csv")
            _csv_writer = OptimizedCSVWriter(csv_file_path, batch_size=20)
    return _csv_writer


def write_final_results(
    integrated_operator_placement_results: Dict[str, Any],
    ines_results: List[Any],
    config: SimulationConfig,
    graph_density: float,
    ines_object: Any,
) -> None:
    """
    Write final combined simulation results to CSV file using optimized writer.
    Moved from operator_placement_legacy_hook.py for consolidated I/O.
    """
    writer = get_csv_writer()
    writer.write_result(
        integrated_operator_placement_results,
        ines_results,
        config,
        graph_density,
        ines_object,
    )

    # Handle detailed logging for latency analysis
    best_solution = integrated_operator_placement_results.get("best_solution", {})
    best_metrics = best_solution.get("metrics", {})
    kraken_latency = best_metrics.get("max_latency", 0)
    ines_latency = _safe_float_convert(ines_results[23])

    # Conditional detailed logging when Kraken latency is at least 1.5x INES latency
    if kraken_latency >= 1.5 * ines_latency:
        central_latency = _safe_float_convert(ines_results[15])
        logger.info(
            f"[LATENCY_ANALYSIS] Kraken latency ({kraken_latency:.3f}) >= 1.5x INES latency ({ines_latency:.3f})"
        )
        logger.info(
            f"[LATENCY_ANALYSIS] INES vs Kraken comparison for simulation {ines_results[0]}:"
        )
        logger.info(f"  - INES latency: {ines_latency:.3f}")
        logger.info(f"  - Kraken latency: {kraken_latency:.3f}")
        logger.info(f"  - Central latency: {central_latency:.3f}")

        # Log detailed placement decisions for both INES and Kraken
        _log_detailed_placement_decisions(
            integrated_operator_placement_results, config, kraken_latency, ines_latency
        )


def flush_csv_writer() -> None:
    """Flush any remaining data in the CSV writer."""
    writer = get_csv_writer()
    writer.flush()


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
            "parameter_set_id": self.parameter_set_id,
        }


class OptimizedSimulationManager:
    """
    Manages optimized parallel simulation execution with hybrid batching approach.

    Key optimizations:
    - Batches jobs to reduce process overhead
    - Uses optimal worker count based on CPU cores
    - Buffers I/O operations for better throughput
    - Persistent worker processes with job batching
    """

    def __init__(
        self,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.enable_parallel = enable_parallel

        # Optimize worker count: use fewer processes to reduce overhead
        if max_workers is None:
            if enable_parallel:
                # Use half the CPU cores to reduce process overhead
                self.max_workers = max(1, multiprocessing.cpu_count() // 2)
            else:
                self.max_workers = 1
        else:
            self.max_workers = max_workers

        # Auto-calculate optimal batch size if not provided
        if batch_size is None:
            if enable_parallel and self.max_workers > 1:
                # Larger batches for parallel processing to amortize overhead
                self.batch_size = 4
            else:
                # No batching for sequential processing
                self.batch_size = 1
        else:
            self.batch_size = batch_size

        # Force single worker for sequential mode
        if not enable_parallel:
            self.max_workers = 1
            self.batch_size = 1

        self.successful_jobs = 0
        self.failed_jobs = 0
        self.start_time = None

        # Results buffer for batched I/O
        self.result_buffer = []
        self.io_batch_size = 10  # Write results in batches of 10

    def run_simulations(self, jobs: List[SimulationJob]) -> None:
        """
        Execute all simulation jobs using optimized batching strategy.

        Args:
            jobs: List of SimulationJob objects to execute
        """
        total_jobs = len(jobs)
        logger.info(
            f"[SIMULATION] Starting {total_jobs} jobs with {self.max_workers} workers"
        )
        logger.info(
            f"[SIMULATION] Parallel: {'Enabled' if self.enable_parallel else 'Disabled'}, "
            f"Batch size: {self.batch_size}, I/O batch size: {self.io_batch_size}"
        )

        self.start_time = time.time()

        if self.enable_parallel and self.max_workers > 1:
            self._run_parallel_batched(jobs)
        else:
            self._run_sequential(jobs)

        # Flush any remaining buffered results
        self._flush_result_buffer()

        # Ensure CSV writer flushes any remaining data
        flush_csv_writer()

        # Print final statistics
        total_time = time.time() - self.start_time
        avg_time_per_job = total_time / total_jobs if total_jobs > 0 else 0

        logger.info(
            f"\n[RESULTS] All {total_jobs} jobs completed in {total_time:.1f}s "
            f"(avg: {avg_time_per_job:.1f}s per job)"
        )
        logger.info(
            f"[RESULTS] Successful: {self.successful_jobs}, Failed: {self.failed_jobs}"
        )

        if self.failed_jobs > 0:
            logger.warning(f"[RESULTS] Warning: {self.failed_jobs} job(s) failed")

    def _run_parallel_batched(self, jobs: List[SimulationJob]) -> None:
        """
        Execute jobs in parallel using batching to reduce process overhead.
        """
        # Ensure main process also has correct path for unpickling worker results
        import sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.info(f"[MAIN_PROCESS] Added {parent_dir} to sys.path")

        # Create job batches to amortize process creation overhead
        total_jobs = len(jobs)
        job_batches = []

        for i in range(0, total_jobs, self.batch_size):
            batch = jobs[i : i + self.batch_size]
            job_batches.append([job.to_dict() for job in batch])

        logger.info(
            f"[BATCHING] Created {len(job_batches)} batches of {self.batch_size} jobs each"
        )

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_setup_worker_path,
            initargs=(parent_dir,)
        ) as executor:
            # Submit batches instead of individual jobs
            future_to_batch = {
                executor.submit(process_job_batch, batch): i
                for i, batch in enumerate(job_batches)
            }

            # Process batch results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()

                    # Process each result in the batch
                    for result in batch_results:
                        self._process_completed_job(result)

                    # Progress reporting
                    completed_jobs = self.successful_jobs + self.failed_jobs
                    if completed_jobs % 50 == 0 or batch_idx % 5 == 0:
                        elapsed = time.time() - self.start_time
                        logger.info(
                            f"[PROGRESS] {completed_jobs}/{total_jobs} jobs completed "
                            f"- {elapsed:.1f}s elapsed"
                        )

                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    logger.error(f"[ERROR] Batch {batch_idx} execution failed: {str(e)}")
                    logger.error(f"[ERROR] Traceback:\n{error_trace}")
                    # Mark all jobs in the failed batch as failed
                    batch_size_actual = len(job_batches[batch_idx])
                    self.failed_jobs += batch_size_actual

    def _run_sequential(self, jobs: List[SimulationJob]) -> None:
        """
        Execute jobs sequentially for debugging.
        """
        for i, job in enumerate(jobs):
            try:
                logger.info(
                    f"[SEQUENTIAL] Processing job {i + 1}/{len(jobs)} ({job.parameter_set_id})"
                )
                result = run_simulation_worker(job.to_dict())
                self._process_completed_job(result)

            except Exception as e:
                logger.error(f"[ERROR] Job {job.job_id} failed: {str(e)}")
                self.failed_jobs += 1

    def _process_completed_job(self, result: Dict[str, Any]) -> None:
        """
        Process a completed job result with buffered I/O for better performance.
        """
        if result["success"]:
            # Add to I/O buffer for batched writing
            self.result_buffer.append(result)

            # Flush buffer when it reaches the batch size
            if len(self.result_buffer) >= self.io_batch_size:
                self._flush_result_buffer()

            self.successful_jobs += 1
        else:
            logger.error(
                f"[ERROR] Job {result['job_id']} ({result['parameter_set_id']}) "
                f"failed: {result['error_msg']}"
            )
            self.failed_jobs += 1

    def _flush_result_buffer(self) -> None:
        """
        Write buffered results to files in batch for better I/O performance.
        """
        if self.result_buffer:
            logger.debug(f"[I/O] Writing batch of {len(self.result_buffer)} results")

            for result in self.result_buffer:
                write_final_results(
                    result["integrated_results"],
                    result["ines_results"],
                    result["config"],
                    result["graph_density"],
                    result["ines_object"],
                )

            self.result_buffer.clear()


def process_job_batch(job_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process multiple jobs in sequence within a single worker process to amortize overhead.

    This is the key optimization: instead of creating a new process for each job,
    we process multiple jobs in the same process to amortize the expensive process
    creation and INES initialization overhead.

    Args:
        job_batch: List of serialized job dictionaries

    Returns:
        List of job results with all data needed for write_final_results
    """
    global _worker_state

    # Initialize worker on first use
    _initialize_worker()

    results = []
    batch_start_time = time.time()

    # Track job processing in this worker
    with _worker_lock:
        _worker_state["job_count"] += len(job_batch)
        current_job_count = _worker_state["job_count"]

    logger.info(
        f"[BATCH_WORKER] Processing batch of {len(job_batch)} jobs "
        f"(total processed by this worker: {current_job_count})"
    )

    # Use cached modules from worker state
    INES_class = _worker_state["INES"]
    calculate_graph_density = _worker_state["calculate_graph_density"]

    for job_data in job_batch:
        try:
            job_id = job_data["job_id"]
            config = job_data["config"]
            parameter_set_id = job_data["parameter_set_id"]

            # Execute INES simulation using cached class
            simulation = INES_class(config)

            # Calculate graph density using cached function
            graph_density = calculate_graph_density(simulation.graph)

            # Store successful result
            results.append(
                {
                    "job_id": job_id,
                    "ines_results": simulation.results,
                    "integrated_results": simulation.kraken_results,
                    "config": config,
                    "graph_density": graph_density,
                    "ines_object": simulation,
                    "success": True,
                    "error_msg": None,
                    "parameter_set_id": parameter_set_id,
                }
            )

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            error_message = (
                f"Exception in job {job_data.get('job_id', 'unknown')}: {str(e)}\n{error_traceback}"
            )
            logger.error(f"[BATCH_WORKER_ERROR] {error_message}")

            results.append(
                {
                    "job_id": job_data.get("job_id", -1),
                    "ines_results": None,
                    "integrated_results": None,
                    "config": job_data.get("config"),
                    "graph_density": None,
                    "success": False,
                    "error_msg": error_message,
                    "parameter_set_id": job_data.get("parameter_set_id", "unknown"),
                }
            )

    batch_time = time.time() - batch_start_time
    successful_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - successful_count

    logger.info(
        f"[BATCH_WORKER] Completed batch in {batch_time:.1f}s: "
        f"{successful_count} successful, {failed_count} failed"
    )

    return results


def run_simulation_worker(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for parallel simulation execution.

    Args:
        job_data: Dictionary containing serialized job data

    Returns:
        Dictionary with job results including all data needed for write_final_results
    """
    # Ensure sys.path includes the parent directory for 'src' imports
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        job_id = job_data["job_id"]
        config = job_data["config"]
        parameter_set_id = job_data["parameter_set_id"]

        logger.info(
            f"[WORKER] Starting simulation job {job_id + 1} ({parameter_set_id})"
        )

        # Execute INES simulation
        simulation = INES(config)

        # Calculate graph density
        from INES import calculate_graph_density

        graph_density = calculate_graph_density(simulation.graph)

        # Return complete result data
        return {
            "job_id": job_id,
            "ines_results": simulation.results,
            "integrated_results": simulation.kraken_results,
            "config": config,
            "graph_density": graph_density,
            "ines_object": simulation,
            "success": True,
            "error_msg": None,
            "parameter_set_id": parameter_set_id,
        }

    except Exception as e:
        error_message = (
            f"Exception in job {job_data.get('job_id', 'unknown')}: {str(e)}"
        )
        logger.error(f"[WORKER ERROR] {error_message}")
        return {
            "job_id": job_data.get("job_id", -1),
            "ines_results": None,
            "integrated_results": None,
            "config": job_data.get("config"),
            "graph_density": None,
            "success": False,
            "error_msg": error_message,
            "parameter_set_id": job_data.get("parameter_set_id", "unknown"),
        }


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
    xi: float = 0.0,
    latency_threshold: float = None,
    max_workers: Optional[int] = None,
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
        xi: Xi value for processing latency weight (default 0.0)
        latency_threshold: Threshold for latency calculation
        max_workers: Maximum number of parallel workers (auto-detected if None)
    """
    # Calculate parent_factor from max_parents for consistency with parameter study
    import math

    parent_factor = (
        max_parents / math.ceil(math.log2(network_size)) if network_size > 1 else 1.0
    )

    logger.info(
        f"[SINGLE] Starting single simulation configuration with {num_runs} runs"
    )
    logger.info(
        f"[CONFIG] Network size: {network_size}, Workload: {workload_size}, Query length: {query_length}"
    )
    logger.info(
        f"[CONFIG] Mode: {mode.value}, Max parents: {max_parents}, Parent factor: {parent_factor:.2f}"
    )

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
        parent_factor=parent_factor,
        query_size=workload_size,
        query_length=query_length,
        mode=mode,
        xi=xi,
        latency_threshold=latency_threshold,
    )

    # Generate jobs for the same configuration
    parameter_set_id = (
        f"single_n{network_size}_w{workload_size}_q{query_length}_p{max_parents}"
    )
    jobs = []

    for run_id in range(num_runs):
        job = SimulationJob(
            job_id=run_id,
            config=config,
            parameter_set_id=f"{parameter_set_id}_run{run_id + 1}",
        )
        jobs.append(job)

    # Execute with optimized manager
    manager = OptimizedSimulationManager(
        enable_parallel=enable_parallel,
        max_workers=max_workers,
        batch_size=2,  # Small batches for single simulations
    )

    manager.run_simulations(jobs)
    logger.info(f"[SINGLE] Completed {len(jobs)} runs for single configuration")


def run_parameter_study(
    network_sizes: List[int] = [12],
    workload_sizes: List[int] = [5],
    parent_factors: List[float] = [1.8],
    query_lengths: List[int] = [5],
    runs_per_combination: int = 5,
    node_event_ratio: float = 0.5,
    num_event_types: int = 6,
    event_skew: float = 2.0,
    xi: float = 0,
    latency_threshold: float = None,
    mode: SimulationMode = SimulationMode.FULLY_DETERMINISTIC,
    enable_parallel: bool = False,
    max_workers: int = 1,
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
        xi: Xi value for processing latency weight (default 0.0)
        latency_threshold: Optional latency threshold to make kraken latency aware
        mode: Simulation mode
        enable_parallel: Whether to enable parallel processing
        max_workers: Maximum number of parallel workers
    """
    logger.info("[PARAMETER_STUDY] Starting full parameter study")
    logger.info(f"[PARAMETER_STUDY] Network sizes: {network_sizes}")
    logger.info(f"[PARAMETER_STUDY] Workload sizes: {workload_sizes}")
    logger.info(f"[PARAMETER_STUDY] Parent factors: {parent_factors}")
    logger.info(f"[PARAMETER_STUDY] Query lengths: {query_lengths}")
    logger.info(f"[PARAMETER_STUDY] Runs per combination: {runs_per_combination}")

    # Generate ALL jobs upfront
    jobs = []
    job_id = 0

    for parent_factor in parent_factors:
        for workload_size in workload_sizes:
            for network_size in network_sizes:
                for query_length in query_lengths:
                    # Calculate max_parents using same formula as original
                    max_parents = int(
                        parent_factor * math.ceil(math.log2(network_size))
                    )

                    # Create parameter set identifier
                    parameter_set_id = f"n{network_size}_w{workload_size}_q{query_length}_pf{parent_factor}_p{max_parents}"

                    # Create configuration for this parameter combination
                    config = SimulationConfig(
                        network_size=network_size,
                        node_event_ratio=node_event_ratio,
                        num_event_types=num_event_types,
                        event_skew=event_skew,
                        max_parents=max_parents,
                        parent_factor=parent_factor,
                        query_size=workload_size,
                        query_length=query_length,
                        xi=xi,
                        mode=mode,
                        latency_threshold=latency_threshold,
                    )

                    # Generate multiple runs for this combination
                    for run_num in range(runs_per_combination):
                        job = SimulationJob(
                            job_id=job_id,
                            config=config,
                            parameter_set_id=f"{parameter_set_id}_run{run_num + 1}",
                        )
                        jobs.append(job)
                        job_id += 1

    total_jobs = len(jobs)
    expected_jobs = (
        len(network_sizes)
        * len(workload_sizes)
        * len(parent_factors)
        * len(query_lengths)
        * runs_per_combination
    )

    logger.info(
        f"[PARAMETER_STUDY] Generated {total_jobs} jobs (expected: {expected_jobs})"
    )
    assert total_jobs == expected_jobs, (
        f"Job count mismatch: generated {total_jobs}, expected {expected_jobs}"
    )

    # Execute all jobs with single persistent process pool
    manager = OptimizedSimulationManager(
        enable_parallel=enable_parallel,
        max_workers=max_workers,
        batch_size=4,  # Optimized batching for parameter studies
    )

    manager.run_simulations(jobs)
    logger.info(
        f"[PARAMETER_STUDY] Completed all {total_jobs} jobs across all parameter combinations"
    )


def main() -> None:
    """
    Main entry point with execution options.

    Choose between single simulation debugging or full parameter study.
    """

    # # Option 1: Single simulation debugging (commented by default)
    # # Uncomment for debugging single configurations
    # run_single_simulation(
    #     network_size=30,
    #     mode=SimulationMode.FULLY_DETERMINISTIC,
    #     enable_parallel=False,
    #     num_runs=1
    # )

    # Option 2: Full parameter study (active by default)
    run_parameter_study(
        network_sizes=[10, 30, 50, 100],
        workload_sizes=[3, 5, 8],
        parent_factors=[1.8],
        query_lengths=[3, 5, 8],
        runs_per_combination=4,
        node_event_ratio=0.5,
        num_event_types=6,
        event_skew=2.0,
        mode=SimulationMode.RANDOM,
        enable_parallel=True,
        max_workers=14,
        xi=0,
    )


if __name__ == "__main__":
    main()
