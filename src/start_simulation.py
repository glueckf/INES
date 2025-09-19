import math
import csv
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import logging

from INES import INES, SimulationConfig, SimulationMode

logger = logging.getLogger(__name__)


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
        placements = integrated_operator_placement_results["formatted_results"][
            "placements"
        ]

        logger.info(
            f"[PLACEMENT_ANALYSIS] Detailed placement decisions (Kraken latency {kraken_latency:.3f} >= 1.5x INES {ines_latency:.3f}):"
        )
        logger.info(
            f"[PLACEMENT_ANALYSIS] Network size: {config.network_size}, Workload size: {config.query_size}"
        )

        # Log summary statistics
        total_placements = len(placements)
        cloud_placements = sum(
            1 for p in placements.values() if p.get("placement_node") == 0
        )
        fog_placements = total_placements - cloud_placements

        logger.info(f"[PLACEMENT_ANALYSIS] Total placements: {total_placements}")
        logger.info(
            f"[PLACEMENT_ANALYSIS] Cloud placements (node 0): {cloud_placements}"
        )
        logger.info(f"[PLACEMENT_ANALYSIS] Fog placements: {fog_placements}")

        # Log each placement decision in detail
        for i, (projection, details) in enumerate(placements.items(), 1):
            if details.get("success", False):
                node = details.get("placement_node", "unknown")
                cost = details.get("total_cost", 0)
                strategy = details.get("strategy", "unknown")
                latency = details.get("plan_details", {}).get("latency", 0)

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
                    f"[PLACEMENT_DETAIL] {i:2d}. Projection: {projection[:50]}..."
                )
                logger.info(
                    f"    Node: {node}, Strategy: {strategy}, Cost: {cost:.3f}, Latency: {latency:.3f}"
                )
                logger.info(f"    Hops from cloud: {hops_from_cloud}")

                # Log additional details if available
                plan_details = details.get("plan_details", {})
                if "transmission_ratio" in plan_details:
                    logger.info(
                        f"    Transmission ratio: {plan_details['transmission_ratio']:.3f}"
                    )
                if "computing_time" in plan_details:
                    logger.info(
                        f"    Computing time: {plan_details['computing_time']:.3f}"
                    )
            else:
                logger.warning(
                    f"[PLACEMENT_DETAIL] {i:2d}. Failed placement for: {projection[:50]}..."
                )

    except Exception as e:
        logger.error(
            f"[PLACEMENT_ANALYSIS] Error logging detailed placement decisions: {e}"
        )


def write_final_results(
    integrated_operator_placement_results: Dict[str, Any],
    ines_results: List[Any],
    config: SimulationConfig,
    graph_density: float,
    ines_object: Any,
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
        "parent_factor",
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
        "kraken_execution_time_placement",
        "kraken_execution_time_push_pull",
        # Algorithm metrics
        "prepp_call_count",
        "placement_evaluations_count",
        "prepp_cache_hits",
        "prepp_cache_misses",
        "prepp_cache_hit_rate",
        # Network topology metrics
        "network_clustering_coefficient",
        "network_centralization",
        "avg_node_degree",
        # Query complexity metrics
        "query_complexity_score",
        "highest_query_output_rate",
        "projection_dependency_length",
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

    kraken_metadata = integrated_operator_placement_results["formatted_results"][
        "metadata"
    ]
    kraken_summary = integrated_operator_placement_results["formatted_results"][
        "summary"
    ]

    # Extract IDs
    ines_simulation_id = ines_results[0]
    kraken_simulation_id = integrated_operator_placement_results["kraken_simulation_id"]

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
    median_selectivity = ines_results[11]

    # Extract metadata
    total_projections_placed = kraken_summary.get("successful_placements", 0)
    placement_difference_to_ines_count = kraken_summary.get(
        "placement_difference_count", 0
    )
    placements_at_cloud = kraken_metadata.get("placements_at_cloud", 0)
    graph_density_value = graph_density

    # Computation times
    combigen_time_seconds = ines_results[12]
    ines_placement_time_seconds = ines_results[14]
    ines_push_pull_time_seconds = ines_results[22]
    ines_total_time_seconds = float(ines_placement_time_seconds) + float(
        ines_push_pull_time_seconds
    )
    kraken_execution_time_seconds = kraken_metadata.get(
        "kraken_execution_time_seconds", 0
    )
    kraken_execution_time_placement = kraken_metadata.get(
        "kraken_execution_time_placement", 0
    )
    kraken_execution_time_push_pull = kraken_metadata.get(
        "kraken_execution_time_push_pull", 0
    )

    # Algorithm metrics
    prepp_call_count = kraken_metadata.get("prepp_call_count", 0)
    placement_evaluations_count = kraken_metadata.get("placement_evaluations_count", 0)
    prepp_cache_hits = kraken_metadata.get("prepp_cache_hits", 0)
    prepp_cache_misses = kraken_metadata.get("prepp_cache_misses", 0)
    prepp_cache_hit_rate = kraken_metadata.get("prepp_cache_hit_rate", 0.0)

    # Network topology metrics
    network_clustering_coefficient = kraken_metadata.get(
        "network_clustering_coefficient", 0.0
    )
    network_centralization = kraken_metadata.get("network_centralization", 0.0)
    avg_node_degree = kraken_metadata.get("avg_node_degree", 0.0)

    # Query complexity metrics
    query_complexity_score = kraken_metadata.get("query_complexity_score", 0.0)
    highest_query_output_rate = kraken_metadata.get("highest_query_output_rate")
    projection_dependency_length = kraken_metadata.get(
        "projection_dependency_length", 0
    )

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
        "parent_factor": parent_factor,
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
        "kraken_execution_time_placement": kraken_execution_time_placement,
        "kraken_execution_time_push_pull": kraken_execution_time_push_pull,
        # Algorithm metrics
        "prepp_call_count": prepp_call_count,
        "placement_evaluations_count": placement_evaluations_count,
        "prepp_cache_hits": prepp_cache_hits,
        "prepp_cache_misses": prepp_cache_misses,
        "prepp_cache_hit_rate": prepp_cache_hit_rate,
        # Network topology metrics
        "network_clustering_coefficient": network_clustering_coefficient,
        "network_centralization": network_centralization,
        "avg_node_degree": avg_node_degree,
        # Query complexity metrics
        "query_complexity_score": query_complexity_score,
        "highest_query_output_rate": highest_query_output_rate,
        "projection_dependency_length": projection_dependency_length,
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

    # Conditional detailed logging when Kraken latency is at least 1.5x INES latency
    if kraken_latency >= 1.5 * ines_latency:
        logger.info(
            f"[LATENCY_ANALYSIS] Kraken latency ({kraken_latency:.3f}) >= 1.5x INES latency ({ines_latency:.3f})"
        )
        logger.info(
            f"[LATENCY_ANALYSIS] INES vs Kraken comparison for simulation {ines_simulation_id}:"
        )
        logger.info(f"  - INES latency: {ines_latency:.3f}")
        logger.info(f"  - Kraken latency: {kraken_latency:.3f}")
        logger.info(f"  - Central latency: {all_push_central_latency:.3f}")

        # Log detailed placement decisions for both INES and Kraken
        _log_detailed_placement_decisions(
            integrated_operator_placement_results, config, kraken_latency, ines_latency
        )

    logger.info(f"[I/O] Appended combined simulation results to {csv_file_path}")
    logger.debug(
        f"[I/O] INES ID: {ines_simulation_id}, Kraken ID: {kraken_simulation_id}"
    )


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
    Manages optimized parallel simulation execution with persistent process pool.
    """

    def __init__(
        self,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
        batch_size: int = 1,
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
        logger.info(
            f"[SIMULATION] Starting {total_jobs} jobs with {self.max_workers} workers..."
        )
        logger.info(
            f"[SIMULATION] Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}"
        )

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

        logger.info(
            f"\n[RESULTS] All {total_jobs} jobs completed in {total_time:.1f}s (avg: {avg_time_per_job:.1f}s per job)"
        )
        logger.info(
            f"[RESULTS] Successful: {self.successful_jobs}, Failed: {self.failed_jobs}"
        )

        if self.failed_jobs > 0:
            logger.warning(f"[RESULTS] Warning: {self.failed_jobs} job(s) failed")

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
                        logger.info(
                            f"[PROGRESS] {completed_jobs}/{len(jobs)} jobs completed - {elapsed:.1f}s elapsed"
                        )

                except Exception as e:
                    logger.error(f"[ERROR] Job {job_id} execution failed: {str(e)}")
                    self.failed_jobs += 1

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
                    result["graph_density"],
                    result["ines_object"],
                )

            self.successful_jobs += 1
        else:
            logger.error(
                f"[ERROR] Job {result['job_id']} ({result['parameter_set_id']}) failed: {result['error_msg']}"
            )
            self.failed_jobs += 1

    def _flush_results(self) -> None:
        """
        Write any batched results to files.
        """
        if self.result_batch:
            logger.info(f"[I/O] Writing batch of {len(self.result_batch)} results...")
            for result in self.result_batch:
                write_final_results(
                    result["integrated_results"],
                    result["ines_results"],
                    result["config"],
                    result["graph_density"],
                    result["ines_object"],
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
            "integrated_results": simulation.integrated_operator_placement_results,
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
        batch_size=1,  # Immediate writing for single simulations
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
                        mode=mode,
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
        batch_size=10,  # Small batching for better progress tracking
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
        runs_per_combination=30,
        node_event_ratio=0.5,
        num_event_types=6,
        event_skew=2.0,
        mode=SimulationMode.RANDOM,
        enable_parallel=False,
        max_workers=14,
    )


if __name__ == "__main__":
    main()
