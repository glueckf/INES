from INES import INES
import logging
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import sys
import multiprocessing

# Configure comprehensive logging for Docker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Docker will capture this
        logging.FileHandler("simulation.log")  # Also keep file logging
    ]
)

logger = logging.getLogger(__name__)


def run_simulation(nodes, node_event_ratio, num_eventtypes, eventskew, max_parents, query_size, query_length, run,
                   deterministic=False):
    """Execute a single INES simulation with comprehensive logging."""
    process_logger = logging.getLogger(f"simulation_run_{run}")
    start_time = time.time()

    # Log simulation parameters
    process_logger.info(f"Starting simulation run {run}")
    mode = "DETERMINISTIC" if deterministic else "RANDOM"
    process_logger.info(f"Mode: {mode}")
    process_logger.info(f"Parameters: nodes={nodes}, event_ratio={node_event_ratio}, "
                        f"event_types={num_eventtypes}, skew={eventskew}, "
                        f"max_parents={max_parents}, query_size={query_size}, query_length={query_length}")

    try:
        process_logger.info(f"Initializing INES system for run {run}")
        simulation_start = time.time()

        simulation = INES(nodes, node_event_ratio, num_eventtypes, eventskew, max_parents,
                          query_size, query_length, use_deterministic_scenario=deterministic)

        simulation_time = time.time() - simulation_start
        process_logger.info(f"INES system initialized successfully in {simulation_time:.2f}s")

        # Log key results
        if hasattr(simulation, 'results') and len(simulation.results) > 3:
            transmission_ratio = simulation.results[1] if len(simulation.results) > 1 else 'N/A'
            total_cost = simulation.results[2] if len(simulation.results) > 2 else 'N/A'
            optimized_cost = simulation.results[3] if len(simulation.results) > 3 else 'N/A'

            process_logger.info(f"Run {run} results: transmission_ratio={transmission_ratio}, "
                                f"total_cost={total_cost}, optimized_cost={optimized_cost}")

            if isinstance(transmission_ratio, (int, float)) and transmission_ratio < 1:
                savings = (1 - transmission_ratio) * 100
                process_logger.info(f"Run {run} achieved {savings:.1f}% transmission savings")

        total_time = time.time() - start_time
        process_logger.info(f"Simulation run {run} completed successfully in {total_time:.2f}s")

        return pd.DataFrame([simulation.results], columns=simulation.schema)

    except Exception as e:
        total_time = time.time() - start_time
        process_logger.error(f"Simulation run {run} failed after {total_time:.2f}s: {str(e)}", exc_info=True)
        return None


def start_simulation(nodes, node_event_ratio, num_eventtypes, eventskew, max_parents, query_size, query_length, runs):
    """Execute multiple INES simulations in parallel with comprehensive logging."""
    batch_start_time = time.time()
    file_name = f"INES-simulation_{datetime.now().strftime('%d%m%Y%H%M%S')}.csv"

    logger.info("=" * 80)
    logger.info("STARTING INES SIMULATION BATCH")
    logger.info("=" * 80)
    logger.info(f"Batch configuration: {runs} runs with {4} max workers")
    logger.info(f"Network parameters: {nodes} nodes, {node_event_ratio} event ratio, {num_eventtypes} event types")
    logger.info(
        f"Simulation parameters: skew={eventskew}, max_parents={max_parents}, query_size={query_size}, query_length={query_length}")
    logger.info(f"Output file: {file_name}")

    all_results = []
    successful_runs = 0
    failed_runs = 0

    logger.info("Launching parallel simulation processes...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_simulation, nodes, node_event_ratio, num_eventtypes, eventskew, max_parents, query_size,
                            query_length, run)
            for run in range(runs)
        ]

        for i, future in enumerate(futures):
            logger.info(f"Processing result for simulation run {i}/{runs}")

            try:
                result = future.result(timeout=300)
                if result is not None:
                    all_results.append(result)
                    successful_runs += 1
                    logger.info(f"Run {i} completed successfully ({successful_runs}/{runs} completed)")
                else:
                    failed_runs += 1
                    logger.warning(f"Run {i} returned no results ({failed_runs} failures so far)")
            except Exception as e:
                failed_runs += 1
                logger.error(f"Error in run {i}: {e} ({failed_runs} failures so far)")
            except TimeoutError:
                failed_runs += 1
                logger.error(f"Run {i} timed out after 300 seconds ({failed_runs} failures so far)")

    batch_time = time.time() - batch_start_time

    # Save results and generate summary
    if all_results:
        all_results = [result for result in all_results if result is not None]
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(f"./res/{file_name}", index=False)

        # Calculate batch statistics
        if len(final_df) > 0 and 'TransmissionRatio' in final_df.columns:
            avg_transmission_ratio = final_df['TransmissionRatio'].mean()
            min_transmission_ratio = final_df['TransmissionRatio'].min()
            max_transmission_ratio = final_df['TransmissionRatio'].max()

            logger.info("=" * 80)
            logger.info("SIMULATION BATCH COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total execution time: {batch_time:.2f}s")
            logger.info(f"Successful runs: {successful_runs}/{runs} ({(successful_runs / runs) * 100:.1f}%)")
            logger.info(f"Failed runs: {failed_runs}/{runs}")
            logger.info(f"Results saved to: ./res/{file_name}")
            logger.info(f"Performance summary:")
            logger.info(f"  Average transmission ratio: {avg_transmission_ratio:.3f}")
            logger.info(f"  Best transmission ratio: {min_transmission_ratio:.3f}")
            logger.info(f"  Worst transmission ratio: {max_transmission_ratio:.3f}")
            if avg_transmission_ratio < 1.0:
                avg_savings = (1 - avg_transmission_ratio) * 100
                logger.info(f"  Average transmission savings: {avg_savings:.1f}%")
            logger.info("=" * 80)
        else:
            logger.info(f"Results saved to: ./res/{file_name} ({len(all_results)} successful runs)")
    else:
        logger.error("No successful simulation runs - no output file generated")
        logger.error(f"Batch failed: {failed_runs}/{runs} runs failed")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    logger.info("INES Simulation System Starting")
    logger.info(f"Python multiprocessing method: spawn (macOS compatible)")

    # Configuration - easily adjustable simulation parameters
    NODES = 12
    NODE_EVENT_RATIO = 0.5
    NUM_EVENTTYPES = 6
    EVENTSKEW = 0.3
    MAX_PARENTS = 10
    QUERY_SIZE = 3
    QUERY_LENGTH = 5
    NUM_RUNS = 1 # How many runs do we want?

    # Set to True to reproduce exact results from the logged simulation
    DETERMINISTIC_MODE = True

    mode_desc = "DETERMINISTIC (reproducible results)" if DETERMINISTIC_MODE else "RANDOM (varied results)"
    logger.info(f"Simulation mode: {mode_desc}")

    if DETERMINISTIC_MODE:
        logger.info("Using hardcoded scenario that reproduces exact logged results:")
        logger.info("  - Expected transmission ratio: 0.314")
        logger.info("  - Expected transmission savings: 68.6%")
        logger.info("  - Expected total cost: 51000.0")
        logger.info("  - Expected optimized cost: 16004.0")
    else:
        logger.info("Simulation configuration:")
        logger.info(f"  Network: {NODES} nodes, {NODE_EVENT_RATIO} event ratio")
        logger.info(f"  Events: {NUM_EVENTTYPES} types, {EVENTSKEW} skew")
        logger.info(f"  Queries: {QUERY_SIZE} size, {QUERY_LENGTH} length")
        logger.info(f"  Execution: {NUM_RUNS} runs")

    # Run single simulation for testing
    all_results = []

    for i in range(NUM_RUNS):
        logger.info(f"Executing simulation {i + 1}/{NUM_RUNS}")
        result = run_simulation(NODES, NODE_EVENT_RATIO, NUM_EVENTTYPES, EVENTSKEW, MAX_PARENTS,
                                QUERY_SIZE, QUERY_LENGTH, i, deterministic=DETERMINISTIC_MODE)
        all_results.append(result)

    # Process results
    successful_results = [result for result in all_results if result is not None]

    if successful_results:
        file_name = f"INES-simulation_{datetime.now().strftime('%d%m%Y%H%M%S')}.csv"
        final_df = pd.concat(successful_results, ignore_index=True)
        final_df.to_csv(f"./res/{file_name}", index=False)

        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: ./res/{file_name}")
        logger.info(f"Successful runs: {len(successful_results)}/{NUM_RUNS}")

        # Log key performance metrics if available
        if 'TransmissionRatio' in final_df.columns:
            avg_ratio = final_df['TransmissionRatio'].mean()
            logger.info(f"Average transmission ratio: {avg_ratio:.3f}")
            if avg_ratio < 1.0:
                savings = (1 - avg_ratio) * 100
                logger.info(f"Average transmission savings: {savings:.1f}%")

            # Validate results if in deterministic mode
            if DETERMINISTIC_MODE:
                from thesis_contribution_fg.hardcoded_scenario import get_expected_results

                expected = get_expected_results()

                logger.info("DETERMINISTIC MODE VALIDATION:")
                ratio_match = abs(avg_ratio - expected['transmission_ratio']) < 0.001
                logger.info(f"  Transmission ratio match: {'✓' if ratio_match else '✗'} "
                            f"(expected: {expected['transmission_ratio']:.3f}, got: {avg_ratio:.3f})")

                if ratio_match:
                    logger.info("  Results successfully reproduced - deterministic scenario working correctly!")
                else:
                    logger.warning("  Results don't match expected values - check hardcoded scenario")

        logger.info("=" * 60)
    else:
        logger.error("All simulation runs failed - no results generated")
        logger.error("Check the error logs above for debugging information")
