from Benchmark import INES
import traceback
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
import sys

# logging.basicConfig(
#     filename="simulation_errors.log",  # Save errors in a .log file
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )


def run_simulation(
    nodes,
    node_event_ratio,
    num_eventtypes,
    eventskew,
    max_parents,
    query_size,
    query_length,
    run,
):
    log_file_name = f"simulation_{os.getpid()}.log"  # Unique log per process
    sys.stdout = open(log_file_name, "a")  # Redirect stdout

    run_timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    log_separator = (
        f"\n{'=' * 40}\nSimulation Run #{run} - {run_timestamp}\n{'=' * 40}\n"
    )

    print(f"\n==== Simulation Run {run} Started ====\n")
    sys.stdout.flush()  # Force immediate write
    try:
        print(f"\n==== Initaiting Run {run} Started ====\n")
        simulation = INES(
            nodes,
            node_event_ratio,
            num_eventtypes,
            eventskew,
            max_parents,
            query_size,
            query_length,
        )
        sys.stdout.flush()
        return pd.DataFrame(
            [[simulation.function_times.get(key, None) for key in simulation.schema]],
            columns=simulation.schema,
        )  # Convert results to DataFrame

    except Exception as e:
        error_message = f"❌ Exception: {str(e)}\n{traceback.format_exc()}"
        log_file_name = f"simulation_errors_{os.getpid()}.log"
        print(error_message)
        sys.stdout.flush()
        # Log with a separator for each simulation run
        with open(log_file_name, "a") as log_file:
            log_file.write(log_separator)
            log_file.write(error_message)

        print(f"Error in simulation run {run} logged.")

        return None


def start_simulation(
    nodes,
    node_event_ratio,
    num_eventtypes,
    eventskew,
    max_parents,
    query_size,
    query_length,
    runs,
):
    """Runs multiple simulations in parallel."""
    file_name = f"INES-Benchmark" + datetime.now().strftime("%d%m%Y%H%M%S") + ".csv"
    result = run_simulation(
        nodes,
        node_event_ratio,
        num_eventtypes,
        eventskew,
        max_parents,
        query_size,
        query_length,
        0,
    )
    print(result)
    all_results = []

    with ProcessPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = [
            executor.submit(
                run_simulation,
                nodes,
                node_event_ratio,
                num_eventtypes,
                eventskew,
                max_parents,
                query_size,
                query_length,
                run,
            )
            for run in range(runs)
        ]

        for future in futures:
            print(f"🔄 Checking result for run {future}")
            sys.stdout.flush()  # Ensure logs are visible

            try:
                result = future.result(
                    timeout=300
                )  # Set timeout to avoid infinite wait
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                print(f"⚠️ Error in process: {e}")
            except TimeoutError:
                print(f"⏳ Timeout! Process took too long.")

    # Combine all DataFrames and write to CSV
    if all_results:
        all_results = [
            result for result in all_results if result is not None
        ]  # Remove failed runs
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(f"./res/{file_name}", index=False)
        print(f"Results saved to: {file_name}")


if __name__ == "__main__":
    # start_simulation(12, 0.5, 6, 0.3, 10, 3, 5, 4)
    file_name = f"INES-benchmark_" + datetime.now().strftime("%d%m%Y%H%M%S") + ".csv"
    all_results = []
    for i in range(3):
        # parallel laufen lassen
        result = run_simulation(50, 0.5, 6, 0.3, 10, 3, 5, i)
        all_results.append(result)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df.to_csv(f"./res/{file_name}", index=False)
            print(f"✅ Results saved to: ./res/{file_name}")
        else:
            print("❌ No valid results found.")
