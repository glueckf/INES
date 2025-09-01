from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import traceback
import logging
import pandas as pd

from INES import INES, SimulationConfig, SimulationMode


@dataclass
class SimulationRunner:
    """Configuration and execution class for running multiple INES simulations."""
    
    config: SimulationConfig
    num_runs: int = 1
    output_dir: Path = Path("./kraken/result")
    
    def __post_init__(self):
        """Initialize output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
        Execute multiple simulation runs and append results to persistent CSV files.
        
        Runs simulations sequentially and appends each result to ines_results.csv.
        The kraken results are handled separately in the kraken module.
        """
        ines_filepath = self.output_dir / "ines_results.csv"
        
        print(f"[INES] Starting {self.num_runs} simulation runs...")
        print(f"[CONFIG] Mode: {self.config.mode.value}")
        print(f"[CONFIG] Network size: {self.config.network_size}")
        print(f"[CONFIG] Query parameters: size={self.config.query_size}, length={self.config.query_length}")
        
        successful_runs = 0
        
        for run_id in range(self.num_runs):
            result = self.run_single_simulation(run_id)
            if result is not None:
                self._append_ines_result(result, ines_filepath)
                successful_runs += 1
        
        print(f"[RESULTS] {successful_runs} successful runs appended to: {ines_filepath.name}")
        
        if successful_runs < self.num_runs:
            failed_runs = self.num_runs - successful_runs
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
    query_size: int = 3,
    query_length: int = 5,
    num_runs: int = 1,
    mode: SimulationMode = SimulationMode.FULLY_DETERMINISTIC
) -> SimulationRunner:
    """
    Create a SimulationRunner with the specified parameters.
    
    Args:
        network_size: Number of nodes in the network topology
        node_event_ratio: Ratio of nodes that generate events
        num_event_types: Number of different event types
        event_skew: Skewness parameter for event distribution
        max_parents: Maximum number of parent nodes per node
        query_size: Number of queries in the workload
        query_length: Average length of each query
        num_runs: Number of simulation runs to execute
        mode: Simulation mode determining what components are fixed/random
        
    Returns:
        Configured SimulationRunner instance
    """
    config = SimulationConfig(
        network_size=network_size,
        node_event_ratio=node_event_ratio,
        num_event_types=num_event_types,
        event_skew=event_skew,
        max_parents=max_parents,
        query_size=query_size,
        query_length=query_length,
        mode=mode
    )
    
    return SimulationRunner(config=config, num_runs=num_runs)


def main() -> None:
    """Main entry point for simulation execution."""
    network_sizes = [10, 30, 50, 100, 200]
    
    for network_size in network_sizes:
        print(f"\n[INES] Starting simulations for network size: {network_size}")
        runner = create_simulation_runner(
            network_size=network_size,
            node_event_ratio=0.5,
            num_event_types=6,
            event_skew=2.0,
            max_parents=5,
            query_size=5,
            query_length=5,
            num_runs=50,
            mode=SimulationMode.RANDOM
        )
        
        runner.run_simulation_batch()
        print(f"[INES] Network size {network_size} completed successfully")
    
    print("\n[INES] All network size simulations completed successfully")


if __name__ == "__main__":
    main()
