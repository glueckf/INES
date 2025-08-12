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
    output_dir: Path = Path("./res")
    
    def __post_init__(self):
        """Initialize output directory if it doesn't exist."""
        self.output_dir.mkdir(exist_ok=True)
    
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
        Execute multiple simulation runs and save results to CSV.
        
        Runs simulations sequentially to avoid multiprocessing complexities.
        Results are aggregated and saved to a timestamped CSV file.
        """
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        filename = f"INES-simulation_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        print(f"[INES] Starting {self.num_runs} simulation runs...")
        print(f"[CONFIG] Mode: {self.config.mode.value}")
        print(f"[CONFIG] Network size: {self.config.network_size}")
        print(f"[CONFIG] Query parameters: size={self.config.query_size}, length={self.config.query_length}")
        
        successful_results: List[pd.DataFrame] = []
        
        for run_id in range(self.num_runs):
            result = self.run_single_simulation(run_id)
            if result is not None:
                successful_results.append(result)
        
        self._save_results(successful_results, filepath)
    
    def _save_results(self, results: List[pd.DataFrame], filepath: Path) -> None:
        """
        Save aggregated results to CSV file.
        
        Args:
            results: List of DataFrames containing simulation results
            filepath: Path where results should be saved
        """
        if not results:
            print("[RESULTS] Warning: No successful runs to save")
            return
            
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(filepath, index=False)
        print(f"[RESULTS] {len(results)} successful runs saved to: {filepath.name}")
        
        if len(results) < self.num_runs:
            failed_runs = self.num_runs - len(results)
            print(f"[RESULTS] Warning: {failed_runs} run(s) failed")


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
    runner = create_simulation_runner(
        network_size=12,
        node_event_ratio=0.5,
        num_event_types=3,
        event_skew=0.3,
        max_parents=10,
        query_size=1,
        query_length=3,
        num_runs=1,
        mode=SimulationMode.FULLY_DETERMINISTIC
    )
    
    runner.run_simulation_batch()
    print("[INES] Simulation completed successfully")


if __name__ == "__main__":
    main()
