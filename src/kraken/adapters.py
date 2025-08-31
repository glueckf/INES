"""
Legacy function adapters.

This module provides thin wrapper functions that adapt legacy function
calls to work within the new placement engine architecture. These adapters
are zero-logic wrappers that simply forward calls to legacy functions
while documenting their expected shapes and providing clear error messages.
"""

from typing import Any, Dict, List, Tuple
import io
from .logging import get_kraken_logger

logger = get_kraken_logger(__name__)


def build_eval_plan(nw: List[Any], selectivities: Dict[str, Any], my_plan: List[Any], 
                   central_plan: List[Any], workload: List[Any]) -> io.StringIO:
    """
    Adapter for generate_eval_plan function.
    
    This adapter validates inputs and forwards the call to the legacy generate_eval_plan
    function. It provides shape documentation and error handling.
    
    Args:
        nw: Network nodes list
        selectivities: Selectivity parameters dictionary
        my_plan: Evaluation plan structure [plan_obj, id, ms_placements_dict]
        central_plan: Central plan structure [source_node, routing_dict, workload]
        workload: List of queries/projections to evaluate
        
    Returns:
        io.StringIO: Buffer containing the generated evaluation plan
        
    Raises:
        ValueError: If input shapes don't match expected format
        ImportError: If legacy function cannot be imported
    """
    try:
        # Validate basic input shapes
        if not isinstance(my_plan, list) or len(my_plan) != 3:
            raise ValueError(f"my_plan must be a list of length 3, got {type(my_plan)} with length {len(my_plan) if isinstance(my_plan, list) else 'N/A'}")
            
        if not isinstance(central_plan, list) or len(central_plan) != 3:
            raise ValueError(f"central_plan must be a list of length 3, got {type(central_plan)} with length {len(central_plan) if isinstance(central_plan, list) else 'N/A'}")
            
        if not isinstance(workload, list):
            raise ValueError(f"workload must be a list, got {type(workload)}")
            
        if not isinstance(nw, list):
            raise ValueError(f"nw must be a list, got {type(nw)}")
        
        logger.debug(f"build_eval_plan called with nw={len(nw)} nodes, workload={len(workload)} items")
        
        # Import and call legacy function
        from generateEvalPlan import generate_eval_plan
        
        result = generate_eval_plan(nw, selectivities, my_plan, central_plan, workload)
        
        # Validate result
        if not isinstance(result, io.StringIO):
            logger.warning(f"generate_eval_plan returned unexpected type: {type(result)}")
            
        return result
        
    except ImportError as e:
        logger.error(f"Failed to import legacy generate_eval_plan: {e}")
        raise ImportError(f"Cannot import legacy generate_eval_plan function: {e}")
    except Exception as e:
        logger.error(f"Error in build_eval_plan adapter: {e}")
        raise


def run_prepp(input_buffer: io.StringIO, method: str, algorithm: str, samples: int, 
              top_k: int, runs: int, plan_print: bool, all_pairs: List[List[float]], 
              is_deterministic: bool = False) -> List[Any]:
    """
    Adapter for generate_prePP function.
    
    This adapter validates inputs and forwards the call to the legacy generate_prePP
    function for push-pull cost calculation.
    
    Args:
        input_buffer: Buffer containing evaluation plan
        method: PrePP method (e.g., "ppmuse")  
        algorithm: Algorithm type ("e" for exact, etc.)
        samples: Number of samples for approximation
        top_k: Top-k parameter
        runs: Number of runs
        plan_print: Whether to print detailed plan information
        all_pairs: All-pairs distance matrix
        is_deterministic: Whether to use deterministic behavior for reproducibility
        
    Returns:
        List: PrePP results [exact_cost, pushPullTime, maxPushPullLatency, endTransmissionRatio]
        
    Raises:
        ValueError: If input shapes don't match expected format
        ImportError: If legacy function cannot be imported
    """
    try:
        # Validate inputs
        if not isinstance(input_buffer, io.StringIO):
            raise ValueError(f"input_buffer must be StringIO, got {type(input_buffer)}")
            
        if not isinstance(method, str):
            raise ValueError(f"method must be string, got {type(method)}")
            
        if not isinstance(algorithm, str):
            raise ValueError(f"algorithm must be string, got {type(algorithm)}")
            
        if not isinstance(all_pairs, list):
            raise ValueError(f"all_pairs must be a list, got {type(all_pairs)}")
            
        # Check all_pairs is a square matrix
        if all_pairs and not isinstance(all_pairs[0], list):
            raise ValueError("all_pairs must be a list of lists (matrix)")
            
        logger.debug(f"run_prepp called with method={method}, algorithm={algorithm}, matrix_size={len(all_pairs) if all_pairs else 0}")
        
        # Import and call legacy function
        from prepp import generate_prePP
        
        result = generate_prePP(input_buffer, method, algorithm, samples, top_k, runs, plan_print, all_pairs, is_deterministic)
        
        # Validate result
        if not isinstance(result, list):
            logger.warning(f"generate_prePP returned unexpected type: {type(result)}")
            
        return result
        
    except ImportError as e:
        logger.error(f"Failed to import legacy generate_prePP: {e}")
        raise ImportError(f"Cannot import legacy generate_prePP function: {e}")
    except Exception as e:
        logger.error(f"Error in run_prepp adapter: {e}")
        raise

