"""
Tests for legacy function adapters.

These tests verify that adapters correctly forward calls to legacy
functions and validate input/output shapes.
"""

import pytest
import io
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from placement_engine.adapters import build_eval_plan, run_prepp, compute_central_plan


class TestAdapters:
    """Test suite for legacy function adapters."""
    
    def test_build_eval_plan_input_validation(self):
        """Test that build_eval_plan validates input shapes correctly."""
        
        # Valid inputs
        nw = [Mock(), Mock()]
        selectivities = {}
        my_plan = [Mock(), 123, {}]
        central_plan = [0, {}, []]
        workload = [Mock()]
        
        # Test invalid my_plan length
        with pytest.raises(ValueError, match="my_plan must be a list of length 3"):
            build_eval_plan(nw, selectivities, [Mock(), Mock()], central_plan, workload)
            
        # Test invalid central_plan length
        with pytest.raises(ValueError, match="central_plan must be a list of length 3"):
            build_eval_plan(nw, selectivities, my_plan, [Mock()], workload)
            
        # Test invalid workload type
        with pytest.raises(ValueError, match="workload must be a list"):
            build_eval_plan(nw, selectivities, my_plan, central_plan, "invalid")
            
        # Test invalid nw type
        with pytest.raises(ValueError, match="nw must be a list"):
            build_eval_plan("invalid", selectivities, my_plan, central_plan, workload)
    
    @patch('placement_engine.adapters.generate_eval_plan')
    def test_build_eval_plan_forwards_correctly(self, mock_generate):
        """Test that build_eval_plan forwards calls correctly."""
        
        # Setup
        mock_result = io.StringIO("test result")
        mock_generate.return_value = mock_result
        
        nw = [Mock(), Mock()]
        selectivities = {"test": 0.5}
        my_plan = [Mock(), 123, {}]
        central_plan = [0, {}, []]
        workload = [Mock()]
        
        # Call adapter
        result = build_eval_plan(nw, selectivities, my_plan, central_plan, workload)
        
        # Verify call forwarding
        mock_generate.assert_called_once_with(nw, selectivities, my_plan, central_plan, workload)
        assert result == mock_result
    
    def test_run_prepp_input_validation(self):
        """Test that run_prepp validates input shapes correctly."""
        
        # Valid inputs
        input_buffer = io.StringIO("test")
        method = "ppmuse"
        algorithm = "e"
        samples = 0
        top_k = 0
        runs = 1
        plan_print = True
        all_pairs = [[0, 1], [1, 0]]
        
        # Test invalid input_buffer type
        with pytest.raises(ValueError, match="input_buffer must be StringIO"):
            run_prepp("invalid", method, algorithm, samples, top_k, runs, plan_print, all_pairs)
            
        # Test invalid method type
        with pytest.raises(ValueError, match="method must be string"):
            run_prepp(input_buffer, 123, algorithm, samples, top_k, runs, plan_print, all_pairs)
            
        # Test invalid algorithm type
        with pytest.raises(ValueError, match="algorithm must be string"):
            run_prepp(input_buffer, method, 123, samples, top_k, runs, plan_print, all_pairs)
            
        # Test invalid all_pairs type
        with pytest.raises(ValueError, match="all_pairs must be a list"):
            run_prepp(input_buffer, method, algorithm, samples, top_k, runs, plan_print, "invalid")
    
    @patch('placement_engine.adapters.generate_prePP')
    def test_run_prepp_forwards_correctly(self, mock_generate):
        """Test that run_prepp forwards calls correctly."""
        
        # Setup
        mock_result = [10.5, 2.0, 5.0, 0.8]
        mock_generate.return_value = mock_result
        
        input_buffer = io.StringIO("test")
        method = "ppmuse"
        algorithm = "e"
        samples = 0
        top_k = 0
        runs = 1
        plan_print = True
        all_pairs = [[0, 1], [1, 0]]
        
        # Call adapter
        result = run_prepp(input_buffer, method, algorithm, samples, top_k, runs, plan_print, all_pairs, False)
        
        # Verify call forwarding
        mock_generate.assert_called_once_with(input_buffer, method, algorithm, samples, top_k, runs, plan_print, all_pairs, False)
        assert result == mock_result
    
    def test_compute_central_plan_input_validation(self):
        """Test that compute_central_plan validates input shapes correctly."""
        
        # Valid inputs
        workload = [Mock()]
        index_event_nodes = {"A": []}
        all_pairs = [[0, 1], [1, 0]]
        rates = {"A": 1.0}
        event_nodes = []
        graph = Mock()
        
        # Test invalid workload type
        with pytest.raises(ValueError, match="workload must be a list"):
            compute_central_plan("invalid", index_event_nodes, all_pairs, rates, event_nodes, graph)
            
        # Test invalid index_event_nodes type
        with pytest.raises(ValueError, match="index_event_nodes must be dict"):
            compute_central_plan(workload, "invalid", all_pairs, rates, event_nodes, graph)
            
        # Test invalid all_pairs type
        with pytest.raises(ValueError, match="all_pairs must be a list"):
            compute_central_plan(workload, index_event_nodes, "invalid", rates, event_nodes, graph)
            
        # Test invalid rates type
        with pytest.raises(ValueError, match="rates must be dict"):
            compute_central_plan(workload, index_event_nodes, all_pairs, "invalid", event_nodes, graph)
    
    @patch('placement_engine.adapters.NEWcomputeCentralCosts')
    def test_compute_central_plan_forwards_correctly(self, mock_compute):
        """Test that compute_central_plan forwards calls correctly."""
        
        # Setup
        mock_result = (15.5, 2, 3.0, {"A": {"etb1": [0, 2]}})
        mock_compute.return_value = mock_result
        
        workload = [Mock()]
        index_event_nodes = {"A": []}
        all_pairs = [[0, 1], [1, 0]]
        rates = {"A": 1.0}
        event_nodes = []
        graph = Mock()
        
        # Call adapter
        result = compute_central_plan(workload, index_event_nodes, all_pairs, rates, event_nodes, graph)
        
        # Verify call forwarding
        mock_compute.assert_called_once_with(workload, index_event_nodes, all_pairs, rates, event_nodes, graph)
        assert result == mock_result


if __name__ == "__main__":
    pytest.main([__file__])