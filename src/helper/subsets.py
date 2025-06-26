"""
Module for generating combinations and subsets from arrays.

This module provides utility functions to generate combinations of elements
from arrays in different formats (strings, lists, comma-separated strings).
"""

from itertools import combinations


def generate_string_combinations(arr, r):
    """
    Generate combinations of size r from array, returning concatenated strings.
    
    Args:
        arr: List of elements to combine
        r: Size of each combination
        
    Returns:
        List of strings, each representing a combination
        
    Example:
        generate_string_combinations(['A', 'B', 'C'], 2) -> ['AB', 'AC', 'BC']
    """
    return [''.join(combo) for combo in combinations(arr, r)]


def generate_list_combinations(arr, r):
    """
    Generate combinations of size r from array, returning lists.
    
    Args:
        arr: List of elements to combine
        r: Size of each combination
        
    Returns:
        List of lists, each representing a combination
        
    Example:
        generate_list_combinations(['A', 'B', 'C'], 2) -> [['A', 'B'], ['A', 'C'], ['B', 'C']]
    """
    return [list(combo) for combo in combinations(arr, r)]


def generate_csv_combinations(arr, r):
    """
    Generate combinations of size r from array, returning comma-separated strings.
    
    Args:
        arr: List of elements to combine
        r: Size of each combination
        
    Returns:
        List of comma-separated strings, each representing a combination
        
    Example:
        generate_csv_combinations(['A', 'B', 'C'], 2) -> ['A,B', 'A,C', 'B,C']
    """
    return [','.join(combo) for combo in combinations(arr, r)]