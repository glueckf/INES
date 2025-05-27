"""
Module for processing projection strings and event type manipulations.

This module provides utilities for handling event type strings, including
numbering, filtering, and generating combinations of event types.
"""

import string
import re
from itertools import combinations


def generate_event_pairs(event_types):
    """
    Generate all unique pairs from a list of event types.
    
    Args:
        event_types: List of event type strings
        
    Returns:
        List of concatenated string pairs
        
    Example:
        generate_event_pairs(['A', 'B', 'C']) -> ['AB', 'AC', 'BC']
    """
    unique_types = list(set(event_types))
    return [''.join(pair) for pair in combinations(unique_types, 2)]


def remove_numbers_from_string(input_string):
    """
    Remove all digit characters from a string.
    
    Args:
        input_string: String or list to filter
        
    Returns:
        String with all digits removed
        
    Example:
        remove_numbers_from_string("A1B2C3") -> "ABC"
    """
    if isinstance(input_string, list):
        input_string = [str(x) for x in input_string]
    
    filtered_chars = [char for char in str(input_string) if not char.isdigit()]
    return ''.join(filtered_chars)


def swap_string_order(two_char_string):
    """
    Swap the order of characters in a two-character string.
    
    Args:
        two_char_string: String with exactly 2 characters
        
    Returns:
        String with characters in reversed order
        
    Example:
        swap_string_order("AB") -> "BA"
    """
    if len(two_char_string) != 2:
        raise ValueError("Input must be exactly 2 characters")
    
    return two_char_string[1] + two_char_string[0]


def find_duplicate_event_types(projection_key):
    """
    Find event types that appear multiple times in a projection key.
    
    Args:
        projection_key: String containing event types with optional numbers
        
    Returns:
        Dict mapping duplicate event types to their count
        
    Example:
        find_duplicate_event_types("A1B2A3C") -> {'A': 2}
    """
    event_list = parse_event_string(projection_key)
    event_types_only = [remove_numbers_from_string(event) for event in event_list]
    unique_types = list(set(event_types_only))
    
    duplicates = {}
    for event_type in unique_types:
        count = event_types_only.count(event_type)
        if count > 1:
            duplicates[event_type] = count
    
    return duplicates


def normalize_projection_key(projection_key):
    """
    Normalize projection key by renumbering duplicate event types sequentially.
    
    Takes care of query projections by ensuring consistent numbering.
    
    Args:
        projection_key: String like "A1B2C3A2" or "A1A3C"
        
    Returns:
        Normalized string like "A1BCA2" or "A1A2C"
        
    Example:
        normalize_projection_key("A1B2C3A2") -> "A1A2BC"
        normalize_projection_key("A1A3C") -> "A1A2C"
    """
    duplicates = find_duplicate_event_types(projection_key)
    result_events = []
    
    # Add non-duplicate events without numbers
    for event in parse_event_string(projection_key):
        event_type = remove_numbers_from_string(event)
        if event_type not in duplicates:
            result_events.append(event_type)
    
    # Add duplicate events with sequential numbering
    for event_type, count in duplicates.items():
        for i in range(1, count + 1):
            result_events.append(f"{event_type}{i}")
    
    return ''.join(sorted(result_events))


def add_sequential_numbering(projection_key):
    """
    Add sequential numbering to duplicate event types in a projection string.
    
    Args:
        projection_key: String like "AAB" or "ABCA"
        
    Returns:
        String with numbered duplicates like "A1A2B" or "A1BA2"
        
    Example:
        add_sequential_numbering("AAB") -> "A1A2B"
        add_sequential_numbering("ABCA") -> "A1BA2"
    """
    unique_types = list(set(projection_key))
    result = list(projection_key)
    
    for event_type in unique_types:
        occurrences = result.count(event_type)
        if occurrences > 1:
            counter = 1
            for i, char in enumerate(result):
                if char == event_type:
                    result[i] = f"{event_type}{counter}"
                    counter += 1
    
    return ''.join(result)


def parse_event_string(event_string):
    """
    Parse an event string into individual event components.
    
    Separates event types from their numbers.
    
    Args:
        event_string: String like "A1B2C" or "ABC"
        
    Returns:
        List of event components like ["A1", "B2", "C"] or ["A", "B", "C"]
        
    Example:
        parse_event_string("A1B2C") -> ["A1", "B2", "C"]
        parse_event_string("ABC") -> ["A", "B", "C"]
    """
    if not any(char.isdigit() for char in event_string):
        return list(event_string)
    
    # Use regex to split on uppercase letters while keeping them
    pattern = r'([A-Z][0-9]*)'
    events = re.findall(pattern, event_string)
    
    return events


# Legacy functions for backward compatibility
def generate_twosets(match):
    """Legacy function - use generate_event_pairs instead."""
    return generate_event_pairs(match)


def filter_numbers(in_string):
    """Legacy function - use remove_numbers_from_string instead."""
    return remove_numbers_from_string(in_string)


def changeorder(duo):
    """Legacy function - use swap_string_order instead."""
    return swap_string_order(duo)


def getdoubles_k(subopkey):
    """Legacy function - use find_duplicate_event_types instead."""
    return find_duplicate_event_types(subopkey)


def rename_without_numbers(projkey):
    """Legacy function - use normalize_projection_key instead."""
    return normalize_projection_key(projkey)


def add_numbering(projkey):
    """Legacy function - use add_sequential_numbering instead."""
    return add_sequential_numbering(projkey)


def sepnumbers(evlist):
    """Legacy function - use parse_event_string instead."""
    return parse_event_string(evlist)


def printcombination(arr):
    """Legacy function - use generate_event_pairs instead."""
    return generate_event_pairs(arr)