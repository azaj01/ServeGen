#!/usr/bin/env python3
"""
Example: Analyze client data and get CDFs.

This script demonstrates how to:
1. Load and filter client data
2. Get CDFs of client behaviors
3. Analyze different statistics from the CDFs
"""

from servegen import Category, ClientPool
import numpy as np

def print_cdf_info(cdfs, field, timestamp):
    """Print information about a specific CDF."""
    if field not in cdfs:
        print(f"No CDFs available for {field}")
        return
    
    if timestamp not in cdfs[field]:
        print(f"No CDF available for {field} at timestamp {timestamp}")
        return
    
    values, probs = cdfs[field][timestamp]
    print(f"\n{field.upper()} CDF at timestamp {timestamp}:")
    print(f"  Values: {values}")
    print(f"  Probabilities: {probs}")

def print_dataset_stats(cdfs, field, timestamp):
    """Print statistics for a dataset field."""
    if field not in cdfs:
        print(f"No statistics available for {field}")
        return
    
    print(f"\n{field.upper()} statistics at timestamp {timestamp}:")
    for stat in ["avg", "p50", "p95", "p99"]:
        if stat in cdfs[field] and timestamp in cdfs[field][stat]:
            values, probs = cdfs[field][stat][timestamp]
            print(f"  {stat.upper()}:")
            print(f"    Values: {values}")
            print(f"    Probabilities: {probs}")

def main():
    # Load client pool
    print("Loading client pool...")
    pool = ClientPool(Category.LANGUAGE, "m-large")
    print(f"Loaded {len(pool.clients)} clients")

    # Filter clients by various criteria
    print("\nFiltering clients...")
    filtered_view = (
        pool
        .span(72000, 75600)
        .filter_by_cv(0.5, 1.5)                 # Filter by coefficient of variation
        .filter_by_avg_input_len(100, 1000)     # Filter by average input length
        .filter_by_max_output_len(2000)         # Filter by maximum output length
    )
    windows = filtered_view.get()
    print(f"Filtered view contains {len(windows)} windows")

    # Get CDFs of client behaviors
    print("\nComputing CDFs...")
    cdfs = filtered_view.get_cdfs()

    # Print information about available CDFs
    print("\nAvailable CDFs:")
    for field in cdfs:
        if field in ["rate", "cv"]:
            timestamps = sorted(cdfs[field].keys())
            print(f"  {field}: {len(timestamps)} timestamps")
        else:
            stats = cdfs[field].keys()
            print(f"  {field}: {len(stats)} statistics")

    # Print detailed information for the first window
    first_ts = min(cdfs["rate"].keys())
    print_cdf_info(cdfs, "rate", first_ts)
    print_cdf_info(cdfs, "cv", first_ts)
    print_dataset_stats(cdfs, "input_tokens", first_ts)
    print_dataset_stats(cdfs, "output_tokens", first_ts)

if __name__ == "__main__":
    main() 