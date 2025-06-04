#!/usr/bin/env python3
"""
Example: Generate workloads using real-world client data.

This script demonstrates how to:
1. Load client data from JSON files
2. Create time window views
3. Generate workloads with different rate functions
"""

from servegen import Category, ClientPool, generate_workload
from servegen.utils import get_constant_rate_fn, get_bounded_rate_fn

def main():
    # Load client pool from JSON data
    print("Loading client pool...")
    pool = ClientPool(Category.LANGUAGE, "m-large")
    print(f"Loaded {len(pool.clients)} clients")

    # Create a view for 18:00-19:00
    print("\nCreating view for 18:00-19:00...")
    view = pool.span(64800, 68400)
    windows = view.get()
    print(f"View contains {len(windows)} windows")

    # Generate workload with constant rate
    print("\nGenerating workload with constant rate...")
    rate_fn = get_constant_rate_fn(view, 100.0)  # 100 requests per second
    requests = generate_workload(view, rate_fn, duration=3600, seed=0)
    print_workload_stats(requests, "Constant rate workload")

    # Create a view for 11:00-12:00
    print("\nCreating view for 11:00-12:00...")
    view = pool.span(39600, 43200)
    windows = view.get()
    print(f"View contains {len(windows)} windows")

    # Generate workload with bounded rate
    print("\nGenerating workload with bounded rate...")
    rate_fn = get_bounded_rate_fn(view, 100.0)  # maximum 100 requests per second
    requests = generate_workload(view, rate_fn, duration=3600, seed=0)
    print_workload_stats(requests, "Bounded rate workload")

    # Note there's a 10% shift in data distributions!
    
def print_workload_stats(requests, title):
    """Print statistics about the generated workload."""
    print(f"\n{title} statistics:")
    print(f"  Total requests: {len(requests)}")
    print(f"  Time range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}")
    print(f"  Average input length: {sum(r.data['input_tokens'] for r in requests)/len(requests):.2f}")
    print(f"  Average output length: {sum(r.data['output_tokens'] for r in requests)/len(requests):.2f}")

if __name__ == "__main__":
    main() 