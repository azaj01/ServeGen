#!/usr/bin/env python3
"""
Example: Generate a workload and save it as a CSV file.

This script demonstrates how to:
1. Basic usage of load client data and generate a workload
2. Save the generated workload to a CSV file
"""

from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import save_requests_to_csv


def main():
    # Load client data
    print("Loading client pool...")
    pool = ClientPool(Category.LANGUAGE, "m-large")
    print(f"Loaded {len(pool.clients)} clients.")

    # Generate workload
    print("Generating workload...")
    rate_fn = {0: 100.0, 600: 150.0}  # requests per second
    requests = generate_workload(pool, rate_fn, duration=1200)
    print(f"Generated {len(requests)} requests.")

    # Save to CSV
    output_file = "workload.csv"
    print(f"Saving to {output_file}...")
    save_requests_to_csv(requests, output_file)
    print(f"Saved to {output_file}.")

    # Print first few rows
    print("\nFirst 10 rows of the generated workload:")
    with open(output_file, "r") as f:
        lines = f.readlines()
        for line in lines[:11]:  # header + 10 rows
            print(line.strip())


if __name__ == "__main__":
    main()
