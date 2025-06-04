#!/usr/bin/env python3
"""
Example: Generate workloads using custom clients with custom patterns.

This script demonstrates how to:
1. Create custom clients with custom (bursty vs stable) workload patterns
2. Mix clients with different characteristics 
3. Generate workloads with custom rate functions
"""

from servegen import Category, Client, ClientPool
from servegen.construct import generate_workload
import numpy as np
from scipy import stats

def create_bursty_client(client_id: int) -> Client:
    """Create a bursty client with concentrated input/output distributions.
    
    This client characterizes an API application, that:
    - Makes requests in bursts (low shape parameter in Gamma and thus high CV)
    - Has concentrated input/output lengths (similar input and output patterns)
    """
    return Client(
        client_id=client_id,
        trace={
            0: {"rate": 4.0, "cv": 2.0, "pat": ("Gamma", (0.25, 1.0))},  
            60: {"rate": 3.33, "cv": 2.0, "pat": ("Gamma", (0.25, 1.2))},
            120: {"rate": 4.0, "cv": 2.0, "pat": ("Gamma", (0.25, 1.0))},
        },
        dataset={
            0: {
                # Concentrated around 1000 tokens for input, 200 for output
                "input_tokens": [0.0] * 1000 + [0.1, 0.2, 0.4, 0.2, 0.1],
                "output_tokens": [0.0] * 200 + [0.1, 0.2, 0.4, 0.2, 0.1],
            },
            60: {
                # Similar distribution but slightly shifted
                "input_tokens": [0.0] * 1010 + [0.1, 0.2, 0.4, 0.2, 0.1],
                "output_tokens": [0.0] * 190 + [0.1, 0.2, 0.4, 0.2, 0.1],
            },
            120: {
                # Back to original distribution
                "input_tokens": [0.0] * 1000 + [0.1, 0.2, 0.4, 0.2, 0.1],
                "output_tokens": [0.0] * 200 + [0.1, 0.2, 0.4, 0.2, 0.1],
            },
        }
    )

def create_stable_client(client_id: int) -> Client:
    """Create a stable client with spread out input/output distributions.
    
    This client characterizes human interaction with a chatbot, which
    - Has consistent request rates (shape and CV equal to 1)
    - Has varied input/output lengths following Pareto (input) and Exponential (output) distributions
    (Or, provide other custom distributions from existing datasets, like ShareGPT.)
    """
    # Generate Pareto distribution for input tokens (mean = 500)
    # For Pareto, mean = a * b / (a-1) where a > 1
    # We use a = 2.5, then b = mean * (a-1)/a = 500 * 1.5/2.5 = 300
    pareto_a, pareto_b = 2.5, 300
    input_pdf = stats.pareto.pdf(np.arange(32768), pareto_a, scale=pareto_b)
    input_pdf = input_pdf / np.sum(input_pdf)  # Normalize to sum to 1

    # Generate Exponential distribution for output tokens (mean = 300)
    # For Exponential, mean = 1/lambda
    exp_lambda = 1/300
    output_pdf = stats.expon.pdf(np.arange(32768), scale=1/exp_lambda)
    output_pdf = output_pdf / np.sum(output_pdf)  # Normalize to sum to 1

    return Client(
        client_id=client_id,
        trace={
            0: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 0.5))}, 
            60: {"rate": 2.5, "cv": 1.0, "pat": ("Gamma", (1.0, 0.4))},
            120: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 0.5))},
        },
        dataset={
            0: {
                "input_tokens": input_pdf.tolist(),
                "output_tokens": output_pdf.tolist(),
            },
            60: {
                "input_tokens": input_pdf.tolist(),
                "output_tokens": input_pdf.tolist(),
            },
            120: {
                "input_tokens": input_pdf.tolist(),
                "output_tokens": output_pdf.tolist(),
            },
        }
    )

def main():
    # Create custom clients with contrasting patterns
    print("Creating custom clients...")
    bursty_client = create_bursty_client(1)
    stable_client = create_stable_client(2)
    print("Created 2 clients with contrasting patterns:")
    print("  - Bursty client: High CV, concentrated distributions")
    print("  - Stable client: Low CV, Pareto input and Exponential output distributions")

    # Create pool from custom clients
    print("\nCreating client pool...")
    pool = ClientPool.from_clients(Category.LANGUAGE, "custom", [bursty_client, stable_client])
    print(f"Pool contains {len(pool.clients)} clients")

    # Generate workload with custom rate function
    print("\nGenerating workload with custom rate function...")
    rate_fn = {
        0: 10.0,        # 10 requests/second in first window
        60: 15.0,       # 15 requests/second in second window
        120: 8.0,       # 8 requests/second in third window
    }
    requests = generate_workload(pool, rate_fn, duration=180, seed=0)
    print_workload_stats(requests)

def print_workload_stats(requests):
    """Print statistics about the generated workload."""
    print("\nWorkload statistics:")
    print(f"  Time range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}")
    
    # Calculate average input/output lengths
    avg_input = sum(r.data['input_tokens'] for r in requests)/len(requests)
    avg_output = sum(r.data['output_tokens'] for r in requests)/len(requests)
    print(f"  Average input length: {avg_input:.2f}")
    print(f"  Average output length: {avg_output:.2f}")

    # Calculate statistics per window
    windows = [
        ("First window (0-60s)", [r for r in requests if r.timestamp < 60]),
        ("Second window (60-120s)", [r for r in requests if 60 <= r.timestamp < 120]),
        ("Third window (120-180s)", [r for r in requests if r.timestamp >= 120])
    ]

    for window_name, window_requests in windows:
        if not window_requests:
            continue
            
        # Calculate rate
        rate = len(window_requests) / 60 
        
        # Calculate CV
        if len(window_requests) > 1:
            timestamps = [r.timestamp for r in window_requests]
            iats = np.diff(timestamps)  # inter-arrival times
            cv = np.std(iats) / np.mean(iats)
        else:
            cv = 0.0
            
        print(f"\n{window_name}:")
        print(f"  Rate: {rate:.2f} req/s")
        print(f"  CV: {cv:.2f}")

if __name__ == "__main__":
    main() 