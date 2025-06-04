#!/usr/bin/env python3
"""
Example: Generate workloads for multimodal and reasoning models.

This script demonstrates how to:
1. Generate multimodal workloads with image/audio/video tokens
2. Generate reasoning workloads with reason_ratio field
"""

from servegen import Category, ClientPool
from servegen.construct import generate_workload
from servegen.utils import get_constant_rate_fn
import numpy as np

def print_sample_requests(requests, num_samples=3):
    """Print a few sample requests to show the workload structure."""
    print("\nSample Requests:")
    for i, req in enumerate(requests[:num_samples]):
        print(f"\nRequest {i+1}:")
        print(f"  Timestamp: {req.timestamp:.2f}")
        print(f"  Data:")
        for field, value in req.data.items():
            if isinstance(value, list):
                print(f"    {field}: {value}")
            else:
                print(f"    {field}: {value}")

def generate_multimodal_workload():
    """Generate and analyze a multimodal workload."""
    print("\n=== Generating Multimodal Workload ===")
    
    # Create a client pool for multimodal data
    pool = ClientPool(Category.MULTIMODAL, "mm-image")
    print(f"Loaded {len(pool.clients)} multimodal clients")
    
    # Create a view for the first hour
    view = pool.span(0, 600)
    rate_fn = get_constant_rate_fn(view, 50.0)  # 50 requests per second
    
    # Generate workload
    requests = generate_workload(view, rate_fn, duration=600, seed=321)
    
    # Print statistics
    print("\nMultimodal Workload Statistics:")
    print(f"Total Requests: {len(requests)}")
    print(f"Time Range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}")
    
    # Calculate average token counts
    avg_text = sum(r.data['text_tokens'] for r in requests) / len(requests)
    avg_output = sum(r.data['output_tokens'] for r in requests) / len(requests)
    avg_image = sum(len(r.data['image_tokens']) for r in requests) / len(requests)
    avg_audio = sum(len(r.data['audio_tokens']) for r in requests) / len(requests)
    avg_video = sum(len(r.data['video_tokens']) for r in requests) / len(requests)
    
    print("\nAverage Token Counts:")
    print(f"  Text tokens: {avg_text:.2f}")
    print(f"  Output tokens: {avg_output:.2f}")
    print(f"  Image count: {avg_image:.2f}")
    print(f"  Audio count: {avg_audio:.2f}")
    print(f"  Video count: {avg_video:.2f}")
    
    # Print sample requests
    print_sample_requests(requests, num_samples=5)

def generate_reasoning_workload():
    """Generate and analyze a reasoning workload."""
    print("\n=== Generating Reasoning Workload ===")
    
    # Create a client pool for reasoning data
    pool = ClientPool(Category.REASON, "deepseek-r1")
    print(f"Loaded {len(pool.clients)} reasoning clients")
    
    # Create a view for the first hour
    view = pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 50.0)  # 50 requests per second
    
    # Generate workload
    requests = generate_workload(view, rate_fn, duration=3600, seed=0)
    
    # Print statistics
    print("\nReasoning Workload Statistics:")
    print(f"Total Requests: {len(requests)}")
    print(f"Time Range: {requests[0].timestamp:.2f} to {requests[-1].timestamp:.2f}")
    
    # Calculate average token counts and reason ratio
    avg_input = sum(r.data['input_tokens'] for r in requests) / len(requests)
    avg_output = sum(r.data['output_tokens'] for r in requests) / len(requests)
    avg_ratio = sum(r.data['reason_ratio'] for r in requests) / len(requests)
    
    print("\nAverage Values:")
    print(f"  Input tokens: {avg_input:.2f}")
    print(f"  Output tokens: {avg_output:.2f}")
    print(f"  Reason ratio: {avg_ratio:.2f}")
    
    # Print sample requests
    print_sample_requests(requests)

def main():
    generate_multimodal_workload()
    generate_reasoning_workload()

if __name__ == "__main__":
    main() 