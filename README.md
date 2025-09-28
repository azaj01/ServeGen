# ServeGen

ServeGen is a framework for generating realistic large language model (LLM) serving workloads.
Powered by the analysis of billions of inference requests across 12 production models on Alibaba Cloud Model Studio ([百炼](https://www.aliyun.com/product/bailian)), ServeGen is able to replicate the nuanced complexity of real-world workloads, such as:

+ **Bursty** request arrivals beyond simple Poisson models
+ **Shifting** input/output length distributions over days and weeks
+ **Heterogeneous** data composition in multimodal workloads (Qwen-VL)
+ **Bimodal** reasoning length distribution in reasoning workloads (DeepSeek-R1)

We hope ServeGen can become a data-driven bridge between frontier research and production realities when designing and deploying new LLM serving systems.

For more detailed analysis results, check out our characterization [paper](https://www.arxiv.org/abs/2505.09999)!


## Requirements

ServeGen requires Python 3.8 or higher and the following dependencies:

- **numpy** (>=1.20.0): For numerical computations and array operations
- **scipy** (>=1.7.0): For statistical distributions and sampling
- **pytest** (>=7.0.0): For running tests (optional, only needed for development)

You can install all dependencies and this project using pip:

```bash
pip install -r requirements.txt
pip install -e .
```

## Examples

### Basic Usage

```python
from servegen import Category, ClientPool
from servegen.construct import generate_workload

# Load client data
pool = ClientPool(Category.LANGUAGE, "m-large")

# Generate workload
rate_fn = {0: 100.0, 600: 150.0}  # requests per second
requests = generate_workload(pool, rate_fn, duration=1200)
```

### Custom Workloads

```python
# Create custom clients with different patterns
bursty_client = create_bursty_client(1)  # High CV, concentrated distributions
stable_client = create_stable_client(2)  # Low CV, Pareto/Exponential distributions

# Generate workload with custom rate function
rate_fn = {0: 10.0, 60: 15.0, 120: 8.0}
requests = generate_workload(pool, rate_fn, duration=180)
```

### Multimodal and Reasoning Workloads

```python
# Generate multimodal workload
pool = ClientPool(Category.MULTIMODAL, "mm-image")
requests = generate_workload(pool, rate_fn, duration=600)

# Generate reasoning workload
pool = ClientPool(Category.REASON, "deepseek-r1")
requests = generate_workload(pool, rate_fn, duration=3600)
```

See `examples/` for more detailed examples:
- `basic_usage.py`: Basic workload generation and saving to CSV
- `generate_custom.py`: Custom workload patterns
- `generate_realistic.py`: Realistic workload generation
- `generate_advanced.py`: Multimodal and reasoning workloads
- `clientpool_example.py`: Client pool analysis and filtering

### Filtering Client Data and Getting CDFs

```python
from servegen import Category, ClientPool
import numpy as np

# Load client pool
pool = ClientPool(Category.LANGUAGE, "m-large")

# Filter clients by various criteria
filtered_view = (
    pool
    .span(72000, 75600)  # 20:00-21:00
    .filter_by_cv(0.5, 1.5)  # Filter by coefficient of variation
    .filter_by_avg_input_len(100, 1000)  # Filter by average input length
    .filter_by_max_output_len(2000)  # Filter by maximum output length
)

# Get CDFs of client behaviors
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

# Print detailed information for the first timestamp
first_ts = min(cdfs["rate"].keys())
values, probs = cdfs["rate"][first_ts]
print(f"\nRate CDF at timestamp {first_ts}:")
print(f"  Values: {values}")
print(f"  Probabilities: {probs}")

# Print statistics for input tokens
print("\nInput token statistics:")
for stat in ["avg", "p50", "p95", "p99"]:
    if stat in cdfs["input_tokens"] and first_ts in cdfs["input_tokens"][stat]:
        values, probs = cdfs["input_tokens"][stat][first_ts]
        print(f"  {stat.upper()}:")
        print(f"    Values: {values}")
        print(f"    Probabilities: {probs}")
```

## Data Structure

The framework comes with data organized as follows:

```
data/
├── language/
│   ├── m-large/
│   │   ├── chunk-1-dataset.json
│   │   ├── chunk-1-trace.csv
│   │   ├── chunk-2-dataset.json
│   │   └── chunk-2-trace.csv
│   ├── m-mid/
│   │   └── ...
│   ├── m-small/
│   │   └── ...
├── reason/
│   ├── deepseek-r1/
│   │   ├── chunk-1-dataset.json
│   │   └── chunk-1-trace.csv
└── multimodal/
    ├── mm-image/
    │   ├── chunk-1-dataset.json
    │   └── chunk-1-trace.csv
```

Each category (LANGUAGE, REASON, MULTIMODAL) contains model-specific data with:
- `chunk-i-dataset.json`: Request data distributions
- `chunk-i-trace.csv`: Rate and arrival pattern information


## Citation

If you find our work helpful, feel free to give us a cite.

```txt
@misc{servegen,
    title={ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production}, 
    author={Yuxing Xiang and Xue Li and Kun Qian and Wenyuan Yu and Ennan Zhai and Xin Jin},
    year={2025},
    eprint={2505.09999},
    archivePrefix={arXiv},
    primaryClass={cs.DC},
    url={https://arxiv.org/abs/2505.09999}, 
}
```
