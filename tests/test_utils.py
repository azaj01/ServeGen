import pytest
import numpy as np
import os
import tempfile

from servegen.clientpool import ClientPool, Client
from servegen.workload_types import Category
from servegen.construct import Request
from servegen.utils import (
    get_constant_rate_fn,
    get_scaled_rate_fn,
    get_bounded_rate_fn,
    sample_from_cdf,
    save_requests_to_csv,
)

# Helper fixture to create a mock ClientPool for testing
@pytest.fixture
def mock_client_pool():
    client1 = Client(
        client_id=1,
        trace={
            0: {"rate": 5.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 7.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={},
    )

    client2 = Client(
        client_id=2,
        trace={
            0: {"rate": 7.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 5.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={},
    )

    return ClientPool.from_clients(Category.LANGUAGE, "test", [client1, client2])


def test_get_constant_rate_fn(mock_client_pool):
    """Test get_constant_rate_fn produces correct structure."""
    rate_fn = get_constant_rate_fn(mock_client_pool, target_rate=10.0)
    assert isinstance(rate_fn, dict)
    assert set(rate_fn.keys()) == {0, 600}
    assert all(v == 10.0 for v in rate_fn.values())


def test_get_scaled_rate_fn(mock_client_pool):
    """Test get_scaled_rate_fn scales correctly."""
    scale_factor = 2.0
    rate_fn = get_scaled_rate_fn(mock_client_pool, scale_factor)

    expected_total_0 = (5.0 + 7.0) * scale_factor  # 12 * 2 = 24
    expected_total_600 = (7.0 + 5.0) * scale_factor  # 12 * 2 = 24

    assert rate_fn[0] == pytest.approx(expected_total_0)
    assert rate_fn[600] == pytest.approx(expected_total_600)


def test_get_bounded_rate_fn_no_scaling_needed(mock_client_pool):
    """Test get_bounded_rate_fn when no scaling is needed."""
    max_rate = 50.0
    rate_fn = get_bounded_rate_fn(mock_client_pool, max_rate)
    # Original total rate at each timestamp is 12, so no scaling should occur
    assert rate_fn[0] == pytest.approx(12.0)
    assert rate_fn[600] == pytest.approx(12.0)


def test_get_bounded_rate_fn_scaling_applied(mock_client_pool):
    """Test get_bounded_rate_fn applies scaling when needed."""
    max_rate = 6.0
    rate_fn = get_bounded_rate_fn(mock_client_pool, max_rate)
    # Original rate is 12, so scale factor = 6 / 12 = 0.5
    assert rate_fn[0] == pytest.approx(6.0)
    assert rate_fn[600] == pytest.approx(6.0)


def test_sample_from_cdf():
    """Test sampling from a CDF."""
    values = np.array([1, 2, 3, 4])
    probs = np.array([0.1, 0.3, 0.6, 1.0])  # Cumulative probabilities
    cdf = (values, probs)

    # Fix seed for reproducibility
    rng = np.random.RandomState(42)
    samples = sample_from_cdf(cdf, n_samples=1000, rng=rng)

    assert len(samples) == 1000
    assert all(s in values for s in samples)

    # Check distribution roughly matches CDF (simple check)
    counts = np.bincount(samples)[1:]  # Skip index 0 since values start at 1
    freqs = counts / len(samples)
    expected_freqs = [0.1, 0.2, 0.3, 0.4]
    for f, ef in zip(freqs, expected_freqs):
        assert abs(f - ef) < 0.05  # Allow some tolerance due to randomness


def test_sample_from_cdf_empty():
    """Test sample_from_cdf with empty input."""
    cdf = (np.array([]), np.array([]))
    samples = sample_from_cdf(cdf, n_samples=5)
    assert len(samples) == 0


def test_save_requests_to_csv():
    """Test saving requests to CSV file."""
    requests = [
        Request(
            request_id=0, timestamp=0.5, data={"input_tokens": 10, "output_tokens": 20}
        ),
        Request(
            request_id=1, timestamp=1.2, data={"input_tokens": 15, "output_tokens": 25}
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_requests.csv")
        save_requests_to_csv(requests, filepath)

        # Read back and verify contents
        with open(filepath, "r") as f:
            lines = f.readlines()
            header = lines[0].strip().split(",")
            assert header == [
                "request_id",
                "timestamp",
                "input_tokens",
                "output_tokens",
            ]

            row1 = lines[1].strip().split(",")
            assert row1 == ["0", "0.5", "10", "20"]

            row2 = lines[2].strip().split(",")
            assert row2 == ["1", "1.2", "15", "25"]


def test_save_requests_to_csv_empty():
    """Test that saving empty list raises error."""
    with pytest.raises(ValueError, match="No requests to save"):
        save_requests_to_csv([], "dummy.csv")
