import pytest
import numpy as np
import json
from servegen.workload_types import Category, ArrivalPat, ClientWindow
from servegen.clientpool import ClientPool, Client
from servegen.construct import generate_workload, Request

@pytest.fixture
def mock_client_pool():
    """Create a mock client pool with test data."""
    # Create individual clients
    client1 = Client(
        client_id=1,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
            1200: {"rate": 3.0, "cv": 1.5, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.5, 0.5],  # avg=0.5, p50=0, p95=1, p99=1
                "output_tokens": [0.0, 0.5, 0.5],  # avg=1.5, p50=1, p95=2, p99=2
            },
            600: {
                "input_tokens": [0.0, 0.5, 0.5],  # avg=1.5, p50=1, p95=2, p99=2
                "output_tokens": [0.0, 0.0, 0.5, 0.5],  # avg=2.5, p50=2, p95=3, p99=3
            },
        }
    )
    
    client2 = Client(
        client_id=2,
        trace={
            0: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 3.0, "cv": 1.5, "pat": ("Gamma", (1.0, 1.0))},
            1200: {"rate": 4.0, "cv": 2.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],  # avg=1.5, p50=1, p95=2, p99=2
                "output_tokens": [0.0, 0.0, 0.5, 0.5],  # avg=2.5, p50=2, p95=3, p99=3
            },
            600: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],  # avg=2.5, p50=2, p95=3, p99=3
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],  # avg=3.5, p50=3, p95=4, p99=4
            },
        }
    )
    
    return ClientPool.from_clients(Category.LANGUAGE, "test", [client1, client2])

@pytest.fixture
def real_client_pool():
    """Create a client pool with real data."""
    pool = ClientPool(Category.LANGUAGE, "m-large")
    return pool

def test_client_validation():
    """Test client validation."""
    # Test valid client
    valid_client = Client(
        client_id=1,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            600: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    valid_client.validate()  # Should not raise

    # Test unsorted timestamps
    with pytest.raises(ValueError, match="Trace timestamps must be in ascending order"):
        Client(
            client_id=1,
            trace={
                600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
                0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            },
            dataset={0: {"input_tokens": [0.0, 1.0]}}
        ).validate()

    # Test invalid trace data
    with pytest.raises(ValueError, match="Trace data at 0 must be a dictionary"):
        Client(
            client_id=1,
            trace={0: []},
            dataset={0: {"input_tokens": [0.0, 1.0]}}
        ).validate()

    # Test missing trace fields
    with pytest.raises(ValueError, match="Trace data at 0 missing required fields"):
        Client(
            client_id=1,
            trace={0: {"rate": 1.0}},  # Missing cv and pat
            dataset={0: {"input_tokens": [0.0, 1.0]}}
        ).validate()

    # Test invalid PDF
    with pytest.raises(ValueError, match="PDF for input_tokens at 0 must sum to 1.0"):
        Client(
            client_id=1,
            trace={0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))}},
            dataset={0: {"input_tokens": [0.0, 0.5]}}  # Sums to 0.5
        ).validate()

def test_client_pool_validation():
    """Test client pool validation."""
    # Test different trace timestamps between clients
    client1 = Client(
        client_id=1,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            600: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    client2 = Client(
        client_id=2,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            1200: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},  # Different timestamp
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            1200: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    with pytest.raises(ValueError, match="Client 2 has different trace timestamps than the first client"):
        ClientPool(Category.LANGUAGE, "m-large", clients={1: client1, 2: client2}).validate()

    # Test dataset timestamp not in trace timestamps
    client = Client(
        client_id=1,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            1200: {  # Extra timestamp not in trace
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    with pytest.raises(ValueError, match="Client 1 has dataset timestamps that are not in trace timestamps"):
        ClientPool(Category.LANGUAGE, "m-large", clients={1: client}).validate()

def test_client_pool_from_clients():
    """Test client pool creation from list of clients."""
    # Test duplicate client IDs
    client1 = Client(
        client_id=1,
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            600: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    client2 = Client(
        client_id=1,  # Same ID as client1
        trace={
            0: {"rate": 1.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 2.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.5, 0.5],
            },
            600: {
                "input_tokens": [0.0, 0.0, 0.5, 0.5],
                "output_tokens": [0.0, 0.0, 0.0, 0.5, 0.5],
            },
        }
    )
    with pytest.raises(ValueError, match="Duplicate client ID: 1"):
        ClientPool.from_clients(Category.LANGUAGE, "m-large", [client1, client2])

def test_span_basic(mock_client_pool):
    """Test basic span functionality."""
    # Test on ClientPool
    view = mock_client_pool.span(0, 600)
    windows = view.get()
    assert len(windows) == 2  # 2 clients, 1 window each
    assert all(w.timestamp == 0 for w in windows)
    
    # Test on ClientPoolView
    view2 = view.span(0, 300)
    windows = view2.get()
    assert len(windows) == 2
    assert all(w.timestamp == 0 for w in windows)
    assert all(w.window_size == 300 for w in windows)

def test_span_overlap(mock_client_pool):
    """Test span with overlapping windows."""
    view = mock_client_pool.span(300, 900)
    windows = view.get()
    assert len(windows) == 4  # 2 clients, 2 windows each
    timestamps = sorted(set(w.timestamp for w in windows))
    assert timestamps == [0, 300]

def test_filter_by_cv(mock_client_pool):
    """Test filtering by coefficient of variation."""
    # Test on ClientPool
    view = mock_client_pool.filter_by_cv(0.49, 0.51)
    windows = view.get()
    assert len(windows) == 1  # Only client 1's first window has CV=0.5
    
    # Test on ClientPoolView
    view2 = mock_client_pool.span(0, 600).filter_by_cv(0.49, 0.51)
    windows = view2.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].cv == 0.5

def test_filter_by_avg_input_len(mock_client_pool):
    """Test filtering by average input length."""
    # Test on ClientPool
    view = mock_client_pool.filter_by_avg_input_len(0.0, 1.0)
    windows = view.get()
    assert len(windows) == 1  # Only client 1's first window has avg=0.5
    
    # Test on ClientPoolView
    view2 = mock_client_pool.span(0, 600).filter_by_avg_input_len(0.0, 1.0)
    windows = view2.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].timestamp == 0

def test_filter_by_avg_output_len(mock_client_pool):
    """Test filtering by average output length."""
    # Test on ClientPool
    view = mock_client_pool.filter_by_avg_output_len(1.0, 2.0)
    windows = view.get()
    assert len(windows) == 1  # Only client 1's first window has avg=1.5
    
    # Test on ClientPoolView
    view2 = mock_client_pool.span(0, 600).filter_by_avg_output_len(1.0, 2.0)
    windows = view2.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].timestamp == 0

def test_filter_by_max_input_len(mock_client_pool):
    """Test filtering by maximum input length."""
    # Test on ClientPool
    view = mock_client_pool.filter_by_max_input_len(1)
    windows = view.get()
    assert len(windows) == 1  # Only client 1's first window has max<=1
    
    # Test on ClientPoolView
    view2 = mock_client_pool.span(0, 600).filter_by_max_input_len(1)
    windows = view2.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].timestamp == 0

def test_filter_by_max_output_len(mock_client_pool):
    """Test filtering by maximum output length."""
    # Test on ClientPool
    view = mock_client_pool.filter_by_max_output_len(2)
    windows = view.get()
    assert len(windows) == 1  # Only client 1's first window has max<=2
    
    # Test on ClientPoolView
    view2 = mock_client_pool.span(0, 600).filter_by_max_output_len(2)
    windows = view2.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].timestamp == 0

def test_filter_chaining(mock_client_pool):
    """Test chaining multiple filters."""
    view = mock_client_pool.filter_by_cv(0.5, 1.5).filter_by_avg_input_len(0.0, 1.0)
    windows = view.get()
    assert len(windows) == 1
    assert windows[0].client_id == 1
    assert windows[0].timestamp == 0
    assert windows[0].cv == 0.5

def test_get_cdfs_basic(mock_client_pool):
    """Test basic CDF computation."""
    cdfs = mock_client_pool.get_cdfs()
    
    # Check rate CDFs (unweighted)
    assert "rate" in cdfs
    assert 0 in cdfs["rate"]
    values, probs = cdfs["rate"][0]
    assert len(values) == 2  # 2 clients
    assert np.allclose(values, [1.0, 2.0]), values 
    assert np.allclose(probs, [0.5, 1.0]), probs  # Equal weights
    
    # Check CV CDFs (weighted by rate)
    assert "cv" in cdfs
    assert 0 in cdfs["cv"]
    values, probs = cdfs["cv"][0]
    assert len(values) == 2
    assert np.allclose(values, [0.5, 1.0]), values
    # Client 1 has rate 1.0, Client 2 has rate 2.0, so weights should be 1/3 and 2/3
    assert np.allclose(probs, [1/3, 1.0]), probs

def test_get_cdfs_dataset_stats(mock_client_pool):
    """Test CDF computation for dataset statistics."""
    cdfs = mock_client_pool.get_cdfs()
    
    # Check input_tokens statistics (weighted by rate)
    assert "input_tokens" in cdfs
    assert "avg" in cdfs["input_tokens"]
    assert 0 in cdfs["input_tokens"]["avg"]
    values, probs = cdfs["input_tokens"]["avg"][0]
    assert len(values) == 2
    assert np.allclose(values, [0.5, 1.5])
    # Client 1 has rate 1.0, Client 2 has rate 2.0, so weights should be 1/3 and 2/3
    assert np.allclose(probs, [1/3, 1.0])
    
    # Check p95 (weighted by rate)
    assert "p95" in cdfs["input_tokens"]
    assert 0 in cdfs["input_tokens"]["p95"]
    values, probs = cdfs["input_tokens"]["p95"][0]
    assert len(values) == 2
    assert np.allclose(values, [1, 2])
    assert np.allclose(probs, [1/3, 1.0])

def test_get_cdfs_with_filters(mock_client_pool):
    """Test CDF computation with filters applied."""
    cdfs = mock_client_pool.filter_by_cv(0.49, 0.51).get_cdfs()
    
    # Should only include client 1's first window
    assert "rate" in cdfs
    assert 0 in cdfs["rate"]
    values, probs = cdfs["rate"][0]
    assert len(values) == 1
    assert np.allclose(values, [1.0])
    assert np.allclose(probs, [1.0])

def test_get_cdfs_with_span(mock_client_pool):
    """Test CDF computation with time span applied."""
    cdfs = mock_client_pool.span(0, 600).get_cdfs()
    
    # Should only include first window of each client
    assert "rate" in cdfs
    assert 0 in cdfs["rate"]
    values, probs = cdfs["rate"][0]
    assert len(values) == 2
    assert np.allclose(values, [1.0, 2.0])
    assert np.allclose(probs, [0.5, 1.0])
    
    # Should not include later windows
    assert 600 not in cdfs["rate"]

def test_real_data_loading(real_client_pool):
    """Test loading and validation of real data."""
    # Basic validation
    assert len(real_client_pool.clients) > 0
    real_client_pool.validate()  # This will raise if there are any validation errors

def test_real_data_filtering(real_client_pool):
    """Test filtering functionality with real data."""
    # Test various filters
    view = real_client_pool.filter_by_cv(0.5, 1.5)
    windows = view.get()
    assert len(windows) > 0
    assert all(0.5 <= w.cv <= 1.5 for w in windows if w.cv is not None)
    
    view = real_client_pool.filter_by_avg_input_len(100, 1000)
    windows = view.get()
    assert len(windows) > 0
    for window in windows:
        if window.dataset and 'input_tokens' in window.dataset:
            pdf = window.dataset['input_tokens']
            avg_len = sum(i * p for i, p in enumerate(pdf, 0))
            assert 100 <= avg_len <= 1000

def test_real_data_span(real_client_pool):
    """Test time span functionality with real data."""
    # Get all timestamps
    all_timestamps = set()
    for client in real_client_pool.clients.values():
        all_timestamps.update(client.trace.keys())
    timestamps = sorted(all_timestamps)
    
    # Test a span that covers the first hour
    start_time = timestamps[0]
    end_time = start_time + 3600
    view = real_client_pool.span(start_time, end_time)
    windows = view.get()
    
    assert len(windows) > 0
    assert all(start_time <= w.timestamp < end_time for w in windows)
    assert all(w.window_size <= end_time - w.timestamp for w in windows) 