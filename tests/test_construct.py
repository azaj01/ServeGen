import pytest
import numpy as np
from servegen.workload_types import Category, ArrivalPat, ClientWindow
from servegen.clientpool import ClientPool, Client
from servegen.construct import generate_workload, Request
from servegen.utils import get_constant_rate_fn

@pytest.fixture
def mock_client_pool():
    """Create a mock client pool with test data."""
    client1 = Client(
        client_id=1,
        trace={
            0: {"rate": 5.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 7.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.5, 0.5],  # avg=0.5
                "output_tokens": [0.0, 0.5, 0.5],  # avg=1.5
            },
            600: {
                "input_tokens": [0.0, 0.5, 0.5],  # avg=1.5
                "output_tokens": [0.0, 0.0, 0.5, 0.5],  # avg=2.5
            },
        }
    )
    
    client2 = Client(
        client_id=2,
        trace={
            0: {"rate": 7.0, "cv": 1.0, "pat": ("Gamma", (1.0, 1.0))},
            600: {"rate": 5.0, "cv": 0.5, "pat": ("Gamma", (1.0, 1.0))},
        },
        dataset={
            0: {
                "input_tokens": [0.0, 0.5, 0.5],  # avg=1.5
                "output_tokens": [0.0, 0.0, 0.5, 0.5],  # avg=2.5
            },
            600: {
                "input_tokens": [0.5, 0.5],  # avg=0.5
                "output_tokens": [0.0, 0.5, 0.5],  # avg=1.5
            },
        }
    )
    
    return ClientPool.from_clients(Category.LANGUAGE, "test", [client1, client2])

@pytest.fixture
def language_client_pool():
    """Create a client pool with real LANGUAGE category data."""
    return ClientPool(Category.LANGUAGE, "m-large")

@pytest.fixture
def reason_client_pool():
    """Create a client pool with real REASON category data."""
    return ClientPool(Category.REASON, "deepseek-r1")

@pytest.fixture
def multimodal_client_pool():
    """Create a client pool with real MULTIMODAL category data."""
    return ClientPool(Category.MULTIMODAL, "mm-image")

def test_generate_workload_basic(mock_client_pool):
    """Test basic workload generation."""
    # Generate workload with target rate of 8.0 for both windows
    rate_fn = {0: 8.0, 600: 8.0}
    requests = generate_workload(mock_client_pool, rate_fn, duration=1200)
    
    # Check that we got some requests
    assert len(requests) > 0
    
    # Check that requests are sorted by timestamp
    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    
    # Check that all timestamps are within the expected range
    assert all(0 <= t < 1200 for t in timestamps)
    
    # Check that all requests have the expected fields
    for req in requests:
        assert isinstance(req, Request)
        assert "input_tokens" in req.data
        assert "output_tokens" in req.data

def test_generate_workload_rate_scaling(mock_client_pool):
    """Test that request rates are properly scaled."""
    # Generate workload with different target rates
    rate_fn = {0: 8.0, 600: 12.0}
    requests = generate_workload(mock_client_pool, rate_fn, duration=1200)
    
    # Group requests by window
    window1_requests = [r for r in requests if 0 <= r.timestamp < 600]
    window2_requests = [r for r in requests if 600 <= r.timestamp < 1200]
    
    # Calculate actual rates
    window1_rate = len(window1_requests) / 600
    window2_rate = len(window2_requests) / 600
    
    # Check that rates are close to target (within 20% to account for randomness)
    assert abs(window1_rate - 8.0) / 8.0 < 0.2
    assert abs(window2_rate - 12.0) / 12.0 < 0.2

def test_generate_workload_invalid_rate_fn(mock_client_pool):
    """Test that invalid rate functions are rejected."""
    # Missing timestamp
    with pytest.raises(ValueError, match="Rate function timestamps don't match pool timestamps"):
        generate_workload(mock_client_pool, {0: 8.0}, duration=1200)
    
    # Wrong timestamp
    with pytest.raises(ValueError, match="Rate function timestamps don't match pool timestamps"):
        generate_workload(mock_client_pool, {0: 8.0, 300: 8.0}, duration=1200)

def test_generate_workload_with_view(mock_client_pool):
    """Test workload generation with a ClientPoolView."""
    # Create a view for the first window only
    view = mock_client_pool.span(0, 600)
    rate_fn = {0: 8.0}
    requests = generate_workload(view, rate_fn, duration=600)
    
    # Check that all requests are in the first window
    assert all(0 <= r.timestamp < 600 for r in requests)

def test_generate_workload_with_view_and_duration(mock_client_pool):
    """Test workload generation with a ClientPoolView and duration parameter."""
    # Create a view for the first window only
    view = mock_client_pool.span(0, 600)
    rate_fn = {0: 8.0}
    duration = 300  # 5 minutes
    
    # Test with duration
    requests = generate_workload(view, rate_fn, duration=duration)
    
    # Check that all requests are within the duration
    assert all(0 <= r.timestamp < duration for r in requests)
    
    # Check that the window is properly sized
    if requests:
        assert max(r.timestamp for r in requests) < duration

def test_language_workload_generation(language_client_pool):
    """Test workload generation with real LANGUAGE category data."""
    # Create a view for the first hour
    view = language_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    requests = generate_workload(view, rate_fn, duration=3600)
    
    # Check that we got some requests
    assert len(requests) > 0
    
    # Check that requests are sorted by timestamp
    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    
    # Check that all timestamps are within the expected range
    assert all(0 <= t < 3600 for t in timestamps)
    
    # Check that all requests have the expected fields
    for req in requests:
        assert isinstance(req, Request)
        assert "input_tokens" in req.data
        assert "output_tokens" in req.data
        assert req.data["input_tokens"] > 0
        assert req.data["output_tokens"] > 0

def test_language_workload_generation_with_duration(language_client_pool):
    """Test workload generation with real LANGUAGE category data and duration."""
    # Create a view for the first hour
    view = language_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    duration = 1500  # 25 minutes
    
    # Test with duration
    requests = generate_workload(view, rate_fn, duration=duration)
    
    # Check that all timestamps are within the duration
    assert all(0 <= t < duration for t in [r.timestamp for r in requests])
    
    # Check that the last window is properly sized
    last_window_requests = [r for r in requests if r.timestamp >= 1200]
    if last_window_requests:
        assert max(r.timestamp for r in last_window_requests) < duration

def test_reason_workload_generation(reason_client_pool):
    """Test workload generation for REASON category."""
    # Create a view for the first hour
    view = reason_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    requests = generate_workload(view, rate_fn, duration=3600)
    
    # Check that we got some requests
    assert len(requests) > 0
    
    # Check that requests are sorted by timestamp
    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    
    # Check that all timestamps are within the expected range
    assert all(0 <= t < 3600 for t in timestamps)
    
    # Check that all requests have the expected fields
    for req in requests:
        assert isinstance(req, Request)
        assert "input_tokens" in req.data
        assert "output_tokens" in req.data
        assert "reason_ratio" in req.data
        assert req.data["input_tokens"] > 0
        assert req.data["output_tokens"] > 0
        assert 0 <= req.data["reason_ratio"] <= 1  # reason_ratio should be in [0,1]

def test_reason_workload_rate_scaling(reason_client_pool):
    """Test that request rates are properly scaled for REASON category."""
    # Create a view for the first hour
    view = reason_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    requests = generate_workload(view, rate_fn, duration=3600)
    
    # Calculate actual rate
    actual_rate = len(requests) / 3600
    
    # Check that rate is close to target (within 20% to account for randomness)
    assert abs(actual_rate - 100.0) / 100.0 < 0.2

def test_multimodal_workload_generation(multimodal_client_pool):
    """Test workload generation with real MULTIMODAL category data."""
    # Create a view for the first hour
    view = multimodal_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    requests = generate_workload(view, rate_fn, duration=3600)
    
    # Check that we got some requests
    assert len(requests) > 0
    
    # Check that requests are sorted by timestamp
    timestamps = [r.timestamp for r in requests]
    assert timestamps == sorted(timestamps)
    
    # Check that all timestamps are within the expected range
    assert all(0 <= t < 3600 for t in timestamps)
    
    # Check that all requests have the expected fields
    for req in requests:
        assert isinstance(req, Request)
        assert "text_tokens" in req.data
        assert "output_tokens" in req.data
        assert "image_tokens" in req.data
        assert "audio_tokens" in req.data
        assert "video_tokens" in req.data
        
        # Check token counts are positive
        assert req.data["text_tokens"] > 0
        assert req.data["output_tokens"] > 0
        
        # Check that token lists are lists
        assert isinstance(req.data["image_tokens"], list)
        assert isinstance(req.data["audio_tokens"], list)
        assert isinstance(req.data["video_tokens"], list)
        
        # Check that all tokens in lists are positive
        assert all(t > 0 for t in req.data["image_tokens"])
        assert all(t > 0 for t in req.data["audio_tokens"])
        assert all(t > 0 for t in req.data["video_tokens"])

def test_multimodal_workload_rate_scaling(multimodal_client_pool):
    """Test that request rates are properly scaled for MULTIMODAL category."""
    # Create a view for the first hour
    view = multimodal_client_pool.span(0, 3600)
    rate_fn = get_constant_rate_fn(view, 100)
    requests = generate_workload(view, rate_fn, duration=3600)
    
    # Calculate actual rate
    actual_rate = len(requests) / 3600
    
    # Check that rate is close to target (within 20% to account for randomness)
    assert abs(actual_rate - 100.0) / 100.0 < 0.2
