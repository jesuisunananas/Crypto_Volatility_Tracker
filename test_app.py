import pytest
import numpy as np
import json
from unittest.mock import patch, AsyncMock, MagicMock
from app import (
    initialize_scaler, 
    stream_data,
    WEBSOCKET_MESSAGES_TOTAL, 
    CANDLES_PROCESSED_TOTAL,
    TRAINING_LOSS, 
    PREDICTED_PROBABILITY, 
    REPLAY_BUFFER_SIZE,
    REALIZED_OUTCOMES_TOTAL, 
    INFERENCE_LATENCY_SECONDS, 
    OPTIMIZATION_LATENCY_SECONDS
)
from sklearn.preprocessing import StandardScaler
from prometheus_client import REGISTRY

def get_metric(name, labels=None):
    if labels:
        return REGISTRY.get_sample_value(name, labels)
    val = REGISTRY.get_sample_value(name)
    if val is None:
        val = REGISTRY.get_sample_value(f"{name}_total")
    return val or 0.0

@patch('app.requests.get')
def test_initialize_scaler(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = [
        [0, "100", "110", "90", "105", "50.0"],
        [0, "105", "115", "100", "110", "60.0"],
    ]
    mock_get.return_value = mock_response

    scaler = initialize_scaler(symbol="BTCUSDT", limit=2)
    
    assert mock_get.called
    assert isinstance(scaler, StandardScaler)
    assert scaler.n_features_in_ == 2
    assert scaler.n_samples_seen_ == 2

@pytest.mark.asyncio
@patch('builtins.print')
async def test_stream_data_no_scaler(mock_print):
    await stream_data(scaler=None)
    mock_print.assert_called_with("Error: Scaler must be initialized first with initalize_scaler()")

@pytest.mark.asyncio
@patch('app.websockets.connect')
@patch('app.start_http_server')
async def test_stream_data(mock_start_http, mock_connect):
    mock_websocket = AsyncMock()
    
    messages = [
        json.dumps({"k": {"h": "110", "l": "90", "c": "100", "v": "10.0", "x": False, "t": 0}}),
        json.dumps({"k": {"h": "110", "l": "90", "c": "100", "v": "10.0", "x": True, "t": 0}}),
    ]
    mock_websocket.__aiter__.return_value = messages
    mock_connect.return_value.__aenter__.return_value = mock_websocket

    scaler = StandardScaler()
    scaler.fit(np.array([[0.1, 10.0], [0.2, 20.0]]))
    
    await stream_data(scaler=scaler, k=2)
    
    mock_connect.assert_called_once_with("wss://stream.binance.us:9443/ws/btcusdt@kline_1m", ssl=True)

@pytest.mark.asyncio
@patch('app.websockets.connect')
@patch('app.start_http_server')
async def test_prometheus_server_startup(mock_start_http, mock_connect):
    mock_connect.return_value.__aenter__.return_value = AsyncMock()
    scaler = StandardScaler()
    scaler.fit(np.array([[0.1, 10.0], [0.2, 20.0]]))
    await stream_data(scaler=scaler, k=2)
    mock_start_http.assert_called_once_with(port=5000, addr="0.0.0.0")

@pytest.mark.asyncio
@patch('app.websockets.connect')
@patch('app.start_http_server')
async def test_metric_counters(mock_start_http, mock_connect):
    ws_before = get_metric("volatility_websocket_messages_total")
    candles_before = get_metric("volatility_candles_processed_total")
    
    mock_websocket = AsyncMock()
    messages = [
        json.dumps({"k": {"h": "110", "l": "90", "c": "100", "v": "10.0", "x": False, "t": 0}}),
        json.dumps({"k": {"h": "110", "l": "90", "c": "100", "v": "10.0", "x": True, "t": 0}}),
    ]
    mock_websocket.__aiter__.return_value = messages
    mock_connect.return_value.__aenter__.return_value = mock_websocket

    scaler = StandardScaler()
    scaler.fit(np.array([[0.1, 10.0], [0.2, 20.0]]))
    
    await stream_data(scaler=scaler, k=2)
    
    # 2 messages pushed
    assert get_metric("volatility_websocket_messages_total") == ws_before + 2
    # 1 candle closed
    assert get_metric("volatility_candles_processed_total") == candles_before + 1

@pytest.mark.asyncio
@patch('app.websockets.connect')
@patch('app.start_http_server')
@patch('app.json.loads')
async def test_graceful_shutdown(mock_loads, mock_start_http, mock_connect):
    mock_websocket = AsyncMock()
    mock_websocket.__aiter__.return_value = ['dummy_message']
    mock_connect.return_value.__aenter__.return_value = mock_websocket
    
    mock_loads.side_effect = KeyboardInterrupt()
    
    scaler = StandardScaler()
    scaler.fit(np.array([[0.1, 10.0], [0.2, 20.0]]))
    
    # stream_data should now propagate the KeyboardInterrupt to the caller
    with pytest.raises(KeyboardInterrupt):
        await stream_data(scaler=scaler, k=2)

@pytest.mark.asyncio
@patch('app.websockets.connect')
@patch('app.start_http_server')
async def test_optimization_metrics(mock_start_http, mock_connect):
    mock_websocket = AsyncMock()
    
    # 9 closed candles needed to populate pending_sample and then replay_buffer to size 8
    # Spread is (150-90)/100 = 0.6 (> 0.002 threshold), meaning labels will be "spike"
    messages = []
    for i in range(10):
        messages.append(json.dumps({"k": {"h": "150", "l": "90", "c": "100", "v": "10.0", "x": True, "t": i}}))
        
    mock_websocket.__aiter__.return_value = messages
    mock_connect.return_value.__aenter__.return_value = mock_websocket

    scaler = StandardScaler()
    scaler.fit(np.array([[0.1, 10.0], [0.2, 20.0]]))
    
    await stream_data(scaler=scaler, k=2)
    
    # The replay buffer should have hit 8 items
    assert get_metric("volatility_replay_buffer_size") >= 8
    
    # The training loss should have been updated
    assert get_metric("volatility_model_training_loss") > 0
    
    # The label "spike" should have incremented
    # Note: metrics that use labels need the labels dict explicitly to fetch
    spike_count = get_metric("volatility_realized_outcomes_total_total", {"class_label": "spike"}) or 0.0
    if spike_count == 0.0:
        spike_count = get_metric("volatility_realized_outcomes_total", {"class_label": "spike"}) or 0.0
    assert spike_count > 0
