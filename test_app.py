import pytest
import numpy as np
import json
from unittest.mock import patch, AsyncMock, MagicMock
from app import initialize_scaler, stream_data
from sklearn.preprocessing import StandardScaler

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
async def test_stream_data(mock_connect):
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
