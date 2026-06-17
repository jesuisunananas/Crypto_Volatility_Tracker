import asyncio
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from collections import deque
import websockets
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
import torch.nn as nn
from model import VolatilityPredictor
import requests
import random
import time

model = VolatilityPredictor(input_size=2, hidden_size=16, num_layers=2)
model.train()
VOLATILITY_THRESHOLD = 0.002
BATCH_SIZE = 8
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

WEBSOCKET_MESSAGES_TOTAL = Counter(
    "volatility_websocket_messages_total", 
    "Total raw JSON frames received from the WebSocket stream"
)
CANDLES_PROCESSED_TOTAL = Counter(
    "volatility_candles_processed_total", 
    "Total number of finalized 1-minute candle intervals completed"
)
TRAINING_LOSS = Gauge(
    "volatility_model_training_loss", 
    "Current binary cross-entropy loss value from the latest online optimization step"
)
PREDICTED_PROBABILITY = Gauge(
    "volatility_prediction_probability", 
    "The model's inferred probability (0.0 to 1.0) that a volatility spike is imminent"
)
REPLAY_BUFFER_SIZE = Gauge(
    "volatility_replay_buffer_size", 
    "Current total element count stored inside the optimization experience replay buffer"
)
REALIZED_OUTCOMES_TOTAL = Counter(
    "volatility_realized_outcomes_total", 
    "Count of actual market states observed labeled by classification",
    ["class_label"]
)
INFERENCE_LATENCY_SECONDS = Histogram(
    "volatility_inference_latency_seconds", 
    "Time taken to execute a forward evaluation pass on the live sequence window"
)
OPTIMIZATION_LATENCY_SECONDS = Histogram(
    "volatility_optimization_latency_seconds", 
    "Time taken to run backpropagation and step the optimizer weights"
)


def initialize_scaler(symbol="BTCUSDT", limit=1000):
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&limit={limit}"
    response = requests.get(url).json()
    prev_features = []
    
    for kline in response:
        price_high = float(kline[2])
        price_low = float(kline[3])
        price_close = float(kline[4])
        volume = float(kline[5])
        prev_features.append([(price_high - price_low)/price_close, volume])

    arr = np.array(prev_features)
    scaler = StandardScaler()
    scaler.fit(arr)
    return scaler

async def stream_data(scaler=None, k=100):
    if not scaler:
        print("Error: Scaler must be initialized first with initalize_scaler()")
        return
    
    start_http_server(port=5000, addr="0.0.0.0")
    print("Prometheus metrics exporter running on http://localhost:5000/metrics")

    uri = "wss://stream.binance.us:9443/ws/btcusdt@kline_1m"
    window = deque(maxlen=k)
    pending_sample = None
    replay_buffer = deque(maxlen=200)

    async with websockets.connect(uri, ssl=True) as websocket:
        print("Connected to Binance Stream")
        try:
            async for message in websocket:
                WEBSOCKET_MESSAGES_TOTAL.inc()

                data = json.loads(message)
                kline = data['k']
                price_high = float(kline['h'])
                price_low = float(kline['l'])
                price_close = float(kline['c'])
                volume = float(kline['v'])
                is_closed = kline['x']
                candle_start_time = kline['t']

                if is_closed and volume > 0.0:
                    CANDLES_PROCESSED_TOTAL.inc()
                    realized_spread = (price_high - price_low) / price_close
                    if pending_sample is not None:
                        is_spike = realized_spread > VOLATILITY_THRESHOLD
                        target_label = 1.0 if is_spike else 0.0
                        label_str = "spike" if is_spike else "normal"
                        REALIZED_OUTCOMES_TOTAL.labels(class_label=label_str).inc()

                        replay_buffer.append((pending_sample['input'], target_label))
                        REPLAY_BUFFER_SIZE.set(len(replay_buffer))

                        if len(replay_buffer) >= BATCH_SIZE:
                            start_opt = time.perf_counter()
                            batch = random.sample(replay_buffer, BATCH_SIZE)
                            torch_inputs = torch.cat([item[0] for item in batch], dim=0)
                            batch_targets = [[item[1]] for item in batch]
                            torch_targets = torch.tensor(batch_targets, dtype=torch.float32)
                            model.train()
                            optimizer.zero_grad()
                            predictions = model(torch_inputs)
                            loss = criterion(predictions, torch_targets)
                            loss.backward()
                            optimizer.step()
                            OPTIMIZATION_LATENCY_SECONDS.observe(time.perf_counter() - start_opt)
                            TRAINING_LOSS.set(loss.item())
                            print(f"--- [Online Batch Optimization] Loss: {loss.item():.6f} | Buffer Size: {len(replay_buffer)} ---")
                        pending_sample = None

                    print(f"Candle Closed | Close: ${price_close} | High: ${price_high} | Low: ${price_low} | Vol: {volume}")
                    window.append([realized_spread, volume])
                    if len(window) >= k:
                        start_inf = time.perf_counter()
                        arr = np.array(window)
                        input_tensor = torch.from_numpy(scaler.transform(arr)).unsqueeze(0).float()
                        model.eval()
                        with torch.no_grad():
                            volatility_probability = model(input_tensor)
                        INFERENCE_LATENCY_SECONDS.observe(time.perf_counter() - start_inf)
                        PREDICTED_PROBABILITY.set(volatility_probability.item())
                        print(f"Predicted Volatility Probability: {volatility_probability.item():.4f}")
                        pending_sample = {
                            "input": input_tensor
                        }
                else:
                    print(f"Live price: ${price_close:.2f}", end='\r')

        except websockets.ConnectionClosed:
            print("Connection to Binance Stream closed")

        except KeyboardInterrupt:
            print("User cancelled execution")

if __name__ == "__main__":
    scaler = initialize_scaler()
    asyncio.run(stream_data(scaler))