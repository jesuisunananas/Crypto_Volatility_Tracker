import asyncio
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

model = VolatilityPredictor(input_size=2, hidden_size=16, num_layers=2)
model.train()
VOLATILITY_THRESHOLD = 0.002
BATCH_SIZE = 8
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

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

    uri = "wss://stream.binance.us:9443/ws/btcusdt@kline_1m"
    window = deque(maxlen=k)
    pending_sample = None
    replay_buffer = deque(maxlen=200)

    async with websockets.connect(uri, ssl=True) as websocket:
        print("Connected to Binance Stream")
        try:
            async for message in websocket:
                data = json.loads(message)
                kline = data['k']
                price_high = float(kline['h'])
                price_low = float(kline['l'])
                price_close = float(kline['c'])
                volume = float(kline['v'])
                is_closed = kline['x']
                candle_start_time = kline['t']

                if is_closed and volume > 0.0:
                    realized_spread = (price_high - price_low) / price_close
                    if pending_sample is not None:
                        target_label = 1.0 if realized_spread > VOLATILITY_THRESHOLD else 0.0

                        replay_buffer.append((pending_sample['input'], target_label))

                        if len(replay_buffer) >= BATCH_SIZE:
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
                            print(f"--- [Online Batch Optimization] Loss: {loss.item():.6f} | Buffer Size: {len(replay_buffer)} ---")
                        pending_sample = None

                    print(f"Candle Closed | Close: ${price_close} | High: ${price_high} | Low: ${price_low} | Vol: {volume}")
                    window.append([realized_spread, volume])
                    if len(window) >= k:
                        arr = np.array(window)
                        input_tensor = torch.from_numpy(scaler.transform(arr)).unsqueeze(0).float()
                        model.eval()
                        with torch.no_grad():
                            volatility_probability = model(input_tensor)
                        print(f"Predicted Volatility Probability: {volatility_probability.item():.4f}")
                        pending_sample = {
                            "input": input_tensor
                        }
                else:
                    print(f"Live price: ${price_close:.2f}", end='\r')

                print(json.dumps(data, indent=2))

        except websockets.ConnectionClosed:
            print("Connection to Binance Stream closed")

if __name__ == "__main__":
    scaler = initialize_scaler()
    asyncio.run(stream_data(scaler))