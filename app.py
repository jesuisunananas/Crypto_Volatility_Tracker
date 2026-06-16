import asyncio
from collections import deque
import websockets
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from model import VolatilityPredictor

scaler = StandardScaler()
model = VolatilityPredictor(input_size=2, hidden_size=16, num_layers=2)
model.eval()

async def stream_data(k=100):
    uri = "wss://stream.binance.us:9443/ws/btcusdt@kline_1m"
    window = deque(maxlen=k)
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

                if is_closed and volume > 0.0:
                    print(f"Candle Closed | Close: ${price_close} | High: ${price_high} | Low: ${price_low} | Vol: {volume}")
                    new_features = [price_high - price_low, volume]
                    window.append(new_features)
                    if len(window) >= k:
                        arr = np.array(window)
                        input_tensor = torch.from_numpy(scaler.fit_transform(arr)).unsqueeze(0).float()
                        with torch.no_grad():
                            volatility_probability = model(input_tensor)
                        print(f"Predicted Volatility Probability: {volatility_probability.item():.4f}")
                else:
                    print(f"Live price: ${price_close:.2f}", end='\r')

                print(json.dumps(data, indent=2))

        except websockets.ConnectionClosed:
            print("Connection to Binance Stream closed")

if __name__ == "__main__":
    asyncio.run(stream_data())