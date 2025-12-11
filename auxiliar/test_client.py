import asyncio
import json
import websockets

async def main():
    uri = "ws://localhost:8001/traffic_stream"
    async with websockets.connect(uri, max_size=None) as ws:
        print("[test] Connected, sending start")
        await ws.send(json.dumps({"type": "start", "speed": 0.5}))
        i = 0
        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                print("[test] got", data.get("type"), data.get("index"), data.get("timestamp"))
                i += 1
                if i >= 5:
                    break
        except Exception as e:
            print("[test] exception:", e)

asyncio.run(main())
