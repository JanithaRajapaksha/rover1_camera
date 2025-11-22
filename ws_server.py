import asyncio
import websockets

clients = set()

async def handler(ws, path):
    clients.add(ws)
    try:
        async for msg in ws:
            print("Received:", msg)
    finally:
        clients.remove(ws)

async def main():
    async with websockets.serve(handler, "127.0.0.1", 5007):
        print("WS Server started at ws://127.0.0.1:5007")
        await asyncio.Future()

asyncio.run(main())
