"""Aether Edge AI Engine - Python API Client Integration Example.

Demonstrates how external applications (web dashboards, Slack bots, Home Assistant,
or mobile backends) interact with Aether over REST and WebSockets in just a few lines.

Usage:
    python -m examples.api_client
"""
import asyncio
import httpx


API_BASE = "http://localhost:8000"


async def main():
    print("=" * 60)
    print("   Aether Engine Client Integration Demo")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=API_BASE, timeout=10.0) as client:
        # 1. Health & Hardware Status
        print("\n[1] Checking Engine Health...")
        try:
            res = await client.get("/api/health")
            res.raise_for_status()
            health = res.json()
            print(f"    Status: {health['status'].upper()}")
            print(f"    Enrolled People: {health['enrolled_identities']}")
            print(f"    Battery: {health['battery']}")
            print(f"    Hardware: {health['resources']}")
        except Exception as exc:
            print(f"    Failed to connect to Aether at {API_BASE}: {exc}")
            print("    Please make sure the server is running: python main.py --server")
            return

        # 2. Query Camera Presence
        print("\n[2] Checking Visual Presence...")
        res = await client.get("/api/presence")
        presence = res.json()
        print(f"    Visible Face Count: {presence['count']}")
        if presence["count"] > 0:
            for p in presence["people"]:
                print(f"    - Identity: {p['identity']} (Confidence: {p['confidence'] * 100:.0f}%, Verified: {p['is_verified']})")
        else:
            print("    No faces currently in front of the camera.")

        # 3. List Stored Identities
        print("\n[3] Listing Enrolled Identities...")
        res = await client.get("/api/identities")
        identities = res.json()
        print(f"    Total Registered: {identities['count']}")
        for id_info in identities["identities"]:
            print(f"    - {id_info['display']} ({id_info['name']})")

        # 4. Trigger Remote TTS
        print("\n[4] Triggering Voice Announcement...")
        await client.post("/api/speak", json={"text": "Aether API client demo executed successfully."})
        print("    Spoken command dispatched.")

        # 5. Add a Persistent Note
        print("\n[5] Saving a Persistent Note...")
        res = await client.post("/api/notes", json={"text": "Review commercial buyer documentation", "author": "API_Client"})
        note = res.json()
        print(f"    Note saved with ID #{note['id']}.")

        # 6. Stream Live Events over WebSocket
        print("\n[6] Connecting to Real-Time Event WebSocket for 5 seconds...")
        import websockets
        ws_url = "ws://localhost:8000/ws/events"
        try:
            async with websockets.connect(ws_url) as ws:
                print("    Connected to WebSocket. Listening for engine events...")
                # Trigger a command to observe the event
                await client.post("/api/command", json={"text": "status"})
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    print(f"    Received Bus Event: {msg}")
                except asyncio.TimeoutError:
                    print("    No events received during window.")
        except ImportError:
            print("    (Install `websockets` to run the live WS demo)")
        except Exception as exc:
            print(f"    WebSocket test: {exc}")

    print("\n" + "=" * 60)
    print("Demo complete! API documentation available at http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
