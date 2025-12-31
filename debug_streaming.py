"""
Debug script to test streaming at each layer
"""
import requests
import json
import time

API_URL = "http://localhost:8001"

print("="*80)
print("STREAMING DEBUG TEST")
print("="*80)

# Test 1: Check API health
print("\n1. Testing API health...")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"✅ API is running: {response.json()}")
except Exception as e:
    print(f"❌ API not reachable: {e}")
    exit(1)

# Test 2: Initialize if needed
print("\n2. Checking initialization...")
health = requests.get(f"{API_URL}/health").json()
if not health.get('initialized'):
    print("Initializing...")
    init_resp = requests.post(f"{API_URL}/initialize")
    print(f"✅ Initialized: {init_resp.json()}")

# Test 3: Test streaming endpoint directly
print("\n3. Testing streaming endpoint...")
print("URL:", f"{API_URL}/query/database/stream")

payload = {
    "query": "Hello can you assist me ? ",
    "chat_history": []
}

print("Payload:", json.dumps(payload, indent=2))

try:
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/query/database/stream",
        json=payload,
        stream=True,
        timeout=120
    )

    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print("\nStreaming events:")
    print("-" * 80)

    chunk_count = 0

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            print(f"[RAW] {line_str[:100]}...")  # Show first 100 chars

            if line_str.startswith('data: '):
                data_json = line_str[6:]
                try:
                    event = json.loads(data_json)
                    event_type = event.get("type")

                    print(f"[EVENT] Type: {event_type}")

                    if event_type == "chunk":
                        chunk_count += 1
                        content = event.get("content", "")
                        print(f"  Chunk #{chunk_count}: '{content}'")

                    elif event_type == "metadata":
                        elapsed = event.get("elapsed_time", 0)
                        print(f"  ✅ Generation completed in {elapsed:.2f}s")
                        print(f"  Total chunks received: {chunk_count}")

                    elif event_type == "sql":
                        print(f"  SQL: {event.get('content', '')[:50]}...")

                    elif event_type == "results":
                        results = event.get('content', [])
                        print(f"  Results: {len(results)} rows")

                    elif event_type == "error":
                        print(f"  ❌ Error: {event.get('content')}")

                except json.JSONDecodeError as e:
                    print(f"  [JSON ERROR] {e}")

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"\n✅ Stream completed in {total_time:.2f}s")
    print(f"Total chunks: {chunk_count}")

    if chunk_count == 0:
        print("\n⚠️  WARNING: No chunks received! Streaming may not be working.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
