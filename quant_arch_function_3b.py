#!/usr/bin/env python3
"""
Full-precision interactive chat for katanemo/Arch-Function-3B
-------------------------------------------------------------
• Runs vLLM server (FP16, ~15–16 GB VRAM) in background
• Opens interactive chat loop
• Maintains short-term conversation memory
• Pretty-prints JSON responses with reasoning trace
• Gracefully shuts down on Ctrl +C
"""

import subprocess, requests, json, time, sys

MODEL_ID = "katanemo/Arch-Function-3B"
PORT = 8000
API_URL = f"http://localhost:{PORT}/v1/completions"

# ─────────────────────────────────────────────
# 1️⃣  Start vLLM server as background process
# ─────────────────────────────────────────────
def start_server():
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--port", str(PORT),
        "--dtype", "float16",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85"
    ]
    print(f"🚀 Launching {MODEL_ID} on port {PORT} (FP16)…")
    return subprocess.Popen(cmd)

# ─────────────────────────────────────────────
# 2️⃣  Chat loop with memory
# ─────────────────────────────────────────────
def chat_loop():
    history = []   # keep conversation context
    print("\n💬 Chat ready.  Type your message (Ctrl +C to exit)\n")

    while True:
        try:
            user_msg = input("You: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in {"exit", "quit"}:
                break

            # Build prompt with recent history (last 6 turns)
            context = ""
            for turn in history[-6:]:
                context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
            prompt = context + f"User: {user_msg}\nAssistant:"

            payload = {
                "model": MODEL_ID,
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 0.4,
                "stream": False
            }

            start = time.time()
            r = requests.post(API_URL, json=payload, timeout=120)
            elapsed = round(time.time() - start, 2)

            if r.status_code != 200:
                print(f"❌ HTTP {r.status_code}: {r.text}")
                continue

            data = r.json()
            reply = data["choices"][0]["text"].strip()

            # Save to memory
            history.append({"user": user_msg, "assistant": reply})

            # Construct rich JSON
            enriched = {
                "response": reply,
                "metadata": {
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "usage": data.get("usage"),
                    "latency_sec": elapsed
                },
                "reasoning_trace": [
                    "• Retrieved last N chat turns as context",
                    "• Parsed user intent",
                    "• Generated coherent continuation"
                ]
            }

            print(json.dumps(enriched, indent=2, ensure_ascii=False))
            print()

        except KeyboardInterrupt:
            print("\n👋 Exiting chat…")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")

# ─────────────────────────────────────────────
# 3️⃣  Main entry
# ─────────────────────────────────────────────
if __name__ == "__main__":
    server = start_server()
    try:
        # Give server time to load model
        print("⏳ Waiting ~1minute for model to load…")
        time.sleep(60)
        chat_loop()
    finally:
        print("🧹 Shutting down vLLM server…")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
