---

# 🧠 Arch-Function-3B Local Chat (vLLM + Python)

Interactive local deployment of **`katanemo/Arch-Function-3B`** using [vLLM](https://github.com/vllm-project/vllm).
This setup runs the **full-precision FP16 model** (≈ 15 – 16 GB VRAM) on your GPU, exposes an OpenAI-compatible API, and provides a lightweight terminal chat client with short-term memory and pretty-printed JSON outputs.

---

## 🚀 Features

* 🔧 **FP16 full-precision serving** (no quantization)
* 💬 **Interactive chat REPL** in your terminal
* 🧠 **Conversation memory** (retains last 6 turns)
* 🪶 **Formatted JSON responses** with token-usage metadata
* 🧹 **Graceful startup / shutdown** in a single Python script

---

## ⚠️ POC Disclaimer

This project is a proof-of-concept for running `katanemo/Arch-Function-3B` locally with vLLM, plus a minimal Streamlit UI and a REST analyzer. It is intended for local experiments, rapid iteration, and evaluating prompt behavior. It is not a production-ready system.

### What can go wrong (and why)

The current setup has known limitations that can impact correctness and UX:

1. **Limited domain knowledge**
   - The model can be generic, shallow, or inconsistent outside its training biases. Some outputs may feel "lost" or vague.

2. **No tool calling or grounding**
   - The model cannot look things up, execute tools, or verify facts. Answers are unverified, which is risky for accuracy-heavy requests.

3. **Guardrail mismatch**
   - The `/analyze` endpoint rejects prompts unless they are clearly fantasy-photography. Many valid requests will be refused unless you adapt `is_photography_prompt()`.

4. **Strict JSON parsing**
   - The analysis flow expects JSON-only model responses. If the model returns malformed JSON, `/analyze` will fail with a 502.

5. **Heavy GPU footprint**
   - FP16 serving needs ~15–16 GB VRAM. Background services + Streamlit can add overhead and reduce headroom.

6. **Latency and warmup**
   - vLLM startup and first-token latency are non-trivial, especially on WSL2; expect slow first responses.

### Tradeoffs

- **Simplicity over robustness**: Single-script launch and minimal UI keep the POC easy to run, but error handling and resilience are limited.
- **Local-only over hosted**: No external dependencies make this safe to test, but it offers no multi-user support or scaling.
- **Minimal prompting over strong guarantees**: Lightweight prompts are easy to tweak, but do not enforce safety, determinism, or JSON validity.

## 📦 Prerequisites

| Requirement     | Minimum                                      |
| --------------- | -------------------------------------------- |
| **GPU**         | NVIDIA RTX 4070 / 4080 / 4090 (≥ 16 GB VRAM) |
| **OS**          | Ubuntu 22 / WSL2 with CUDA 12.1+             |
| **Python**      | 3.10 – 3.12                                  |
| **CUDA Driver** | ≥ 535                                        |
| **vLLM**        | 0.11 +                                       |

Check GPU from inside WSL2:

```bash
nvidia-smi
```

---

## ⚙️ Installation

1. **Clone / create project folder**

   ```bash
   mkdir arch3b_chat && cd arch3b_chat
   ```

2. **Create virtual environment**

   ```bash
   python3 -m venv vllm-env
   source vllm-env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -U pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install vllm requests
   ```
4. **Optional (but not)**
  -install requirements.txt
---

## 🧩 Running the Chat

Run the analysis server (REST + vLLM launcher):

```bash
python quant_arch_function_3b.py
```

Environment knobs:

| Variable | Default | Description |
| --- | --- | --- |
| `SERVICE_PORT` | 8100 | Port for the FastAPI server exposing `/analyze` + `/health`. |
| `VLLM_MODEL_ID` | katanemo/Arch-Function-3B | HF model served through vLLM. |
| `AUTO_START_VLLM` | true | Launch `vllm serve …` automatically. Set `false` if you manage vLLM separately. |
| `STATUS_WEBHOOK_URL` | _(unset)_ | Optional webhook notified with `vllm_started` / `vllm_completed`. |
| `VLLM_POST_PROCESSING_URL` | _(external)_ | Point the nanoGPT API at `http://<host>:8100/analyze` so it can dispatch completed chats. |

When `/analyze` receives a request, it:

1. Notifies `STATUS_WEBHOOK_URL` (if configured) that vLLM work started.
2. Uses vLLM to judge whether the prompt is **fantasy-photography** related. If not, it returns `{"status":"rejected"}` with the polite refusal.
3. Builds a ranking prompt with all six expert responses, asks vLLM for the best role + rationale, and returns structured JSON including a 10-sentence chain of thought.
4. Sends a `vllm_completed` webhook containing the final payload so the Node gateway/UI can surface the result asynchronously.

### What happens

1. The script launches `vllm serve katanemo/Arch-Function-3B` in the background.
2. Wait ≈ 10 s for the model to load.
3. You’ll see:

   ```
   💬 Chat ready. Type your message (Ctrl +C to exit)
   You:
   ```
4. Type messages → receive structured JSON replies:

   ```json
   {
     "response": "1. **Modularity** ...",
     "metadata": {
       "finish_reason": "length",
       "usage": {"prompt_tokens": 10, "completion_tokens": 100},
       "latency_sec": 1.47
     },
     "reasoning_trace": [
       "• Retrieved last N chat turns as context",
       "• Parsed user intent",
       "• Generated coherent continuation"
     ]
   }
   ```

---

## 🧠 Conversation Memory

The script automatically includes the **last 6 exchanges** in each new prompt.
Change this line to adjust:

```python
for turn in history[-6:]:
```

Persistent memory (saving to disk) can be added by dumping `history` to JSON on exit.

---

## 🔧 Common Adjustments

| Parameter       | Flag                       | Description                      |
| --------------- | -------------------------- | -------------------------------- |
| GPU usage limit | `--gpu-memory-utilization` | set 0.8 – 0.9 to avoid OOM       |
| Context window  | `--max-model-len`          | reduce to 4096 if VRAM tight     |
| Precision       | `--dtype`                  | `float16` (default) or `float32` |
| Creativity      | `temperature`              | lower = more deterministic       |
| Tokens          | `max_tokens`               | increase for longer answers      |

---

## 📤 API Access

While the chat script runs, the server also exposes an **OpenAI-compatible endpoint**:

```
POST http://localhost:8000/v1/completions
```

Example curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"katanemo/Arch-Function-3B","prompt":"Hello!","max_tokens":50}'
```

**Guardrail reminder:** `/analyze` rejects prompts unless they clearly describe photography in a fantasy setting. Make sure the payload (prompt or raw multi response) contains that context, or adjust `is_photography_prompt()` if needed.

- TODO
- [ ] - Check the other repo test_vllm_post_processing.sh for testing 
- [ ] - Check @Todo.md for post mvp
---

## 🧹 Shutdown

Press **Ctrl +C** once to exit chat, twice to terminate the vLLM server.
All resources release cleanly and CUDA context resets.

---

## 🧾 License

MIT / Apache-2.0 — for local experimentation only.
Refer to [katanemo/Arch-Function-3B on Hugging Face](https://huggingface.co/katanemo/Arch-Function-3B) for original model terms.

---

### ❤️ Credits

* [vLLM Team](https://github.com/vllm-project/vllm) for the inference engine
* [Katanemo AI](https://huggingface.co/katanemo) for Arch-Function-3B
* Script & docs adapted by @ Mrdjan Stajić 2025



