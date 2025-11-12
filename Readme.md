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

Run the all-in-one script:

```bash
python quant_arch_function_3b.py
```

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
