#!/usr/bin/env python3
"""Arch-Function-3B vLLM analysis service (REST + webhooks)."""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import signal
import subprocess
import time
from typing import List, Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

MODEL_ID = os.getenv("VLLM_MODEL_ID", "katanemo/Arch-Function-3B")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_API_URL = os.getenv("VLLM_API_URL", f"http://localhost:{VLLM_PORT}/v1/completions")
AUTO_START_VLLM = os.getenv("AUTO_START_VLLM", "true").lower() == "true"
VLLM_WARMUP_SECONDS = int(os.getenv("VLLM_WARMUP_SECONDS", "60"))
SERVICE_HOST = os.getenv("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8100"))
STATUS_WEBHOOK_URL = os.getenv("STATUS_WEBHOOK_URL")
STATUS_WEBHOOK_TIMEOUT = float(os.getenv("STATUS_WEBHOOK_TIMEOUT", "5"))
REQUEST_TIMEOUT = float(os.getenv("VLLM_REQUEST_TIMEOUT", "120"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("arch_function_service")

VLLM_PROCESS: Optional[subprocess.Popen] = None


class ExpertResponse(BaseModel):
    role: str
    text: str

    @model_validator(mode="after")
    def strip_text(self) -> "ExpertResponse":
        self.role = self.role.strip()
        self.text = self.text.strip()
        return self


class AnalyzeRequest(BaseModel):
    chat_id: str
    prompt: str
    responses: Optional[List[ExpertResponse]] = None
    raw_multi_response: Optional[str] = None

    @model_validator(mode="after")
    def ensure_responses(self) -> "AnalyzeRequest":
        if self.responses:
            return self
        if not self.raw_multi_response:
            raise ValueError("Either responses or raw_multi_response must be provided")
        parsed = parse_raw_multi_response(self.raw_multi_response)
        if not parsed:
            raise ValueError("raw_multi_response could not be parsed")
        self.responses = parsed
        return self


class AnalyzeResult(BaseModel):
    status: str
    message: Optional[str] = None
    result: Optional[dict] = None


def start_vllm_server() -> Optional[subprocess.Popen]:
    if not AUTO_START_VLLM:
        logger.info("AUTO_START_VLLM disabled; assuming server already running")
        return None
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
    ]
    logger.info("Launching vLLM: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    time.sleep(VLLM_WARMUP_SECONDS)
    return proc


def stop_vllm_server():
    global VLLM_PROCESS
    if VLLM_PROCESS and VLLM_PROCESS.poll() is None:
        logger.info("Stopping vLLM server")
        VLLM_PROCESS.terminate()
        try:
            VLLM_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not exit, killing")
            VLLM_PROCESS.kill()
        VLLM_PROCESS = None


def parse_raw_multi_response(raw: str) -> List[ExpertResponse]:
    sections: List[ExpertResponse] = []
    pattern = re.compile(r"🔷\s+(.*?)\n-+\n?([\s\S]*?)(?=\n🔷|$)")
    for match in pattern.finditer(raw):
        role = match.group(1).strip()
        text = match.group(2).strip()
        if role and text:
            sections.append(ExpertResponse(role=role, text=text))
    return sections


def call_vllm(prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(VLLM_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"vLLM error: {response.text}")
    data = response.json()
    return data["choices"][0]["text"].strip()


def post_status(status: str, *, chat_id: str, payload: Optional[dict] = None) -> None:
    if not STATUS_WEBHOOK_URL:
        return
    body = {"chat_id": chat_id, "status": status}
    if payload:
        body["payload"] = payload
    try:
        requests.post(STATUS_WEBHOOK_URL, json=body, timeout=STATUS_WEBHOOK_TIMEOUT)
    except Exception as exc:
        logger.debug("Status webhook failed: %s", exc)


def is_photography_prompt(prompt: str) -> bool:
    judge_prompt = (
        "You are a classifier. Determine if the following request is specifically about "
        "creating photography in a fantasy setting (fantasy worlds, creatures, magic, etc.).\n"
        "Answer strictly with YES or NO.\n\n"
        f"Request: \"\"\"{prompt.strip()}\"\"\"\n"
    )
    result = call_vllm(judge_prompt, max_tokens=8, temperature=0.0).upper()
    return "YES" in result.split()


def build_analysis_prompt(prompt: str, responses: List[ExpertResponse]) -> str:
    response_blocks = []
    for idx, item in enumerate(responses, start=1):
        response_blocks.append(
            f"{idx}. Role: {item.role}\nResponse: {item.text}\n"
        )

    guidance = (
        "You are an elite photography art director evaluating expert advice for a fantasy scene. "
        "Determine which response best fulfills the user's request, describe strengths/weaknesses, "
        "and explain your decision."
    )

    template = f"""
{guidance}
User prompt: \"\"\"{prompt.strip()}\"\"\"

Expert responses:
{os.linesep.join(response_blocks)}

Tasks:
1. Verify the responses relate to photography in a fantasy context; if any deviate, note it.
2. Select the single best response and explain why.
3. Rank all responses from best to weakest with brief justification (strengths/risks).
4. Provide exactly 10 sentences describing your reasoning chain.

Return pure JSON with the structure:
{{
  "best_role": "...",
  "best_reason": "...",
  "ranking": [
    {{"role": "...", "verdict": "...", "reason": "..."}}
  ],
  "chain_of_thought": ["Sentence 1", ..., "Sentence 10"]
}}
"""
    return template


def analyze_responses(prompt: str, responses: List[ExpertResponse]) -> dict:
    analysis_prompt = build_analysis_prompt(prompt, responses)
    raw = call_vllm(analysis_prompt, max_tokens=700, temperature=0.4)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse vLLM JSON: %s", raw)
        raise HTTPException(status_code=502, detail="vLLM returned invalid JSON") from exc
    return parsed


app = FastAPI(title="Arch-Function-3B Analyzer")


@app.post("/analyze", response_model=AnalyzeResult)
def analyze(payload: AnalyzeRequest) -> AnalyzeResult: 
    responses = payload.responses or []
    chat_id = payload.chat_id
    prompt_for_guardrail = payload.raw_multi_response or payload.prompt
    prompt_for_guardrail = prompt_for_guardrail.replace('==================== MULTI-BRAIN OUTPUT ====================', '').replace('====================== END OF OUTPUT =======================', '').strip()
    logger.info("/analyze chat_id=%s guardrail=%r raw_len=%s structured=%s", chat_id, prompt_for_guardrail[:80], len(payload.raw_multi_response or '') if payload.raw_multi_response else 0, bool(payload.responses))
    post_status("vllm_started", chat_id=chat_id)

    prompt_for_guardrail = payload.raw_multi_response or payload.prompt
    prompt_for_guardrail = prompt_for_guardrail.replace('==================== MULTI-BRAIN OUTPUT ====================', '').replace('====================== END OF OUTPUT =======================', '').strip()
    if not is_photography_prompt(prompt_for_guardrail):
        message = "I cannot respond to that. Please provide a photography prompt in a fantasy setting."
        result = AnalyzeResult(status="rejected", message=message)
        post_status("vllm_completed", chat_id=chat_id, payload=result.model_dump())
        return result

    try:
        analysis = analyze_responses(payload.prompt, responses)
    except HTTPException as exc:
        post_status("vllm_completed", chat_id=chat_id, payload={"error": exc.detail})
        raise

    result = AnalyzeResult(status="ok", result=analysis)
    post_status("vllm_completed", chat_id=chat_id, payload=result.model_dump())
    return result


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    if AUTO_START_VLLM:
        VLLM_PROCESS = start_vllm_server()
        atexit.register(stop_vllm_server)
        signal.signal(signal.SIGTERM, lambda *_: stop_vllm_server())
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
