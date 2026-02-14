import json
import os
from typing import List

import requests
import streamlit as st


DEFAULT_MODEL = os.getenv("VLLM_MODEL_ID", "katanemo/Arch-Function-3B")
DEFAULT_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/completions")
DEFAULT_MAX_TOKENS = int(os.getenv("STREAMLIT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("STREAMLIT_TEMPERATURE", "0.3"))


def post_vllm(prompt: str, *, api_url: str, model: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["text"].strip()


def parse_questions(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        questions = data.get("questions", [])
        return [q.strip() for q in questions if q.strip()]
    except json.JSONDecodeError:
        lines = [line.strip("- ") for line in text.splitlines()]
        return [line for line in lines if line]


def get_clarifying_questions(prompt: str, *, api_url: str, model: str) -> List[str]:
    clarifier_prompt = (
        "You are a helpful assistant analyzing user requests. Look at this prompt and determine if it's vague or missing important details. "
        "If the prompt is vague or unclear, ask 1-2 specific clarifying questions. If it's already clear and specific, return empty. "
        "Respond ONLY as valid JSON with this exact format: {\"questions\": [\"question 1\", \"question 2\"]} or {\"questions\": []}\n\n"
        f"User prompt: {prompt.strip()}\n\n"
        "JSON response:"
    )
    raw = post_vllm(
        clarifier_prompt,
        api_url=api_url,
        model=model,
        max_tokens=200,
        temperature=0.1,
    )
    print(f"DEBUG clarifier raw response: {raw}")
    questions = parse_questions(raw)
    print(f"DEBUG parsed questions: {questions}")
    return questions


def get_followup_questions(prompt: str, answer: str, *, api_url: str, model: str) -> List[str]:
    followup_prompt = (
        "Generate exactly 2 short follow-up questions that could refine or extend the answer. "
        "Respond only as JSON: {\"questions\": [\"...\", \"...\"]}.\n\n"
        f"User prompt: {prompt.strip()}\n"
        f"Answer: {answer.strip()}"
    )
    raw = post_vllm(
        followup_prompt,
        api_url=api_url,
        model=model,
        max_tokens=120,
        temperature=0.3,
    )
    questions = parse_questions(raw)
    return questions[:2]


def build_final_prompt(prompt: str, questions: List[str], answers: List[str]) -> str:
    if not questions:
        return prompt.strip()
    pairs = []
    for idx, (q, a) in enumerate(zip(questions, answers), start=1):
        pairs.append(f"{idx}. Q: {q}\n   A: {a}")
    clarifications = "\n".join(pairs)
    return (
        "Use the clarifications to answer the original prompt.\n\n"
        f"Original prompt: {prompt.strip()}\n\n"
        f"Clarifications:\n{clarifications}\n\n"
        "Answer:" 
    )


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = ""
    if "pending_questions" not in st.session_state:
        st.session_state.pending_questions = []
    if "pending_answers" not in st.session_state:
        st.session_state.pending_answers = []
    if "followups" not in st.session_state:
        st.session_state.followups = []
    if "prefill_prompt" not in st.session_state:
        st.session_state.prefill_prompt = ""


def reset_session() -> None:
    st.session_state.history = []
    st.session_state.pending_prompt = ""
    st.session_state.pending_questions = []
    st.session_state.pending_answers = []
    st.session_state.followups = []
    st.session_state.prefill_prompt = ""


def render_history() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])


def main() -> None:
    st.set_page_config(page_title="Arch-Function-3B QA", layout="centered")
    init_state()

    st.title("Arch-Function-3B Local Q/A")
    st.caption("Local Streamlit UI for vLLM completions with clarifying subquestions.")

    with st.sidebar:
        st.subheader("Settings")
        api_url = st.text_input("vLLM API URL", value=DEFAULT_API_URL)
        model = st.text_input("Model", value=DEFAULT_MODEL)
        max_tokens = st.number_input("Max tokens", min_value=64, max_value=2048, value=DEFAULT_MAX_TOKENS, step=64)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)
        if st.button("Reset session"):
            reset_session()
            st.rerun()

    render_history()

    if st.session_state.pending_questions:
        st.markdown("### Clarifying questions")
        with st.form("clarifications_form"):
            answers = []
            for idx, q in enumerate(st.session_state.pending_questions, start=1):
                answers.append(st.text_input(f"{idx}. {q}", key=f"clarify_{idx}"))
            submitted = st.form_submit_button("Submit clarifications")
        if submitted:
            st.session_state.pending_answers = answers
            final_prompt = build_final_prompt(
                st.session_state.pending_prompt,
                st.session_state.pending_questions,
                st.session_state.pending_answers,
            )
            try:
                answer = post_vllm(
                    final_prompt,
                    api_url=api_url,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
            st.session_state.history.append({"role": "user", "content": st.session_state.pending_prompt})
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.session_state.pending_questions = []
            st.session_state.pending_answers = []
            st.session_state.pending_prompt = ""
            st.session_state.followups = get_followup_questions(
                st.session_state.history[-2]["content"],
                st.session_state.history[-1]["content"],
                api_url=api_url,
                model=model,
            )
            st.rerun()
        return

    with st.form("prompt_form"):
        prompt = st.text_area("Your prompt", value=st.session_state.prefill_prompt, height=140)
        submitted = st.form_submit_button("Send")
    if submitted and prompt.strip():
        st.session_state.prefill_prompt = ""
        try:
            questions = get_clarifying_questions(prompt, api_url=api_url, model=model)
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            return
        if questions:
            st.session_state.pending_prompt = prompt
            st.session_state.pending_questions = questions
            st.rerun()
        else:
            try:
                answer = post_vllm(
                    prompt,
                    api_url=api_url,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return
            st.session_state.history.append({"role": "user", "content": prompt})
            st.session_state.history.append({"role": "assistant", "content": answer})
            st.session_state.followups = get_followup_questions(
                prompt,
                answer,
                api_url=api_url,
                model=model,
            )
            st.rerun()

    if st.session_state.followups:
        st.markdown("### Follow-up questions")
        cols = st.columns(len(st.session_state.followups))
        for idx, question in enumerate(st.session_state.followups):
            if cols[idx].button(question, key=f"followup_{idx}"):
                st.session_state.prefill_prompt = question
                st.session_state.followups = []
                st.rerun()


if __name__ == "__main__":
    main()
