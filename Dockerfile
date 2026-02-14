FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8100 \
    VLLM_PORT=8000 \
    VLLM_API_URL=http://localhost:8000/v1/completions \
    VLLM_MODEL_ID=katanemo/Arch-Function-3B

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt /app/requirements-app.txt
RUN python3 -m pip install -U pip \
    && python3 -m pip install -r /app/requirements-app.txt

COPY . /app

EXPOSE 8000 8100 8501

CMD ["bash", "-lc", "python /app/quant_arch_function_3b.py & streamlit run /app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]
