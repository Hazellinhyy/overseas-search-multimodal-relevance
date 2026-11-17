FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /code
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/code/src
COPY pyproject.toml README.md /code/
COPY src /code/src
COPY configs /code/configs
COPY scripts /code/scripts
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg && rm -rf /var/lib/apt/lists/* && python3 -m pip install --no-cache-dir -e .
ENTRYPOINT ["/code/scripts/run_gpu_preflight.sh"]
