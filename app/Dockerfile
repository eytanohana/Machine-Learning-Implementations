
FROM ghcr.io/astral-sh/uv:python3.13-alpine

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN uv venv && uv pip install -r requirements.txt

COPY . .

ARG VERSION
ENV DOCKER_TAG=$VERSION

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]
