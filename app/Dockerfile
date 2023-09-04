
FROM python:3.11

WORKDIR /app

COPY app-requirements.txt ./requirements.txt
RUN pip install --upgrade pip -r requirements.txt

COPY . .

ARG VERSION
ENV DOCKER_TAG=$VERSION

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "index.py", "--server.port=8502", "--server.address=0.0.0.0"]

