FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV HOST=0.0.0.0 \
    PORT=8790 \
    CONFIG_PATH=/data/config.json

VOLUME ["/data"]
EXPOSE 8790

CMD ["mistral-relay-gateway"]
