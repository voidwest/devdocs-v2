FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY ./app ./app

RUN chown -R appuser:appgroup /app

USER appuser

WORKDIR /app/app
ENV PYTHONPATH=/app/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
