FROM python:3.11-slim

LABEL maintainer="Pranav Mahesh Palled <smriti-os@hacktoberfest2026.tech>"
LABEL org.opencontainers.image.title="SMRITI-OS"
LABEL org.opencontainers.image.description="Multimodal Open-Source Archival Rescue & Cultural Context Engine powered by Google Gemma 4"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Seed initial archive records at build time
RUN python seed_demo_archive.py

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
