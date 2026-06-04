FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p ./data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
