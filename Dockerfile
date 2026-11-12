FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py analysis.py importer.py import_datasets.py datasets.yaml ./

# Imported datasets live here; mount a volume so they survive rebuilds.
RUN mkdir -p /app/data
VOLUME /app/data

EXPOSE 8001

# --proxy-headers: runs behind nginx-proxy (HTTPS terminates there)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers", "--forwarded-allow-ips", "*"]
