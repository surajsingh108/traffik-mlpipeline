FROM python:3.11-slim

WORKDIR /app

# libgomp1 required by LightGBM; supervisor manages api + poller processes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY . .

# Make poller executable
RUN chmod +x /app/infra/poller.sh

# Create directories for runtime artifacts
RUN mkdir -p /app/data /app/model

EXPOSE 8000

CMD ["supervisord", "-n", "-c", "/app/supervisord.conf"]
