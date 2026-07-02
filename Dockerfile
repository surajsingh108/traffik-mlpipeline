FROM python:3.11-slim

WORKDIR /app

# Install Python deps before copying source (better layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY . .

# Create directories for runtime artifacts
RUN mkdir -p /app/data /app/model

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
