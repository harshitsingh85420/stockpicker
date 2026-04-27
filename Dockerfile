# P51 — Production Docker image for the 5-Session Stock Picker
# Build:  docker build -t stockpicker .
# Run:    docker run --env-file .env stockpicker
# Tests:  docker run stockpicker pytest tests/ -v

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements-pinned.txt .
RUN pip install --no-cache-dir -r requirements-pinned.txt

# Copy source
COPY production/      production/
COPY tests/           tests/
COPY momentum_features.py .
COPY train_model.py .
COPY run_full_cycle.py .
COPY stock_picker_data/models/ stock_picker_data/models/

# Data directories (populated at runtime via volume mount)
RUN mkdir -p stock_picker_data/cache/bse \
             stock_picker_data/results \
             stock_picker_data/audit \
             stock_picker_data/shap_logs \
             stock_picker_data/backtest_results

ENV PYTHONUNBUFFERED=1

# Default: run daily picks
CMD ["python", "run_full_cycle.py"]
