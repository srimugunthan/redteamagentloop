# Multi-stage build: Node (frontend) → Python (backend + static files)

# ── Stage 1: Build React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /frontend
COPY console/frontend/package.json ./
RUN npm install --frozen-lockfile

COPY console/frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Install Python deps with the console extra
COPY pyproject.toml uv.lock ./
RUN uv sync --extra console --no-dev

# Copy project source
COPY redteamagentloop/ ./redteamagentloop/
COPY console/backend/ ./console/backend/
COPY console/__init__.py ./console/__init__.py
COPY config.yaml authorization.txt ./
COPY prompts/ ./prompts/
COPY reports/templates/ ./reports/templates/

# Copy compiled frontend into the location main.py expects
COPY --from=frontend-build /frontend/dist ./console/frontend/dist

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "console.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
