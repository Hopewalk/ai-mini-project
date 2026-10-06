# ---- frontend assets (pnpm) -------------------------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /build
RUN npm install -g pnpm@12.9.1
COPY package.json pnpm-lock.yaml ./
COPY scripts/build-vendor.mjs ./scripts/
# postinstall runs scripts/build-vendor.mjs -> app/static/vendor
RUN pnpm install --frozen-lockfile

# ---- python app (poetry) ----------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

WORKDIR /app

# geopandas / pyproj / pyogrio wheels bundle GDAL and PROJ; libgomp is needed by scikit-learn;
# fonts-thai-tlwg gives matplotlib a Thai font (Loma) for cluster labels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==2.5.1
COPY pyproject.toml poetry.lock ./
# drop poetry's wheel cache in the same layer (~0.7 GB otherwise)
RUN poetry install --only main --no-root && rm -rf "$POETRY_CACHE_DIR"

COPY ml ./ml
COPY scripts ./scripts
COPY app ./app
COPY config.py run.py ./
COPY --from=frontend /build/app/static/vendor ./app/static/vendor

# data and artifacts are mounted as volumes (see docker-compose.yml)
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "run:app"]
