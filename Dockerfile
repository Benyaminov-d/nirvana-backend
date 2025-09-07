FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    gcc \
    g++ \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

COPY pip.conf /etc/pip.conf

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# COPY . /app/
# COPY data-snapshots/ /app/data-snapshots/
# COPY lambert/ /app/lambert/

ENV NVAR_LICENSE=DEV \
    PYTHONPATH=/app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]