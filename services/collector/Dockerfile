FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY services/collector/requirements.txt ./collector-requirements.txt
RUN python -m pip install --no-cache-dir -r collector-requirements.txt
COPY data ./data
COPY configs/services.json ./configs/services.json
COPY shared ./shared
COPY services/collector ./service
CMD ["uvicorn", "service.app:app"]
