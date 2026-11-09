FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY services/ml/requirements.txt ./ml-requirements.txt
RUN python -m pip install --no-cache-dir -r ml-requirements.txt
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY configs/templates ./configs/templates
COPY configs/services.json ./configs/services.json
COPY shared ./shared
COPY services/ml ./service
CMD ["uvicorn", "service.app:app"]
