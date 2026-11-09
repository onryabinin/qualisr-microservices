FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir --no-deps --target /tmp/plotly plotly==7.1.0 \
    && mkdir -p ./service/static \
    && cp /tmp/plotly/plotly/package_data/plotly.min.js ./service/static/plotly.min.js \
    && rm -rf /tmp/plotly
COPY configs/services.json ./configs/services.json
COPY shared ./shared
COPY services/visualization ./service
CMD ["uvicorn", "service.app:app"]
