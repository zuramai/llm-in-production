FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ch11_loki_handler.py ch11_sim.py main.py ./
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011", "--log-level", "warning"]
