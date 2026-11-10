# Test double for the LLM provider (load, rollout and autoscaling experiments). Not for prod.
#   docker build -t shopflow-fakellm:1.0.0 ch20/01_main-chapter-code/services/fakellm
FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi==0.162.0 uvicorn==0.54.0 \
    && useradd --system --uid 10001 app
WORKDIR /app
COPY ch20_fakellm.py .
USER 10001
EXPOSE 8080
CMD ["uvicorn", "ch20_fakellm:app", "--host", "0.0.0.0", "--port", "8080"]
