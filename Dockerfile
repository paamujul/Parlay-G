FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so source edits don't invalidate the pip layer.
COPY NCAA/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Preserve repo layout: config.py resolves ../../config/ncaa.yaml from its own location.
COPY NCAA/app/ ./NCAA/app/
COPY config/ ./config/

WORKDIR /app/NCAA/app
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]