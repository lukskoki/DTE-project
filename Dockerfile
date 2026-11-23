# One image for every Python service; docker-compose picks the command.
FROM python:3.11-slim
# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shadowhome/ shadowhome/
COPY pipeline/ pipeline/
COPY services/ services/
COPY config/ config/
COPY models/ models/
COPY reports/ reports/
COPY .streamlit/ .streamlit/
ENV PYTHONUNBUFFERED=1
