FROM python:3.11-slim
WORKDIR /opt/octave
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /opt/octave/backend/requirements.txt
RUN pip install --no-cache-dir -r /opt/octave/backend/requirements.txt
COPY backend/ /opt/octave/backend/
COPY frontend/dist/ /opt/octave/frontend/dist/
WORKDIR /opt/octave/backend
ENV OCTAVE_DATA_DIR=/var/lib/octave/data PYTHONUNBUFFERED=1
RUN mkdir -p /var/lib/octave/data
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
