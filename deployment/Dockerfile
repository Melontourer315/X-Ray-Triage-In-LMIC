# Reference Dockerfile for the TB-Triage Flask app.
# Builds a CPU-only image with onnxruntime, Flask, Pillow, numpy.
# No PyTorch dependency: the model ships as ONNX (opset 17).
#
# Build:   docker build -t tbtriage:latest .
# Run:     docker run --rm -p 5000:5000 tbtriage:latest
# Verify:  curl http://127.0.0.1:5000/healthz

FROM python:3.11-slim

WORKDIR /app

# system deps for Pillow / numpy on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the application code + the bundled model
COPY app.py .
COPY templates/ ./templates/
COPY model/ ./model/

EXPOSE 5000
ENV FLASK_ENV=production

CMD ["python", "app.py"]
