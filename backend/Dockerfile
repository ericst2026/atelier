# The API image, and — with one build argument — the CPU worker as well.
#
#   docker build -t atelier-api ./backend
#   docker build --build-arg WITH_EXPERIMENTS=1 -t atelier-cpu ./backend
#
# There is no CUDA here and no torch. The API never touches a GPU: NVML is read
# through the host driver when the container is given the nvidia runtime.
#
# WITH_EXPERIMENTS=1 adds the pure-Python libraries the experiment scripts import,
# which is what a worker needs on a machine with no GPU — the laptop stack in
# docker-compose.windows.yml runs the API and the worker from this one image. It
# does not add torch: that is a two-gigabyte download for a trial that does not need
# it, and the GPU worker has its own image (Dockerfile.worker).
FROM python:3.12-slim
ARG WITH_EXPERIMENTS=0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_EXPERIMENTS" = "1" ]; then pip install --no-cache-dir numpy tqdm; fi
# To run the pretraining experiment at the tiny preset on a laptop, add the CPU
# wheel as well — about 200 MB rather than two gigabytes:
#   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY atelier ./atelier
EXPOSE 8000
CMD ["uvicorn", "atelier.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
