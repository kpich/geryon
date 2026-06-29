# Sandbox image for running LLM-authored analysis scripts.
#
# Build (context = this directory):
#   docker build -t geryon-sandbox src/geryon/sandbox
#
# Scripts get geryon_runtime on the path (db(), report(), artifact_path()).
# Run is always: --network none, data mounted read-only at /data, scratch at
# /scratch. The image itself contains no project source beyond runtime.py and
# no credentials.
FROM python:3.11-slim

# Non-interactive, no bytecode cruft, unbuffered stdout so logs stream out.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/geryon \
    GERYON_DATA_DIR=/data \
    GERYON_SCRATCH_DIR=/scratch

WORKDIR /work

COPY requirements.txt /opt/geryon/requirements.txt
RUN pip install --no-cache-dir -r /opt/geryon/requirements.txt

COPY runtime.py /opt/geryon/geryon_runtime.py

# Run as an unprivileged user; /scratch is bind-mounted writable at run time.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["python", "-c", "import geryon_runtime; print('geryon-sandbox ready')"]
