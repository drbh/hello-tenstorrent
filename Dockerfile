FROM python:3.8-slim

WORKDIR /app

# Copy project files
COPY . /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install uv && \
    uv venv

# Install the project using uv
RUN . .venv/bin/activate && \
    uv pip install -e .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD [".venv/bin/python", "scripts/working.py"]