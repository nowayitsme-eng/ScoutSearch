FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies (sometimes needed for building pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user matching Hugging Face's requirements
RUN useradd -m -u 1000 user

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container and set ownership
COPY --chown=user . /app

# Switch to the non-root user
USER user

# Hugging Face Spaces routes traffic to port 7860 by default
EXPOSE 7860

# Start Gunicorn server bound to 0.0.0.0:7860
# Since we have 16GB of RAM, we can afford multi-workers
CMD ["gunicorn", "Backend.src.app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "3600"]
