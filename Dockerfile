# Use official lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy application source files
COPY lfm2_vl_stream.py /app/
COPY query_db.py /app/

# Create folders for mounted data (optional but good practice)
RUN mkdir -p /app/testData /app/video_captions_db

# The default behavior is to run the streaming captioner
# Users can override the arguments at runtime (e.g. video files, output path)
ENTRYPOINT ["python", "lfm2_vl_stream.py"]
