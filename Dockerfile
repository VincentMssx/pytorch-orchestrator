# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY models/ /app/models/
COPY worker.py /app/

# The entrypoint will be the worker script
# Arguments will be passed to this script at runtime
ENTRYPOINT ["python", "worker.py"]
