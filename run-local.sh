#!/bin/bash

# Stop any running containers and remove them
docker-compose down

# Build the Docker image
docker-compose build

# Start the training job
docker-compose up
