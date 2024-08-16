# syntax=docker/dockerfile:1
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Default command to run your app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]