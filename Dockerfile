FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    curl

# Set work directory
WORKDIR /app

# Copy files
COPY requirements.txt requirements.txt

# Install numpy first to avoid build errors
RUN pip install --upgrade pip && pip install numpy

# Then install the rest
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Flask runs on
EXPOSE 3004

# Run the app
CMD ["python", "app.py"]
