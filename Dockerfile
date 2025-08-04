FROM python:3.12

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

# Make sure build tools are present
RUN apt-get update && apt-get install -y build-essential python3-dev

# Upgrade pip and tools
RUN pip install --upgrade pip setuptools wheel

# Install numpy first
RUN pip install numpy

# Use binary-only install for requirements
RUN pip install --only-binary=:all: -r requirements.txt



# Copy the rest of the project
COPY . .

# Expose the port Flask runs on
EXPOSE 3004

# Run the app
CMD ["python", "app.py"]
