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

# Upgrade pip and tools
RUN pip install --upgrade pip setuptools wheel

# Install numpy separately
RUN pip install numpy

# Disable isolated build for the rest (especially for pandas)
RUN PIP_NO_BUILD_ISOLATION=1 pip install -r requirements.txt


# Copy the rest of the project
COPY . .

# Expose the port Flask runs on
EXPOSE 3004

# Run the app
CMD ["python", "app.py"]
