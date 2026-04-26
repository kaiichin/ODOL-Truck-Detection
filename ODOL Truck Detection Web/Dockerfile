FROM python:3.10

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for Hugging Face with open permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Copy the rest of the application
COPY . .

# Expose port 7860 (Hugging Face Spaces requirement)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
