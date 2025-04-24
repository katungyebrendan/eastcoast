# Use an official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run with uvicorn (make sure your app file and object name are correct)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
