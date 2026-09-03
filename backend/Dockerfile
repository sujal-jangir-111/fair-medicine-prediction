# 1. Use official lightweight Python 3.11 image
FROM python:3.11-slim

# 2. Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project files
COPY . .

# 6. Expose default port
EXPOSE 8000

# 7. Start Uvicorn (supports Render's dynamic $PORT or defaults to 8000)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]