FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV PORT=8080

# Make port 8080 available
EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 8