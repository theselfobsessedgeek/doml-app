# Use Python base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy app files
COPY app.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]
