# Use a minimal Python 3.9 base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all files from the current directory to the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Run the Streamlit app on port 8501 accessible externally
# Disable XSRF protection to allow batch prediction file uploads from the notebook
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
