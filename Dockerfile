FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit requests

# Copy Streamlit app
COPY app/app.py ./app/app.py

# Expose port
EXPOSE 8080

# Run Streamlit
CMD streamlit run app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
