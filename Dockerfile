FROM python:3.11.5-slim

WORKDIR /app

# Copy your dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose ports for both Streamlit & FastAPI options
ENV PORT=7860
EXPOSE 7860

# Use an environment variable to control which mode to launch
# Default to Streamlit for now
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]
