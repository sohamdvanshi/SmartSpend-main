FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/ ./backend/

# Copy trained model files
COPY Expense_model/models/ ./Expense_model/models/

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5000

# Start the backend
CMD ["python", "backend/app.py"]