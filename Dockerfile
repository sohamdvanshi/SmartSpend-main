FROM python:3.11-slim

WORKDIR /app

# Install system dependencies:
# - tesseract-ocr: required by pytesseract in app.py
# - poppler-utils: required by pdfplumber for PDF processing
# - libgl1, libglib2.0-0: required by OpenCV (cv2)
# - libsm6, libxext6, libxrender1: additional OpenCV deps
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Copy trained ML model files
COPY Expense_model/models/ ./Expense_model/models/

# Expose Flask port
EXPOSE 5000

# Start the Flask backend
CMD ["python", "backend/app.py"]
