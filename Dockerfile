FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
# - tesseract-ocr + English traineddata: fallback OCR for scanned PDFs
# - ghostscript + unpaper + pngquant: OCRmyPDF runtime prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    unpaper \
    pngquant \
    qpdf \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml/models ml/knowledge_base uploads

# Create a named user for UID 1000 so getpass.getuser() resolves correctly
# (torch._dynamo calls getpwuid at import time; bare UID with no passwd entry crashes).
# HF Spaces also runs as UID 1000.
RUN groupadd --system --gid 1000 appuser && \
    useradd --system --uid 1000 --gid 1000 --no-create-home appuser

RUN chown -R appuser:appuser /app/ml/models /app/uploads && \
    chmod -R 755 /app/ml/models /app/uploads

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
