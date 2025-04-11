# Sử dụng base image nhẹ có Python
FROM python:3.10-slim

# Cài thư viện hệ thống cần thiết để YOLOv8 hoạt động
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file requirements trước để tận dụng cache layer
COPY requirements.txt .

# Cài các thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Expose cổng cho FastAPI
EXPOSE 8000

# Chạy ứng dụng bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
