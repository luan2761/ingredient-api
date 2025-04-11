# Sử dụng base image nhẹ có Python
FROM python:3.10-slim

# Cài các thư viện hệ thống cần thiết cho YOLOv8 và xử lý ảnh
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file requirements trước để tận dụng cache
COPY requirements.txt .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Mở port 8000 (FastAPI)
EXPOSE 8000

# Chạy ứng dụng với Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
