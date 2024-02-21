# Dockerfile

# Sử dụng hình ảnh cơ sở chứa Python
FROM python:3.9-slim

# Thiết lập thư mục làm việc của ứng dụng
WORKDIR /app

# Sao chép mã nguồn ứng dụng và tệp requirements vào hình ảnh
COPY . .

# Cài đặt các phụ thuộc từ requirements.txt
RUN pip install -r requirements.txt

# Mở cổng mà ứng dụng chạy trên
EXPOSE 5000

# Khởi chạy ứng dụng khi container được bắt đầu
CMD ["python", "app.py"]
