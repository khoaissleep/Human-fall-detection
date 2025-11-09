# Human Fall Detection 

Dự án phát hiện ngã của con người sử dụng YOLO và Norfair tracking để theo dõi và phát hiện các trường hợp ngã trong video.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam hoặc file video để test

## Cài đặt

### Bước 1: Clone repository hoặc tải xuống dự án

```bash
git clone https://github.com/khoaissleep/Human-fall-detection.git
cd Human-fall-detection
```

### Bước 2: Cài đặt norfair và các thư viện cần thiết

```bash
pip install norfair
pip install -r requirements.txt
```

Các thư viện sẽ được cài đặt:
- `ultralytics`
- `cvzone`
- `norfair`
- `opencv-python`
- `numpy`

### Bước 3: Tải mô hình YOLO (nếu chưa có)

Mô hình `yolov8n.pt` sẽ được tự động tải xuống khi chạy lần đầu.

## Chạy chương trình

### Chuẩn bị file video

Đảm bảo có file video `fall.mp4` trong thư mục dự án, hoặc thay đổi đường dẫn trong file `main.py`:

```python
VIDEO_PATH = "fall.mp4"  # Thay đổi đường dẫn tại đây
```

### Chạy chương trình

```bash
python main.py
```

## Cấu trúc dự án

```
Human-fall-detection/
├── main.py              # File chính
├── requirements.txt     # Danh sách thư viện
├── classes.txt          # Danh sách các lớp đối tượng
├── fall.mp4             # Video mẫu
├── yolov8n.pt           # Mô hình YOLO
├── Norfair              # Thư viện tracking đối tượng
└── README.md            # File hướng dẫn này
```

## Giải thích hoạt động

1. **Phát hiện đối tượng**: Sử dụng YOLO để phát hiện người trong video
2. **Theo dõi đối tượng**: Sử dụng Norfair để theo dõi người qua các frame
3. **Phát hiện ngã**: Phân tích tỷ lệ khung hình để phát hiện ngã
4. **Hiển thị kết quả**: Vẽ bounding box và nhãn trên video
