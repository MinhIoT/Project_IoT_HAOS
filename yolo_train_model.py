from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("models/yolo11n.pt")

    # Train model
    train_results = model.train(
        data="data.yaml",
        epochs=100,            # số vòng lặp huấn luyện
        imgsz=640,             # kích thước ảnh đầu vào
        device=0,              # GPU đầu tiên, CPU = "cpu"
        batch=8,               # số ảnh xử lý trong cùng 1 tệp huấn luyện
        name="yolo11_people",  # đặt tên thư mục lưu trữ
        verbose=True,          # hiển thị log chi tiết
        workers=8              # số luồng CPU dùng để đọc dữ liệu
    )

    print(train_results)

if __name__ == "__main__":
    main()
