from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import io

app = FastAPI()

# Đọc danh sách nhãn nguyên liệu từ file
with open("ingredient_labels.txt", "r", encoding="utf-8") as f:
    ingredient_labels = set([line.strip().lower() for line in f])

# Load YOLOv8n model
model = YOLO("yolov8n.pt")  # Tự động tải nếu chưa có
print(model)  # Hiển thị YOLOv8n summary trong log khi khởi động

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API nhận diện nguyên liệu bằng YOLOv8n!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh từ request
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Dự đoán bằng mô hình YOLO
        results = model(image)
        result_list = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                is_ingredient = label.lower() in ingredient_labels
                result_list.append({
                    "label": label,
                    "confidence": f"{confidence * 100:.2f}%",
                    "is_ingredient": is_ingredient
                })

        return JSONResponse(content={"results": result_list})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
