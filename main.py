from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import io
import logging

# Cấu hình logger để hiển thị log trên Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load danh sách nhãn nguyên liệu
try:
    with open("ingredient_labels.txt", "r", encoding="utf-8") as f:
        ingredient_labels = set([line.strip().lower() for line in f])
    logger.info(f"📦 Đã tải {len(ingredient_labels)} nhãn nguyên liệu.")
except Exception as e:
    logger.exception("❌ Không thể đọc file ingredient_labels.txt.")
    ingredient_labels = set()

# Load mô hình YOLOv8n đã fine-tuned nếu có, nếu không sẽ dùng mặc định
try:
    model = YOLO("yolov8n.pt")
    logger.info("✅ Đã load mô hình YOLOv8n.")
except Exception as e:
    logger.exception("❌ Không thể load YOLOv8n.")

@app.get("/")
def home():
    return {"message": "🍜 Chào mừng đến với API nhận diện nguyên liệu bằng YOLOv8n!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"📥 Nhận file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Dự đoán bằng YOLOv8
        logger.info("🚀 Đang nhận diện bằng YOLOv8n...")
        results = model(image)[0]  # Lấy kết quả đầu tiên

        result_list = []
        for box in results.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            confidence = float(box.conf)
            is_ingredient = label.lower() in ingredient_labels

            result_list.append({
                "label": label,
                "confidence": f"{confidence * 100:.2f}%",
                "is_ingredient": is_ingredient
            })

        logger.info("✅ Nhận diện xong. Kết quả: %s", result_list)
        return JSONResponse(content={"results": result_list})

    except Exception as e:
        logger.exception("❌ Lỗi xử lý ảnh.")
        return JSONResponse(content={"error": str(e)}, status_code=500)
