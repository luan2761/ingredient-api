from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import io
import logging

# Cấu hình logger để xem log trên Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Đọc danh sách nhãn nguyên liệu từ file
try:
    with open("ingredient_labels.txt", "r", encoding="utf-8") as f:
        ingredient_labels = set([line.strip().lower() for line in f])
    logger.info(f"📦 Đã tải {len(ingredient_labels)} nhãn nguyên liệu")
except Exception as e:
    logger.exception("❌ Lỗi khi đọc ingredient_labels.txt")
    ingredient_labels = set()

# Load YOLOv8n model
try:
    model = YOLO("yolov8n.pt")  # Tự động tải nếu chưa có
    logger.info("✅ YOLOv8n model đã được load")
    logger.info(f"🔍 Model classes: {model.names}")
except Exception as e:
    logger.exception("❌ Không thể load YOLOv8n model")

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API nhận diện nguyên liệu bằng YOLOv8n!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("📥 Nhận file: %s", file.filename)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        logger.info("🚀 Đang chạy YOLOv8n...")
        results = model(image)
        logger.info("✅ YOLOv8n trả về kết quả")

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

        logger.info("🎯 Kết quả nhận diện: %s", result_list)
        return JSONResponse(content={"results": result_list})

    except Exception as e:
        logger.exception("❌ Lỗi khi xử lý ảnh")
        return JSONResponse(content={"error": str(e)}, status_code=500)
