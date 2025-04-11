from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torchvision.transforms as transforms
import torchvision.models as models
import torch

app = FastAPI()

# Dùng mô hình ResNet pretrained
from torchvision.models import ResNet50_Weights
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

# Lấy danh sách nhãn từ ImageNet
categories = weights.meta["categories"]

# Hàm xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def read_root():
    return {"message": "Welcome to Ingredient Recognition API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Lấy top 5 kết quả
        top5 = torch.topk(probs, 5)
        result = []
        for idx, prob in zip(top5.indices, top5.values):
            result.append({
                "label": categories[idx],
                "confidence": f"{prob.item()*100:.2f}%"
            })

        return JSONResponse(content={"results": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
