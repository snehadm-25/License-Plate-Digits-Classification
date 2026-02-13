from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import os
from model import DigitClassifier

app = FastAPI(title="License Plate Digits Classification")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 
               'V', 'W', 'X', 'Y', 'Z']

model = DigitClassifier(num_classes=len(class_names))
if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
            
        return {
            "prediction": class_names[predicted.item()],
            "confidence": f"{confidence * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    with open(os.path.join("static", "index.html"), "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces require port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
