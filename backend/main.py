import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from predictor import ShipPredictor

app = FastAPI(title="Ship Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows React to access API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
# It will load the model from model/faster_rcnn_ship.pth
predictor = ShipPredictor()

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Create unique filenames
    unique_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1]
    if not ext:
        ext = ".jpg"
        
    upload_filename = f"{unique_id}{ext}"
    result_filename = f"{unique_id}_result{ext}"
    
    upload_path = os.path.join(UPLOAD_DIR, upload_filename)
    result_path = os.path.join(RESULT_DIR, result_filename)
    
    # Save uploaded file
    with open(upload_path, "wb") as buffer:
        buffer.write(await image.read())
        
    # Run prediction
    # Assuming threshold is 0.5 as requested
    start_time = time.time()
    result = predictor.predict(upload_path, result_path, confidence_threshold=0.5)
    print(f"Prediction time: {time.time() - start_time:.2f}s")
    
    # Add result image URL to result
    result["result_image"] = f"/results/{result_filename}"
    
    return result

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
