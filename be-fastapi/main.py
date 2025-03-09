import io

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

from utils.models import ModelManager
from utils.image_processing import preprocess_image, save_image_to_memory

# Initialize model manager
model_manager = ModelManager()

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Perform object detection on an uploaded image."""

    # Read image file
    contents = await file.read()
    image = preprocess_image(contents)

    # Perform inference
    results = model_manager.predict_chest_xray(image)
    detections = model_manager.extract_detections(results)

    return {"detections": detections}


@app.post('/yolo_predict')
async def yolo_predict(file: UploadFile = File(...)):
    """Process an image using YOLO and return the annotated image."""
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    results = model_manager.predict_chest_xray(img)

    # Convert result to PIL Image
    img_with_boxes = results[0].plot()
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Save to buffer
    img_buffer = save_image_to_memory(img_with_boxes)

    return StreamingResponse(img_buffer, media_type="image/jpeg")


@app.post('/acne-yolo-predict')
async def acne_yolo_predict(file: UploadFile = File(...)):
    """Process an image using YOLOv12 acne detection model and return the annotated image."""
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    results = model_manager.predict_acne(img)

    # Convert result to PIL Image
    img_with_boxes = results[0].plot()
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Save to buffer
    img_buffer = save_image_to_memory(img_with_boxes)

    return StreamingResponse(img_buffer, media_type="image/jpeg")


# Run API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
