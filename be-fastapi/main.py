import io

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.responses import StreamingResponse

# Load YOLOv8 model
yolo_model = YOLO("./models/chest-xray/best.pt")

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Perform object detection on an uploaded image."""

    # Read image file
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format

    # Convert PIL image to NumPy array (H, W, C)
    image = np.array(image)

    # Ensure the image is in the correct format (uint8 and has 3 color channels)
    if image.ndim == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Perform inference
    results = yolo_model.predict(image, imgsz=640)

    # Extract bounding boxes
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()
            })

    return {"detections": detections}


def save_image_to_memory(image: Image.Image, fmt='JPEG'):
    """Save image to a BytesIO memory buffer."""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=fmt)
    img_buffer.seek(0)
    return img_buffer


@app.post('/yolo_predict')
async def yolo_predict(file: UploadFile = File(...)):
    """Process an image using YOLO and return the annotated image."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Perform inference
    results = yolo_model.predict(img, imgsz=640)

    # Convert result to PIL Image (handling OpenCV output properly)
    img_with_boxes = results[0].plot()
    img_with_boxes = Image.fromarray(img_with_boxes)

    # Save to buffer
    img_buffer = save_image_to_memory(img_with_boxes)

    return StreamingResponse(img_buffer, media_type="image/jpeg")


# Run API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
