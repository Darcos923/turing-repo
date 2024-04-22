from fastapi import (
    APIRouter,
    File,
    HTTPException
)
from .services import json_prediction, image_from_bytes
import logging

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router
prediction_router = APIRouter(
    prefix="/predicting",
    tags=["predicting"],
)

@prediction_router.post("/json_prediction")
async def img_to_json(file: bytes = File(...)):
    """
    **Object Detection from an image.**

    **Args:**
        - **file (bytes)**: The image file in bytes format.
    **Returns:**
        - **dict**: JSON format containing the Objects Detections.
    """
    try:
        image = image_from_bytes(file)
        json_result = json_prediction(image)
        return json_result
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")
