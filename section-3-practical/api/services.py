from ultralytics import YOLO
import logging
import io
from PIL import Image
from fastapi import File
import json

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

def image_from_bytes(binary_image: bytes) -> Image:
    """
    Convert image from bytes to PIL RGB format

    **Args:**
        - **binary_image (bytes):** The binary representation of the image

    **Returns:**
        - **PIL.Image:** The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def json_prediction(file: bytes = File(...)) -> dict: #  = File(...
    """
    Processes an image file using a prediction model and returns the results in JSON format.

    **Args:**
        **file (bytes):** The byte array of the image file to be processed.

    Returns:
        **dict:** A dictionary with the key 'results' containing a list of prediction results, each a dictionary.
    """
    results = model(file)
    for result in results:
        return json.loads(result.tojson())