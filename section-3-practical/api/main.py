import uvicorn
import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .routes import prediction_router

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
@app.get("/")
async def root():
    return RedirectResponse(url='/docs')

app.include_router(prediction_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)