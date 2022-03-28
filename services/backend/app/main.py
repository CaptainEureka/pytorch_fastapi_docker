import logging
import sys
from PIL import Image

from fastapi import FastAPI, Request, UploadFile, File, status

from utils.helper import write_image


logger = logging.getLogger("FFLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

#########################
from saved_model.model import load_model, predict

model = load_model()

## ============ API ============ ##
app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    return {"status": "OK"}


# @app.post("/upload", status_code=status.HTTP_201_CREATED)
# async def upload_image(file: UploadFile = File(...)):
#     img = Image.open(file.file)
#     logger.info(f"Trying to save image: {file.filename}")
#     try:
#         # Try to save the image
#         destination = f"/images/{file.filename}"
#         write_image(img, destination)
#         logger.info(f"Image saved to {destination}")
#         await file.close()
#     except:
#         logger.info("An error occured saving the image")

#     return {
#         "msg": "Image uploaded",
#         "file_path": destination
#     }

@app.post("/predict", status_code=status.HTTP_200_OK)
async def get_model_prediction(file: UploadFile = File(...)):
    # img = Image.open(file.file)
    prediction = predict(model, file.file)
    return prediction
