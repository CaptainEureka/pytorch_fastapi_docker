import shutil
import os
import logging
import sys

from fastapi import FastAPI, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic.main import BaseModel
from typing import Optional


logger = logging.getLogger("FFLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

#########################
from PIL import Image
import torch, torchvision
from torchinfo import summary
import torch.nn as nn
from torchvision import transforms

def load_model():
    logger.info(f"Model on {device}")

    logger.info(f"Instantiating Model")
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )
    logger.info(f"Loading Model")

    model.load_state_dict(torch.load(f"model/ckpt_densenet121_catdog.pth", map_location=device))
    logger.info(f"Model succesfully loaded on {device}")
    return model

logger.info("Check GPU availability")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = load_model()

def image_to_tensor(img_path):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])
    img.close()
    return transform(img)

## ============ API ============ ##
class Prediction(BaseModel):
    prediction: dict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def get_model(request: Request, file: Optional[str] = None, prediction: Optional[str] = None):
    return templates.TemplateResponse("model.html", 
        {
            "request": request,
            "file": file,
            "prediction": prediction,
        }
    )

@app.get("/summary", response_class=HTMLResponse)
def get_model_summary(request: Request):
    model_summary = [f"{line}" for line in summary(model).__repr__().split("\n")];
    return templates.TemplateResponse(
        "model-summary.html", 
        {
            "request": request, 
            "model_summary": model_summary
        }
    )

@app.post("/upload")
async def upload_image(request: Request, image: UploadFile):   
    destination_filename = image.filename
    logger.info(f"Trying to save image: {image.filename}")
    try:
        # Try to save the image
        write_image(image, f"static/{destination_filename}")
        logger.info(f"Image saved to static/{destination_filename}")
        await image.close()
    except:
        logger.info("An error occured saving the image")

    url = app.url_path_for("get_model")
    return RedirectResponse(
        url=f"{url}?file={destination_filename}",
        status_code=status.HTTP_303_SEE_OTHER
    )

@app.get("/predict")
async def get_model_prediction(file: str, response_type: Optional[str] = 'html'):

    response = {
        "prediction": {
            "class": None
        }
    }

    try:
        logger.info("Converting image to tensor")
        img_tensor = image_to_tensor(f"static/{file}")
        logger.info(f"Image successfully converted to tensor")
    except TypeError as e:
        logger.info(f"An error occured in PyTorch: {e}")

    logger.info(f"Sending the image tensor to device: {device}")
    input = img_tensor.to(device)

    output = model.eval()(input.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)
    prediction_map = {
        0: "Cat",
        1: "Dog"
    }
    response['prediction']['class'] = prediction_map.get([p.item() for p in prediction][0])
    logger.info(response)

    url = app.url_path_for("get_model")
    r = {
        'html': RedirectResponse(
                    url=f"{url}?file={file}?&prediction={response['prediction']['class']}",
                    status_code=status.HTTP_302_FOUND
                ),
        'json': response
    }
    return r.get(response_type, response)

def write_image(image: UploadFile, path: str):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"filename": image.filename}

def cleanup(path: str):
    os.remove(path)