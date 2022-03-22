import shutil
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic.main import BaseModel
from typing import Optional
import logging
import sys

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
from torch.nn import Module
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
    return transform(img)

app = FastAPI()
class Data(BaseModel):
    data: dict

@app.get("/")
def get_root():
    return {
        "Hello": "World"
    }

@app.get("/test/")
def get_test(request: Optional[Data]):
    return {
        "Test": request.data
    }

@app.get("/model/summary/")
def get_model_summary():
    model_summary = ''.join([f"<div>{line}</div>" for line in summary(model).__repr__().split("\n")]);
    html_response = f"""
    <html>
        <head>
            <title>Model Summary</title>
        </head>
        <body>
            <h1>Model Summary</h1>
            {model_summary}
        </body>
    </html>
    """
    return HTMLResponse(content=html_response, status_code=200)

@app.post("/model/inference/")
async def get_model_inference(file: UploadFile):

    response = {
        "file": {
            "filename": file.filename,
            "content_type": file.content_type
        },
        "prediction": {
            "class": None
        }
    }

    destination_filename = "image.png"
    try:
        # Try to save the image
        write_image(file, destination_filename)
        logger.info("Image saved")
    except:
        logger.info("An error occured saving the image")

    try:
        img_tensor = image_to_tensor(destination_filename)
        logger.info(f"type(img_tensor): {type(img_tensor)}, img_tensor: {img_tensor}")
    except TypeError as e:
        logger.info(f"An error occured in PyTorch: {e}")

    logger.info(f"Sending the image tensor to device: {device}")
    input = img_tensor.to(device)
    logger.info(f"{type(input)}, {input.shape}")

    output = model.eval()(input.unsqueeze(0))
    logger.info(f"type(output): {type(output)}")
    prediction = torch.argmax(output, dim=1)
    prediction_map = {
        0: "Cat",
        1: "Dog"
    }
    response['prediction']['class'] = prediction_map.get([p.item() for p in prediction][0])

    logger.info(response)
    await file.close()
    cleanup(destination_filename)

    return response

def write_image(image: UploadFile, path: str):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {"filename": image.filename}

def cleanup(path: str):
    os.remove(path)