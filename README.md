# Experiment: Deploying containerized ML Applications with FastAPI and Docker

<div align="center"> <a href=https://www.tensorflow.org/>PyTorch</a> - <a href=https://fastapi.tiangolo.com/>FastAPI</a> - <a href=https://www.docker.com/>Docker</a></div>

---

<br>

> **NOTE:** This Repo is experimental and is for personal learning goals only.

## Introduction

I wanted to study the methods for deploying Machine Learning applications in a containerized manner and also play around with PyTorch as I'm more familiar with [TensorFlow](https://www.tensorflow.org/). To that end I developed this repository which contains a `Dockerfile` (which specifies the PyTorch docker image) and a `docker-compose.yaml` file which handles the deployment of the API and ML application. For development locally I also wanted to experiment with Poetry package management. The initial idea was to carry this over to the Docker containers but this deemed to be a bit trickier than I expected with Poetry. Thus, the `requirements.txt` to install dependencies in the container.

### The application

Currently the application simply consists of a `main.py` which houses the central API built using FastAPI. I chose FastAPI because it's well *Fast*. It allows for asynchronous functions this is apparently nice for production applications (the exact reason for this escapes me for now..). Anywho, back to the app.

#### The model
The app implements a pretrained model in `ckpt_densenet121_catdog.pth` (found on [Kaggle](https://www.kaggle.com/code/jaeboklee/pytorch-cat-vs-dog/data)). 
> I fully intend to implement my own model in the future. This was just to get the endpoint up and running and experiment with FastAPI `FileUpload`.

#### The API endpoints
So far I have implemented two significant endpoints which will eventually be linked from a central endpoint `/model`:
- `/model/summary` Which provides an overview of the model architecture,
- `/model/inference` Where an image/file can be uploaded for inference.

Currently the API only provides JSON responses but I hope to expand this to `HTMLResponse` class provided by `FastAPI` (also nice because this can be incorporated with `Jinja2` for a nice frontend).

## Running the app

**Dependencies**:
- Docker
- Docker Compose

**Commands**
```bash
# on cpu
docker compose up

# or on gpu (with cuda)
docker compose --file docker-compose.gpu.yaml up 
```

## Testing

Currently testing is rudementary. Just call the `test.py` 

```bash
python tests/test.py

# or with poetry
poetry run python tests/test.py
```

## Improvements
Currently the stack consists of a singular `api` container which serves both the frontend and does the heavy 'inference'. I believe it would be advantageous to expand these two components and have a seperate container for inference (a model enpoint if you will) and a frontend (webapp).

> **N.B** The model doesn't enjoy `.png`'s probably due to the alpha channel which is not properly accounted for in the `img_to_tensor` transform.