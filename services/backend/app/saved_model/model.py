from os import listdir, rename
from os.path import join, isfile
from PIL import Image
import torch, torchvision
import torch.nn as nn
from torchvision import transforms

def load_model(device: str = "cpu"):

    if device == "cuda":
        assert torch.cuda.is_available() == True, "CUDA was requested but not available"

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )

    model.load_state_dict(torch.load("ckpt_densenet121_catdog.pth",
                                     map_location=device))

    return model


def predict(model, image):
    
    p_map = lambda s: {
        0: "Cat",
        1: "Dog"
    }.get(s)

    input = image_to_tensor(image)
    output = model.eval()(input.unsqueeze(0))
    prediction = (torch.argmax(output, dim=1))
    
    response = { "data":
        { "prediction": p_map(prediction[0].item()) } }
    return response


def image_to_tensor(img):
    img = Image.open(img)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ColorJitter(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
            transforms.ToTensor()
        ]
    )

    response = transform(img)
    img.close()
    return response