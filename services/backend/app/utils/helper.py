from fastapi import UploadFile
import shutil


def write_image(image: UploadFile, path: str):
    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        response = {
            "succeeded": True,
            "filename": image.filename
        }
    except:
        response = {
            "succeeded": False,
            "filename": None
        }


    return response