import requests
import os

URL = "http://localhost:8080/model/inference"
WORKDIR = os.environ['PWD']

def test_upload_file(filepath):
    file = {'file': open(filepath, 'rb')}
    # headers = {'Content-ype': 'image/jpeg'}
    resp = requests.post(url = URL, files = file)
    print(resp.json())

if __name__ == "__main__":
    for file in ['dog1.jpg', 'sleepy-cat.jpg']:
        test_upload_file(f"{WORKDIR}/{file}")