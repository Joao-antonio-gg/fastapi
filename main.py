from typing import Union
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from ultralytics import YOLO
import numpy
import os

app = FastAPI()

# Configuração do CORS

model = YOLO("treinamento/best.pt")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/image/")
async def upload_image(file: UploadFile):
    if file is None:
        return {"error": "No file"}

    try:
        image = Image.open(file.file)
        result = model(image)
        name = result[0].names
        probs = result[0].probs.data.numpy()
        result_f = (name[numpy.argmax(probs)])

        # bloqueios = [FileResponse("img/cadeado.jpg", media_type="image/jpg"), FileResponse("img/chave.jpg", media_type="image/jpg")]
        # if result_f != 'chave':
        #     bloqueios.append (FileResponse("img/"+ result_f + ".jpg", media_type="image")) 
        bloqueios = ['cadeado', 'etiqueta']
        if result_f != 'chave':
            bloqueios.append(result_f)

        return {"result": result_f, "bloqueios": bloqueios}
    

    except Exception as e:
        return {"error": str(e)}