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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a origem do Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        bloqueios = ['cadeado', 'etiqueta']
        if result_f != 'chave':
            bloqueios.append(result_f)
        porcentagem = probs[numpy.argmax(probs)] * 100
        porcentagem = round(porcentagem, 2)

        return {"result": result_f, "bloqueios": bloqueios, "porcentagem": porcentagem}
    

    except Exception as e:
        return {"error": str(e)}