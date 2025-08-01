from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import json
import gdown
import os
import zipfile

#url = "https://drive.google.com/uc?id=1LFEOEliakObh1GAhTFPGuEHR7vYPwrTE"
#output = "modelo_chatbot.zip"

url = "https://drive.google.com/uc?id=1EBgyASxpQG1cVdmtFF-ELsPi_Z8FZdHM"
output = "Chatbot_Multilabel.zip"

# Descagar el modelo del chatbot si no existe
if not os.path.exists("Chatbot_Multilabel"):
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("Chatbot_Multilabel")

with open("respuestas.json", encoding="utf-8") as f:
    respuestas = json.load(f)

with open("id2label.json", encoding="utf-8") as f:
    id2label = json.load(f)


MODEL_PATH = os.path.abspath("Chatbot_Multilabel")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # !Cambiarlo cuando lo levante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    mensaje: str

@app.post("/api/chatbot/")
def responder(pregunta: Pregunta):
    try:
        inputs = tokenizer(pregunta.mensaje, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.sigmoid(outputs.logits)[0]
        
        etiquetas = [
            id2label[str(i)] for i,p in enumerate(probs) if probs[i] > 0.5
        ]

        print("Etiquetas predichas:", etiquetas)

        
        raza = None
        temas = []

        for etiqueta in etiquetas:
            if etiqueta in respuestas:
                raza = etiqueta
            elif etiqueta in ["cuidados","enfermedad","info_general"]:
                temas.append(etiqueta)

        if not raza:
            return {"respuestas": [{"categoria": "desconocido", "respuesta": "No pude identificar la raza mencionada."}]}
        
        if not temas:
            temas = ["info_general"]

        resultados = []

        for tema in temas:
            respuesta = respuestas.get(raza, {}).get(tema, f"No tengo información sobre {tema} para {raza}")
            resultados.append({"categoria": f"{raza}_{tema}", "respuesta": respuesta})

        return {"respuestas": resultados}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
