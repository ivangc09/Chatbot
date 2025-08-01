from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import json
import gdown
import os
import zipfile
import uvicorn

#url = "https://drive.google.com/uc?id=1LFEOEliakObh1GAhTFPGuEHR7vYPwrTE"
#output = "modelo_chatbot.zip"

url = "https://drive.google.com/uc?id=1EBgyASxpQG1cVdmtFF-ELsPi_Z8FZdHM"
zip_path = "Chatbot_Multilabel.zip"
model_dir = "Chatbot_Multilabel"

if not os.path.exists(model_dir):
    gdown.download(url, zip_path, quiet=False)

    if not zipfile.is_zipfile(zip_path):
        raise Exception("ZIP no válido")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

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

        razas_validas = set(respuestas.keys())

        
        raza = None
        temas = []

        for etiqueta in etiquetas:
            if etiqueta in ["sintoma_vomito","sintoma_falta_apetito","sintoma_moco_nasal","sintoma_ojos_rojos","sintoma_convulsiones"]:
                sintoma = etiqueta
                respuesta = respuestas[sintoma]
                return {
                    "respuestas": [
                        {
                            "categoria": sintoma,
                            "respuesta": respuesta
                        }
                    ]
                }

            if etiqueta in razas_validas:
                raza = etiqueta

            elif etiqueta in ["cuidados","enfermedad","info_general"]:
                temas.append(etiqueta)

        if not raza:
            return {
                "respuestas": [
                    {
                        "categoria": etiqueta,
                        "respuesta": f"No tengo información detallada sobre '{etiqueta}' o no pertenece a una raza conocida."
                    } for etiqueta in etiquetas
                ]
            }
        
        if not temas:
            temas = ["info_general"]

        resultados = []

        for tema in temas:
            if isinstance(respuestas.get(raza), dict):
                respuesta = respuestas[raza].get(tema, f"No tengo información sobre {tema} para {raza}")
            else:
                respuesta = f"No tengo información sobre {tema} para {raza}"

            resultados.append({
                "categoria": f"{raza}_{tema}",
                "respuesta": respuesta
            })

        return {"respuestas": resultados}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)