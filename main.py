from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, json
import gdown
import zipfile
import os

url = "https://drive.google.com/uc?id=1LFEOEliakObh1GAhTFPGuEHR7vYPwrTE"
output = "modelo_chatbot.zip"

# Descagar el modelo del chatbot si no existe
if not os.path.exists("modelo_chatbot"):
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("modelo_chatbot")

labels = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "american_shorthair",
    "australian_shepherd",
    "basset_hound",
    "beagle",
    "border_collie",
    "boxer",
    "chartreux",
    "chihuahua",
    "cornish_rex",
    "dachshund",
    "doberman",
    "english_cocker_spaniel",
    "english_setter",
    "french_bulldog",
    "german_shorthaired",
    "golden_retriever",
    "great_pyrenees",
    "havanese",
    "himalayan",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "norwegian_forest_cat",
    "persian",
    "pitbull",
    "pomeranian",
    "poodle",
    "pug",
    "rottweiler",
    "saint_bernard",
    "samoyed",
    "scottish_fold",
    "scottish_terrier",
    "shiba_inu",
    "siamese",
    "siberian_husky",
    "sintoma_convulsiones",
    "sintoma_diarrea",
    "sintoma_falta_apetito",
    "sintoma_moco_nasal",
    "sintoma_ojos_rojos",
    "sintoma_vomito",
    "sphynx",
    "staffordshire_bull_terrier",
    "turkish_angora",
    "wheaten_terrier",
    "yorkshire_terrier"
]

id2label = {str(i): label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}


MODEL_PATH = os.path.abspath("modelo_chatbot")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    id2label=id2label,
    label2id=label2id
    )
model.eval()

with open("respuestas.json", encoding="utf-8") as f:
    respuestas = json.load(f)

app = FastAPI()

class Pregunta(BaseModel):
    mensaje: str

@app.post("/api/chatbot/")
def responder(pregunta: Pregunta):
    try:
        inputs = tokenizer(pregunta.mensaje, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = int(outputs.logits.argmax(dim=-1))
            categoria = id2label.get(str(pred_id), "desconocido")
        

        respuesta = respuestas.get(categoria, "Lo siento, no tengo información sobre eso.")
        return {"categoria": categoria, "respuesta": respuesta}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
