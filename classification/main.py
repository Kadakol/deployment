from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

import uvicorn

import os

from classifier.classification_response import ClassificationResponse
from classifier.alex_net_cat_dog_classifier import AlexNetCatDogClassifier

# Creating FastAPI instance
def start_application():
    app = FastAPI(title='HTML Example', version='1.0')
    templates = Jinja2Templates(directory='templates')
    alex_net_cat_dog_classifier = AlexNetCatDogClassifier()
    return templates, app, alex_net_cat_dog_classifier

templates, app, alex_net_cat_dog_classifier = start_application()


@app.get("/classify")
async def classify(request: Request):
    return templates.TemplateResponse("classification.html", {"request":request})


@app.post('/classify')
async def predict(request: Request, file: UploadFile = File(...), response_class=HTMLResponse):
    response = await alex_net_cat_dog_classifier.classify(file)
    return templates.TemplateResponse("classification-response.html", {"request":request, 'response':response})

