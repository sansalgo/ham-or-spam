from typing import Annotated
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import math
import json

app = FastAPI()

app.mount(
    "/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def load_model(filename="model.json"):
    with open(filename, "r", encoding="utf-8") as f:
        model = json.load(f)
    vocab = set(model["vocab"])
    priors = model["priors"]
    likelihood = model["likelihood"]
    return vocab, priors, likelihood


vocab, priors, likelihood = load_model("model/model.json")


def tokenize(text):
    return text.lower().split()


def predict_with_percentages(message, priors, likelihood, vocab):
    words = tokenize(message)
    log_probs = {}

    for label in ["spam", "ham"]:
        log_prob = math.log(priors[label])
        for word in words:
            if word in vocab:
                # Laplace smoothing
                log_prob += math.log(
                    likelihood[label].get(word, 1e-10))
        log_probs[label] = log_prob

    max_log = max(log_probs.values())  # for stability
    exp_probs = {label: math.exp(
        log_probs[label] - max_log) for label in log_probs}
    total = sum(exp_probs.values())
    return {
        label: round((exp_probs[label] / total) * 100, 2)
        for label in exp_probs
    }


@app.get("/")
def get_root(request: Request):
    return templates.TemplateResponse(request=request, name="app.html")


@app.post("/")
def post_root(request: Request, message: Annotated[str, Form()]):
    if not message:
        raise HTTPException(
            status_code=400, detail="Message is required")

    result = predict_with_percentages(
        message, priors, likelihood, vocab)

    return templates.TemplateResponse(request=request, name="app.html", context={
        "message": message,
        "spam": result['spam'],
        "ham": result['ham']
    })
