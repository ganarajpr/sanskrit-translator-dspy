import dspy
from fastapi import FastAPI, HTTPException
from modules.TranslationModule import TranslationModule
from pydantic import BaseModel
from setup import get_openai_config


app = FastAPI()
