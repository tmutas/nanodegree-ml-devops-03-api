"""A simple test api using FastAPI"""
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def hello_world() -> dict:
    return {"data": "This is API to interact with the Census data model"}
