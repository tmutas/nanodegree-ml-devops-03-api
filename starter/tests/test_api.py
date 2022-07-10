"""Testing requests to API"""
from fastapi.testclient import TestClient
from starter.api.app import app

client = TestClient(app)


def test_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"data": "This is an API to interact with the Census data model"}