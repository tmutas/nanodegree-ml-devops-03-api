"""Testing requests to API"""
from fastapi.testclient import TestClient
from starter.api.app import app

client = TestClient(app)


def test_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "data": "This is API to interact with the Census data model"
    }


"""
@pytest.mark.parametrize("val,query", vals)
def test_get_items_default(val, query):
    r = client.get(f"/items/{val}")
    assert r.status_code == 200
    assert r.json() == {"data" : f"Gotten {val} with query param 1"}

@pytest.mark.parametrize("val,query", vals)
def test_get_items_query(val,query):
    r = client.get(f"/items/{val}?query={query}")
    assert r.status_code == 200
    assert r.json() == {"data" : f"Gotten {val} with query param {query}"}

@pytest.mark.parametrize("path_name", wrong_paths)
def test_wrong_path(path_name):
    r = client.get(path_name)
    assert r.status_code == 404
"""
