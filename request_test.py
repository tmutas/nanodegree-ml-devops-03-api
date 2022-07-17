import requests

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

url = "https://nanodegree-ml-devops-03-api.herokuapp.com"
r = requests.post(
    f"{url}/infer",
    json=data
)

print("Response: ", r.text)
print("Statuscode: ", r.status_code)
