import os
import httpx

API_URL = os.getenv(
    "API_URL",
    "http://api:8000"
)

TIMEOUT = 30.0


def get_client():
    return httpx.Client(
        base_url=API_URL,
        timeout=TIMEOUT
    )

def get(endpoint: str, params=None):
    with httpx.Client(base_url=API_URL, timeout=30.0) as client:
        response = client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()


def post(endpoint: str, payload=None):
    with httpx.Client(base_url=API_URL, timeout=30.0) as client:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()