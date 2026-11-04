import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    # scope="session" : le modèle n'est chargé qu'une fois pour toute la suite de tests
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            main, "LOG_PATH", tmp_path_factory.mktemp("logs") / "api.log"
        )
        with TestClient(main.app) as test_client:
            yield test_client
