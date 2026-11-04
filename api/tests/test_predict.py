from unittest.mock import Mock

import pytest
from loguru import logger

from app import main


@pytest.fixture
def prediction_logs(client):
    records = []
    sink_id = logger.add(
        lambda message: records.append(message.record),
        filter=lambda record: record["name"] == main.__name__,
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def test_predict_valid_review(client, prediction_logs):
    text = "Chambre propre, personnel adorable."
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"sentiment", "scores_etoiles", "latence_ms"}
    assert data["sentiment"] in {"négatif", "neutre", "positif"}
    scores = data["scores_etoiles"]
    assert set(scores) == {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}
    assert all(0 <= score <= 1 for score in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)
    assert isinstance(data["latence_ms"], (int, float))
    assert data["latence_ms"] >= 0

    assert len(prediction_logs) == 1
    record = prediction_logs[0]
    assert record["level"].name == "INFO"
    assert f"texte={text!r}" in record["message"]
    assert f"sentiment={data['sentiment']}" in record["message"]
    assert "statut=200" in record["message"]
    assert float(record["message"].rsplit("duree_ms=", 1)[1]) >= 0


@pytest.mark.parametrize(
    "text",
    ["", " \t\n ", "x" * 2001],
    ids=["vide", "espaces-seuls", "2001-caracteres"],
)
def test_predict_rejects_invalid_text(client, prediction_logs, text):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "text"] for error in response.json()["detail"]
    )
    assert len(prediction_logs) == 1
    record = prediction_logs[0]
    assert record["level"].name == "WARNING"
    assert f"texte={text[:80]!r} | sentiment=indisponible" in record["message"]
    assert "statut=422" in record["message"]


@pytest.mark.parametrize("length", [1, 2000])
def test_predict_accepts_length_boundaries(client, length):
    response = client.post("/predict", json={"text": "a" * length})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "text",
    [
        "é" * 79,
        "é" * 80,
        "é" * 81,
        "Très bon séjour.\n" + "é" * 100 + "MARQUEUR_NON_LOGUE",
    ],
    ids=["79-caracteres", "80-caracteres", "81-caracteres", "multiligne"],
)
def test_predict_logs_only_80_characters(
    client, prediction_logs, monkeypatch, text
):
    prediction = Mock(wraps=main.predict_sentiment)
    monkeypatch.setattr(main, "predict_sentiment", prediction)

    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    prediction.assert_called_once_with(main.state["classifier"], text)
    assert len(prediction_logs) == 1
    message = prediction_logs[0]["message"]
    assert f"texte={text[:80]!r} | sentiment=" in message
    assert "\n" not in message
    assert "\r" not in message
    assert "MARQUEUR_NON_LOGUE" not in message


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": None}, {"text": 123}],
    ids=["champ-absent", "null", "nombre"],
)
def test_predict_logs_invalid_body_without_text(client, prediction_logs, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert len(prediction_logs) == 1
    assert "texte=None | sentiment=indisponible" in prediction_logs[0]["message"]
    assert "statut=422" in prediction_logs[0]["message"]


def test_predict_logs_invalid_json(client, prediction_logs):
    response = client.post(
        "/predict", content="{", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert len(prediction_logs) == 1
    assert "texte=None | sentiment=indisponible" in prediction_logs[0]["message"]
    assert "statut=422" in prediction_logs[0]["message"]


def test_predict_logs_unavailable_model(client, prediction_logs, monkeypatch):
    monkeypatch.delitem(main.state, "classifier")

    response = client.post("/predict", json={"text": "Très bon séjour."})

    assert response.status_code == 503
    assert len(prediction_logs) == 1
    assert prediction_logs[0]["level"].name == "ERROR"
    assert "sentiment=indisponible" in prediction_logs[0]["message"]
    assert "statut=503" in prediction_logs[0]["message"]


def test_predict_logs_model_exception(client, prediction_logs, monkeypatch):
    def failing_prediction(classifier, text):
        raise RuntimeError("Panne du modèle pour le test")

    monkeypatch.setattr(main, "predict_sentiment", failing_prediction)

    with pytest.raises(RuntimeError, match="Panne du modèle pour le test"):
        client.post("/predict", json={"text": "Très bon séjour."})

    assert len(prediction_logs) == 1
    assert prediction_logs[0]["level"].name == "ERROR"
    assert "sentiment=indisponible" in prediction_logs[0]["message"]
    assert "statut=500" in prediction_logs[0]["message"]


def test_health_does_not_create_prediction_logs(client, prediction_logs):
    assert client.get("/health").status_code == 200
    assert prediction_logs == []


@pytest.mark.parametrize(
    "text, expected_sentiment",
    [
        pytest.param(
            "Séjour parfait ! Piscine magnifique, personnel aux petits soins et "
            "petit-déjeuner copieux. On reviendra l'été prochain avec les enfants.",
            "positif",
            id="csv-1-positif",
        ),
        pytest.param(
            "Chambre sale à notre arrivée, cheveux dans la douche et serviettes "
            "tachées. Inadmissible pour ce prix.",
            "négatif",
            id="csv-2-negatif",
        ),
        pytest.param(
            "Très bien situé, à deux pas de la gare. Personnel réactif, "
            "literie confortable.",
            "positif",
            id="csv-7-positif",
        ),
        pytest.param(
            "Petit-déjeuner à 18 € pour du pain industriel et un café tiède. À éviter.",
            "négatif",
            id="csv-15-negatif",
        ),
    ],
)
def test_predict_unambiguous_csv_reviews(client, text, expected_sentiment):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json()["sentiment"] == expected_sentiment
