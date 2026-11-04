"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class ReviewIn(BaseModel):
    """Un avis client à analyser : non vide (espaces ignorés), 2 000 caractères maximum."""

    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
        Field(description="Texte de l'avis client", examples=["Chambre propre, personnel adorable."]),
    ]


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Literal["négatif", "neutre", "positif"]
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float
