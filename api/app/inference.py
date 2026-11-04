"""Chargement du modèle Hugging Face et inférence."""

from time import perf_counter

from transformers import pipeline


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache).
    
    Args:
        model_name (str): Nom du modèle Hugging Face à charger ("cmarkea/distilcamembert-base-sentiment").

    Returns:
        pipeline: Pipeline de classification de texte.
        
    ie.
    docker compose exec api python
    >>> from app.inference import load_pipeline
    >>> clf = load_pipeline("cmarkea/distilcamembert-base-sentiment")
    >>> clf("Chambre propre et personnel adorable.")
    """
    return pipeline("text-classification", model=model_name, top_k=None)


# Conversion des 5 étoiles du modèle vers les 3 classes métier
STAR_TO_CLASS = {
    "1 star": "négatif", "2 stars": "négatif",
    "3 stars": "neutre",
    "4 stars": "positif", "5 stars": "positif",
}


def stars_to_sentiment(scores: dict[str, float]) -> str:
    """Convertit les scores des 5 étoiles en une des 3 classes métier.

    Choix : somme des probabilités par groupe (1-2 = négatif, 3 = neutre, 4-5 = positif),
    puis on garde le groupe le plus probable.
    Justification : on utilise toute la distribution, pas seulement le pic. Un avis
    où le modèle hésite entre 1 et 2 étoiles reste négatif, ce qui limite le risque
    le plus coûteux pour l'équipe qualité : rater un client mécontent.
    """
    if set(scores) != set(STAR_TO_CLASS):
        raise ValueError(f"Labels inattendus : {sorted(scores)}")
    totals = {"négatif": 0.0, "neutre": 0.0, "positif": 0.0}
    for label, score in scores.items():
        totals[STAR_TO_CLASS[label]] += score
    return max(totals, key=totals.get)  # en cas d'égalité stricte : négatif d'abord


def predict_sentiment(classifier, text: str) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis."""
    start = perf_counter()
    # truncation : au-delà de 512 tokens, le modèle planterait (500 sur un avis long)
    output = classifier(text, truncation=True, max_length=512)
    if output and isinstance(output[0], list):  # [[{...}]] pour un texte seul
        output = output[0]
    scores = {item["label"]: float(item["score"]) for item in output}
    return {
        "sentiment": stars_to_sentiment(scores),
        "scores_etoiles": scores,  # scores bruts des 5 étoiles, pour la traçabilité
        "latence_ms": round((perf_counter() - start) * 1000, 2),
    }