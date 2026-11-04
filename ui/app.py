"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os

import streamlit as st
from api_client import get, post
import httpx

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration de l'interface Streamlit
st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

# Zone de saisie de l'avis client
text = st.text_area(
    "Avis client",
    height=160,
    placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
)

def display_status(result):
    if result['sentiment'] == "négatif":
        st.markdown("🔴 **sentiment négatif**")
    elif result == "neutre":
        st.markdown("🟠 **sentiment neutre**")
    elif result['sentiment'] == "positif":
        st.markdown("🟢 **sentiment positif**")
    else:
        st.warning("Statut inconnu")


if st.button("Analyser", type="primary"):
    if not text.strip():
        st.warning("Veuillez saisir un texte.")
    else:
        try:
            result = post(
                "/predict",
                payload={"text": text}
            )

            display_status(result)

        except httpx.HTTPStatusError as e:
                st.error(f"Erreur HTTP : {e.response.status_code}")
        
        except httpx.RequestError as e:
                st.error(f"Erreur de connexion : {e}")

#if st.button("Analyser", type="primary"):
#    st.info("L'interface n'est pas encore branchée à l'API.")
