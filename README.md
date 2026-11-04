# Escale Atlantique — Sentiment des avis clients

> Squelette de départ du brief **M0-B2** (FastIA). Ce README est à **réécrire** pendant le brief (tâche 8).

## Le contexte

Le service qualité d'Escale Atlantique reçoit plusieurs centaines d'avis clients par mois et ne parvient plus à tous les lire. Ce projet classe chaque avis en **négatif, neutre ou positif**, pour traiter les clients mécontents en priorité.

Le modèle fourni, [`cmarkea/distilcamembert-base-sentiment`](https://huggingface.co/cmarkea/distilcamembert-base-sentiment), prédit une note de **1 à 5 étoiles** : la conversion en 3 classes est à faire.

## Structure

```
docker-compose.yml     # 2 services (api, ui), réseau interne, volumes, healthcheck
.env.example           # variables d'environnement, à copier en .env
api/                   # service FastAPI
  app/main.py          #   lifespan (chargement du modèle), /health, /info, /predict à compléter
  app/inference.py     #   chargement du pipeline + predict_sentiment à compléter
  app/schemas.py       #   schémas Pydantic (entrée à contraindre)
  tests/               #   2 tests fournis
  Dockerfile
ui/                    # service Streamlit
  app.py               #   interface, pas encore branchée à l'API
  Dockerfile
data/
  avis_annotes.csv     # 30 avis annotés par l'équipe qualité
models/                # cache du modèle Hugging Face (volume, non versionné)
logs/                  # logs de l'API (volume, non versionné, créé au lancement)
```

## Démarrage rapide

```bash
cp .env.example .env
docker compose up --build
```

⏳ Le premier lancement télécharge le modèle (≈ 270 Mo) : compte quelques minutes. Les lancements suivants sont rapides, et tout fonctionne hors ligne.

- API : http://localhost:8000/health et http://localhost:8000/docs
- Interface : http://localhost:8501
- État des services : `docker compose ps` (l'API doit être *healthy*)
- Tests : `docker compose exec api pytest -v`
- Arrêt : `docker compose down`

## Journalisation et tests de prédiction

Chaque requête `POST /predict` produit une seule entrée applicative Loguru :

```text
POST /predict | texte='Chambre propre, personnel adorable.' | sentiment=positif | statut=200 | duree_ms=162.06
```

- Seuls les 80 premiers caractères du texte sont enregistrés ; le modèle reçoit
  toujours le texte validé complet. Les retours à la ligne sont échappés pour
  conserver une seule ligne physique.
- Un succès est journalisé au niveau `INFO`, une entrée refusée (422) au niveau
  `WARNING`, et une erreur serveur au niveau `ERROR`. Sans prédiction, le sentiment
  est `indisponible` ; sans texte exploitable, l'aperçu vaut `None`.
- Le middleware mesure le traitement de la requête jusqu'à la préparation de la
  réponse, validation incluse. Le champ JSON `latence_ms` mesure seulement
  l'inférence et la conversion des scores.
- Les entrées sont écrites dans la console et dans `/app/logs/api.log`, monté
  dans `logs/api.log` sur l'hôte. Le fichier tourne à 5 Mo ; les archives sont
  compressées et la rétention est configurée à 7 jours. Les logs d'accès Uvicorn
  sont distincts des entrées applicatives.

**Tronquer n'anonymise pas** : un nom ou un courriel peut déjà apparaître dans les
80 premiers caractères. Limiter les accès aux logs et ne pas les versionner.
Pour des données réelles, prévoir une minimisation supplémentaire, par exemple
ne plus enregistrer de texte libre.

Après une modification du code ou des tests, reconstruire l'image depuis la
racine du projet :

```bash
docker compose up -d --build api
docker compose exec api pytest -v
docker compose logs --tail=50 api
```

Les tests utilisent `TestClient` dans un contexte `with` : le modèle réel est
chargé une seule fois par session de tests. Le fichier de logs des tests est
isolé dans un dossier temporaire pour ne pas partager la rotation du fichier du
serveur en cours d'exécution.

La suite vérifie le contrat d'une réponse valide, les limites de 1 et 2 000
caractères, les refus 422, la journalisation unique et tronquée, ainsi que les
erreurs du modèle. Un test paramétré reprend exactement les avis positifs ou
négatifs explicites d'identifiants 1, 2, 7 et 15 du CSV. Ces quelques cas constituent
des tests de non-régression, pas une mesure de la qualité générale du modèle.
Les textes et annotations sont copiés dans les paramètres de test, car le contexte
de construction `./api` n'inclut pas le dossier `data`. Une modification de ces
avis dans le CSV nécessite donc de mettre à jour ces paramètres.

## Les avis annotés

`data/avis_annotes.csv` contient 30 avis clients annotés à la main par l'équipe qualité. Ils sont censés être anonymisés… vérifie quand même.

| Colonne | Contenu |
|---|---|
| `id` | identifiant de l'avis |
| `hotel` | `City Hotel` ou `Resort Hotel` |
| `source` | Booking, Google, TripAdvisor ou Questionnaire (fin de séjour) |
| `texte` | l'avis, tel que rédigé par le client |
| `sentiment_attendu` | l'annotation de l'équipe qualité |

Consigne d'annotation donnée à l'équipe qualité :

- **négatif** : l'insatisfaction domine, ou le client signale un incident qui appelle une réponse ;
- **neutre** : avis factuel, ou mitigé sans que l'un des deux côtés domine ;
- **positif** : la satisfaction domine, même avec un petit bémol.
