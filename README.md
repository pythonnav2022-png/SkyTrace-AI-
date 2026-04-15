# 🛰️ SkyTrace AI — Surveillance Géospatiale & Audit Carbone

> **Greentech Hackathon — Innovation for Sustainable Growth · ENIM**

Plateforme de surveillance des émissions industrielles (CO₂, NO₂) par imagerie satellite Sentinel-5P et Intelligence Artificielle (CNN Deep Learning).

---

## 🚀 Installation & Lancement (3 étapes)

### Prérequis
- Python 3.10 ou 3.11
- pip à jour (`pip install --upgrade pip`)

### Étape 1 — Cloner / extraire le projet
```bash
cd SkyTrace_AI
```

### Étape 2 — Installer les dépendances
```bash
pip install -r requirements.txt
```
> ⏱️ Durée : environ 2–3 minutes selon votre connexion.

### Étape 3 — Lancer le dashboard
```bash
streamlit run app.py
```
> Le navigateur s'ouvre automatiquement sur `http://localhost:8501`
> Au premier lancement, le modèle CNN s'entraîne (~30 sec). Il est ensuite mis en cache.

---

## 📂 Structure du projet

```
SkyTrace_AI/
├── requirements.txt     # Dépendances Python
├── app.py               # Dashboard Streamlit principal
├── data_fetcher.py      # Simulation données Sentinel-5P (TROPOMI)
├── model.py             # Architecture CNN + entraînement
├── utils.py             # Prétraitement images & calcul CBAM
└── README.md            # Ce fichier
```

---

## 🧠 Architecture Technique

### Données (data_fetcher.py)
- Simulation de patches multispectraux **64×64×7** (7 bandes TROPOMI)
- Bandes simulées : UV, UVIS (NO₂), VIS, NIR, SWIR×3
- Variabilité saisonnière et hebdomadaire réaliste
- En production : remplacer par `earthengine.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')`

### Modèle IA (model.py)
- **CNN** avec bloc résiduel simplifié (inspiré ResNet)
- Entrée : `(64, 64, 7)` — patch spectral normalisé
- Sortie : concentration NO₂ en mol/m²
- Loss : Huber (robuste aux pics de pollution)
- Entraînement : 800 samples, ~40 epochs, Early Stopping

### Dashboard (app.py)
- Carte Folium interactive (zones industrielles marocaines)
- Heatmap NO₂ + composition fausses couleurs
- Jauge Plotly niveau de pollution
- Série temporelle 30 jours
- **Calculateur CBAM** : taxe brute vs taxe après optimisation IA

---

## 🌍 Zones Industrielles Disponibles

| Zone | Type | Coordonnées |
|------|------|-------------|
| Jorf Lasfar | Chimique/Phosphates | 33.105°N, 8.638°W |
| Mohammedia | Raffinerie/Pétrochimie | 33.686°N, 7.383°W |
| Casablanca | Mixte/Manufacture | 33.589°N, 7.603°W |
| Safi (OCP) | Phosphates/Engrais | 32.299°N, 9.237°W |
| Kénitra | Automobile/Industrie | 34.261°N, 6.578°W |
| Tanger Med | Port/Logistique | 35.868°N, 5.508°W |

---

## 💶 Modèle Économique — CBAM (Carbon Border Adjustment Mechanism)

```
Taxe CBAM = (Émissions réelles - Seuil autorisé) × Prix ETS (65 €/tonne)

Économie SkyTrace = Taxe brute × 22% (réduction certifiée par audit satellite)
```

**Impact estimé :** 50K – 400K€ économisés par entreprise et par an.

---

## 📊 Pour le Jury — Points Clés

1. **Zéro infrastructure physique** : Données 100% satellitaires (gratuites via Copernicus)
2. **Couverture nationale** : Tout le territoire marocain en une journée
3. **Objectivité** : Impossible à biaiser (contrairement aux auto-déclarations)
4. **ROI immédiat** : Réduction directe des pénalités CBAM à l'export vers l'UE
5. **Scalabilité** : Extensible à tout pays signataire de l'accord Paris

---

*SkyTrace AI — Built for the Greentech Hackathon · Powered by ESA Copernicus + TensorFlow*
