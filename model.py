"""
model.py — SkyTrace AI
CNN de régression pour prédire la concentration NO2 (mol/m²)
à partir d'un patch multispectral Sentinel-5P (64×64×7).
Architecture inspirée de ResNet simplifié — adapté à la démo hackathon.
"""

import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Silencer les warnings TensorFlow

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

IMG_H, IMG_W, N_BANDS = 64, 64, 7
MODELE_PATH = "skytrace_model.keras"


def construire_modele() -> tf.keras.Model:
    """
    CNN custom inspiré ResNet léger.
    Entrée  : (batch, 64, 64, 7) — 7 bandes TROPOMI normalisées
    Sortie  : (batch, 1) — concentration NO2 en mol/m² (valeur normalisée)
    """
    entree = layers.Input(shape=(IMG_H, IMG_W, N_BANDS), name="patch_spectral")

    # Bloc 1 — Extraction bas niveau
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(entree)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)                    # → 32×32

    # Bloc 2 — Extraction motifs de pollution
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)                    # → 16×16

    # Bloc 3 — Résidu simplifié
    residuel = layers.Conv2D(128, 1, padding="same")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residuel])
    x = layers.MaxPooling2D(2)(x)                    # → 8×8

    # Tête de régression
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    sortie = layers.Dense(1, activation="sigmoid", name="concentration_no2")(x)

    modele = models.Model(entree, sortie, name="SkyTrace_CNN")
    modele.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",           # Robuste aux valeurs extrêmes (pics pollution)
        metrics=["mae"]
    )
    return modele


def generer_donnees_entrainement(n_samples: int = 800):
    """
    Génère des paires (patch, concentration) pour l'entraînement rapide.
    Simule la diversité réelle : zones propres, modérées, très polluées.
    """
    from data_fetcher import _generer_patch_spectral, _convertir_reflectance_en_concentration
    from utils import normaliser_patch

    X, y = [], []
    niveaux = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]  # facteurs de capacité

    for i in range(n_samples):
        capacite = niveaux[i % len(niveaux)] + np.random.uniform(-0.1, 0.1)
        patch = _generer_patch_spectral(max(0.1, capacite), seed=i * 7 + 42)
        conc  = _convertir_reflectance_en_concentration(patch)
        X.append(normaliser_patch(patch))
        y.append(conc)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Normaliser les cibles dans [0,1] pour sigmoid
    y_max = y.max() + 1e-9
    y_norm = y / y_max

    return X, y_norm, y_max


def entrainer_ou_charger() -> tuple:
    """
    Charge le modèle si déjà sauvegardé, sinon entraîne rapidement (< 30 sec).
    Retourne (modele, y_max) — y_max permet la dénormalisation des prédictions.
    """
    Y_MAX_PATH = "skytrace_ymax.npy"

    if os.path.exists(MODELE_PATH) and os.path.exists(Y_MAX_PATH):
        modele = tf.keras.models.load_model(MODELE_PATH)
        y_max = float(np.load(Y_MAX_PATH))
        return modele, y_max

    print("⏳ Entraînement du modèle SkyTrace AI (environ 20–40 sec)...")
    X, y_norm, y_max = generer_donnees_entrainement(n_samples=800)

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y_norm[:split]
    X_val,   y_val   = X[split:], y_norm[split:]

    modele = construire_modele()
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
    ]
    modele.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=cb,
        verbose=0,
    )
    modele.save(MODELE_PATH)
    np.save(Y_MAX_PATH, np.array([y_max]))
    print("✅ Modèle entraîné et sauvegardé.")
    return modele, y_max


def predire(modele: tf.keras.Model, patch: np.ndarray, y_max: float) -> float:
    """
    Prédit la concentration NO2 (mol/m²) à partir d'un patch normalisé.
    """
    from utils import preparer_batch_modele
    batch = preparer_batch_modele(patch)
    pred_norm = modele.predict(batch, verbose=0)[0][0]
    return float(pred_norm) * y_max
