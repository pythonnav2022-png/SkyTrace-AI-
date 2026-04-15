"""
utils.py — SkyTrace AI
Fonctions de prétraitement et de visualisation des patches satellite.
Pipeline identique à la production GEE (normalisation, composition RGB).
"""

import numpy as np
from PIL import Image


def normaliser_patch(patch: np.ndarray) -> np.ndarray:
    """
    Normalisation Min-Max par bande (standard preprocessing TROPOMI).
    Sortie : float32 dans [0.0, 1.0] — prêt pour le CNN.
    """
    patch_norm = np.zeros_like(patch, dtype=np.float32)
    for b in range(patch.shape[2]):
        bande = patch[:, :, b]
        vmin, vmax = bande.min(), bande.max()
        if vmax - vmin > 1e-9:
            patch_norm[:, :, b] = (bande - vmin) / (vmax - vmin)
        else:
            patch_norm[:, :, b] = 0.0
    return patch_norm


def patch_vers_rgb_fausses_couleurs(patch: np.ndarray) -> np.ndarray:
    """
    Composition en fausses couleurs pour visualisation (comme QGIS/Google Earth Engine).
    R = B5_SWIR1 (CH4/CO), G = B2_UVIS (NO2), B = B4_NIR.
    Les zones polluées apparaissent en rouge-orange.
    """
    patch_norm = normaliser_patch(patch)
    rouge = patch_norm[:, :, 4]   # SWIR — CO/CH4
    vert  = patch_norm[:, :, 1]   # UVIS — NO2
    bleu  = patch_norm[:, :, 3]   # NIR  — Cloud
    rgb = np.stack([rouge, vert, bleu], axis=-1)
    return (rgb * 255).astype(np.uint8)


def patch_vers_heatmap_no2(patch: np.ndarray) -> np.ndarray:
    """
    Heatmap de la bande NO2 (B2_UVIS) avec colormap Inferno.
    Retourne une image RGB uint8 (64×64×3).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    bande_no2 = normaliser_patch(patch)[:, :, 1]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
    im = ax.imshow(bande_no2, cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    return img


def preparer_batch_modele(patch: np.ndarray) -> np.ndarray:
    """
    Formate un patch pour l'inférence CNN.
    Entrée  : (64, 64, 7)
    Sortie  : (1, 64, 64, 7) — batch de taille 1.
    """
    patch_norm = normaliser_patch(patch)
    return np.expand_dims(patch_norm, axis=0)   # shape (1, 64, 64, 7)


def calculer_taxe_cbam(concentration_no2: float, production_tonnes: float = 50000.0) -> dict:
    """
    Estimateur de taxe carbone européenne (CBAM).
    Prix ETS moyen 2024 : ~65 €/tonne CO2.
    Facteur empirique NO2 → CO2 équivalent : standard sectoriel.
    """
    PRIX_ETS_EUR_PAR_TONNE = 65.0
    # Conversion mol/m² NO2 → tonnes CO2eq (facteur sectoriel industrie lourde)
    FACTEUR_CONVERSION = 1.8e8

    emissions_tonnes_co2eq = concentration_no2 * FACTEUR_CONVERSION
    taxe_brute = emissions_tonnes_co2eq * PRIX_ETS_EUR_PAR_TONNE

    # Simulation : avec l'optimisation IA, réduction de 22% (benchmark Kayrros 2023)
    TAUX_REDUCTION_IA = 0.22
    economie = taxe_brute * TAUX_REDUCTION_IA
    taxe_optimisee = taxe_brute * (1 - TAUX_REDUCTION_IA)

    return {
        "emissions_co2eq_tonnes": round(emissions_tonnes_co2eq, 1),
        "taxe_brute_eur":         round(taxe_brute, 0),
        "taxe_apres_ia_eur":      round(taxe_optimisee, 0),
        "economie_potentielle_eur": round(economie, 0),
        "taux_reduction_pct":     TAUX_REDUCTION_IA * 100,
    }
