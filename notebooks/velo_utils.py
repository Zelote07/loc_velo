"""Fonctions partagées par les notebooks : chargement du dataset Bike Sharing (day.csv),
préparation des données et tracé des courbes d'apprentissage."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "day.csv"
TARGET = "cnt"
CATEGORICAL = ["season", "mnth", "weekday", "weathersit"]

# Les colonnes casual + registered = cnt : les garder serait une fuite de données.
DROPPED = ["instant", "dteday", "casual", "registered"]

# Palette catégorielle (ordre fixe, jamais recyclé) + style sobre pour tous les graphiques.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=PALETTE),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.8,
    "lines.linewidth": 2,
    "figure.dpi": 100,
})


def load_raw():
    """Charge le CSV brut."""
    return pd.read_csv(DATA_PATH, parse_dates=["dteday"])


def prepare(df=None, target_scale=1000.0):
    """Supprime les colonnes inutiles, encode les variables catégorielles en one-hot
    et exprime la cible en milliers de locations (pour garder des valeurs proches de 1)."""
    if df is None:
        df = load_raw()
    df = df.drop(columns=DROPPED)
    df = pd.get_dummies(df, columns=CATEGORICAL, prefix=CATEGORICAL, dtype="float32")
    df = df.astype("float32")
    df[TARGET] = df[TARGET] / target_scale
    return df


def split(df, frac=0.8, seed=0):
    """Découpage aléatoire train/test, comme dans le tutoriel Keras de régression."""
    train = df.sample(frac=frac, random_state=seed)
    test = df.drop(train.index)
    return (
        train.drop(columns=TARGET),
        test.drop(columns=TARGET),
        train[TARGET],
        test[TARGET],
    )


def load_splits(frac=0.8, seed=0):
    """Raccourci : données préparées et découpées -> X_train, X_test, y_train, y_test."""
    return split(prepare(), frac=frac, seed=seed)


def plot_history(histories, metric="loss", title="Courbes d'apprentissage",
                 ylabel="MAE (milliers de locations)", ylim=None, ax=None):
    """Trace `metric` sur le train (pointillés) et la validation (trait plein)
    pour un ou plusieurs entraînements.

    `histories` : un objet History Keras ou un dict {nom: History}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    if not isinstance(histories, dict):
        ax.plot(histories.history[metric], ls="--", lw=1.5, color=PALETTE[0], label="train")
        if "val_" + metric in histories.history:
            ax.plot(histories.history["val_" + metric], color=PALETTE[0], label="validation")
    else:
        for i, (name, h) in enumerate(histories.items()):
            color = PALETTE[i % len(PALETTE)]
            ax.plot(h.history[metric], ls="--", lw=1.5, color=color, alpha=0.8)
            if "val_" + metric in h.history:
                ax.plot(h.history["val_" + metric], color=color, label=name)
        ax.plot([], [], ls="--", color="gray", label="train (pointillés)")
        ax.plot([], [], color="gray", label="validation (plein)")
    ax.set(xlabel="époque", ylabel=ylabel, title=title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()
    return ax
