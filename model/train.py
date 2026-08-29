import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Add, Multiply, Activation, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, auc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION "EXPERT"
# ─────────────────────────────────────────────
CFG = {
    "classes": ["Inondation", "Sécheresse", "Ouragan"],
    "label_cols": ["flood_risk", "drought_risk", "hurricane_risk"],
    "feature_cols": [
        "temperature_max", "cumul_pluie_24h", "humidite_relative_air",
        "vent_soutenu", "pression_atmospherique", "anomalie_niveau_mer",
        "jours_sans_pluie", "humidite_des_sols"
    ],
    "data_path": str(Path(__file__).resolve().parent.parent / "data" / "climate_data_v5_expert.csv"),
    "save_dir": str(Path(__file__).resolve().parent.parent / "saved_models"),
    "plot_dir": str(Path(__file__).resolve().parent.parent / "model" / "plots"),
    "n_samples": 25000,
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-3,
    "seed": 42,
    "n_folds": 5
}

tf.random.set_seed(CFG["seed"])
np.random.seed(CFG["seed"])

for d in [CFG["save_dir"], CFG["plot_dir"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING "EXPERT"
# ─────────────────────────────────────────────
def engineer_features(df):
    d = df.copy()
    d["wind_log"] = np.log1p(d["vent_soutenu"])
    d["press_def"] = 1013.25 - d["pression_atmospherique"]
    d["is_extreme_wind"] = (d["vent_soutenu"] >= 119).astype(int)
    d["sol_sature"] = (d["humidite_des_sols"] > 85).astype(int)

    feats = CFG["feature_cols"] + ["wind_log", "press_def", "is_extreme_wind", "sol_sature"]
    return d[feats].values.astype("float32"), feats

# ─────────────────────────────────────────────
# 3. ARCHITECTURE DU MODÈLE
# ─────────────────────────────────────────────
def build_model(n_feat):
    inputs = Input(shape=(n_feat,))
    x = Dense(256, activation="swish")(inputs)
    x = BatchNormalization()(x)

    res1 = Dense(128)(x)
    h1 = Dense(128, activation="swish")(x)
    h1 = BatchNormalization()(h1)
    h1 = Dense(128)(h1)
    x = Add()([res1, h1])
    x = Activation("swish")(x)

    # Mécanisme d'attention
    att = Dense(128 // 8, activation="relu")(x)
    att = Dense(128, activation="sigmoid")(att)
    x = Multiply()([x, att])

    out_flood = Dense(32, activation="swish")(x)
    out_flood = Dense(1, activation="sigmoid", name="Inondation")(out_flood)

    out_drought = Dense(32, activation="swish")(x)
    out_drought = Dense(1, activation="sigmoid", name="Secheresse")(out_drought)

    out_hurricane = Dense(32, activation="swish")(x)
    out_hurricane = Dense(1, activation="sigmoid", name="Ouragan")(out_hurricane)

    model = Model(inputs, Concatenate()([out_flood, out_drought, out_hurricane]))
    model.compile(optimizer=Adam(CFG["lr"]), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model

# ─────────────────────────────────────────────
# 4. GÉNÉRATION DES GRAPHIQUES DE PERFORMANCE
# ─────────────────────────────────────────────
def save_plots(history, model, X_test, y_test, thresholds, classes):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title('Évolution de la Perte (Loss)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC', color='green')
    plt.plot(history.history['val_auc'], label='Val AUC', color='red', linestyle='--')
    plt.title('Évolution de l\'AUC')
    plt.legend()
    plt.savefig(f"{CFG['plot_dir']}/training_curves.png", bbox_inches='tight')
    plt.close()

    y_prob = model.predict(X_test, verbose=0)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green']
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.title('Courbes ROC Multi-Label')
    plt.legend(loc="lower right")
    plt.savefig(f"{CFG['plot_dir']}/roc_curves.png", bbox_inches='tight')
    plt.close()

    y_pred = (y_prob > np.array(thresholds)).astype(int)
    mcm = multilabel_confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, cls) in enumerate(zip(axes, classes)):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 14})
        ax.set_title(f'{cls}\n(Seuil: {thresholds[i]:.2f})', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(f"{CFG['plot_dir']}/confusion_matrices.png", bbox_inches='tight')
    plt.close()

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT AVEC K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
def train_and_calibrate():
    print(f"Chargement des données depuis {CFG['data_path']}...")
    df = pd.read_csv(CFG["data_path"])
    X_raw, feat_names = engineer_features(df)
    y = df[CFG["label_cols"]].values

    print(f"🔄 Lancement de la Validation Croisée en {CFG['n_folds']} Folds...")

    kf = KFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG["seed"])

    fold_accuracies = []
    best_auc = 0
    best_model = None
    best_scaler = None
    best_thresholds = None
    best_history = None
    X_val_best, y_val_best = None, None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
        X_train, X_val = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = build_model(X_train_scaled.shape[1])

        history = model.fit(
            X_train_scaled, y_train,
            epochs=CFG["epochs"],
            batch_size=CFG["batch_size"],
            validation_data=(X_val_scaled, y_val),
            callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
            verbose=0
        )

        y_prob = model.predict(X_val_scaled, verbose=0)
        thresholds = []
        for i in range(3):
            fpr, tpr, thr = roc_curve(y_val[:, i], y_prob[:, i])
            best_thr = thr[np.argmax(tpr - fpr)]
            thresholds.append(float(np.clip(best_thr, 0.20, 0.60)))

        y_pred = (y_prob > np.array(thresholds)).astype(int)
        accuracy = np.mean(y_pred == y_val)
        fold_accuracies.append(accuracy)

        val_auc = max(history.history['val_auc'])
        print(f"✅ Fold {fold + 1}/{CFG['n_folds']} terminé - Accuracy : {accuracy * 100:.2f}% | AUC : {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_scaler = scaler
            best_thresholds = thresholds
            best_history = history
            X_val_best, y_val_best = X_val_scaled, y_val

    mean_accuracy = np.mean(fold_accuracies)
    print("\n" + "="*50)
    print("🏆 RÉSULTATS FINAUX DU MODÈLE (K-FOLD)")
    print("="*50)
    print(f"👉 Précision (Accuracy) moyenne validée : {mean_accuracy * 100:.2f}%")
    print("="*50 + "\n")

    print("💾 Sauvegarde du meilleur modèle et génération des graphiques...")
    joblib.dump(best_scaler, f"{CFG['save_dir']}/scaler.pkl")

    meta = {"thresholds": best_thresholds, "features": feat_names}
    with open(f"{CFG['save_dir']}/meta.json", "w") as f: json.dump(meta, f)

    best_model.save(f"{CFG['save_dir']}/model.keras")
    save_plots(best_history, best_model, X_val_best, y_val_best, best_thresholds, CFG["classes"])

    print(f"✅ Terminé ! Tous les fichiers sont dans le dossier : {CFG['save_dir']}")
    return best_model, meta, best_scaler

if __name__ == "__main__":
    train_and_calibrate()