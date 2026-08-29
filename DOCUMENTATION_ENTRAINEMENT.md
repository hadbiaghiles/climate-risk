# Documentation Technique - Entraînement du Modèle Climate Risk v5

## Vue d'ensemble

Le modèle Climate Risk v5 est un réseau de neurones profond (Deep Learning) entraîné avec une validation croisée K-Fold pour prédire trois risques climatiques :
- **Inondation** (flood_risk)
- **Sécheresse** (drought_risk)  
- **Ouragan** (hurricane_risk)

---

## Utilisation Rapide (Modèle Pré-entraîné)

Le modèle est déjà entraîné et sauvegardé dans `saved_models/`. Vous pouvez l'utiliser immédiatement :

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
python app.py

# 3. Ouvrir l'interface
http://127.0.0.1:5000
```

**Fichiers du modèle déjà disponibles :**
- `saved_models/model.keras` - Architecture et poids (1.3 MB)
- `saved_models/scaler.pkl` - Paramètres de normalisation
- `saved_models/meta.json` - Seuils optimaux et features

---

## Réentraînement (Optionnel)

Si vous souhaitez réentraîner avec vos propres données :

1. **Préparer votre fichier CSV** dans `data/climate_data_v5_expert.csv`
2. **Lancer l'entraînement** : `python model/train.py`
3. **Le nouveau modèle remplacera l'ancien** dans `saved_models/`

---

## Architecture du Modèle

### 1. Configuration (CFG)

```python
CFG = {
    "classes": ["Inondation", "Sécheresse", "Ouragan"],
    "label_cols": ["flood_risk", "drought_risk", "hurricane_risk"],
    "feature_cols": [
        "temperature_max", "cumul_pluie_24h", "humidite_relative_air",
        "vent_soutenu", "pression_atmospherique", "anomalie_niveau_mer",
        "jours_sans_pluie", "humidite_des_sols"
    ],
    "n_samples": 25_000,      # Nombre d'échantillons synthétiques
    "epochs": 150,             # Nombre d'époques d'entraînement
    "batch_size": 64,          # Taille des batches
    "lr": 1e-3,               # Learning rate (Adam optimizer)
    "n_folds": 5              # Nombre de folds pour K-Fold
}
```

### 2. Features (12 au total)

#### 8 Features d'entrée :
1. `temperature_max` - Température maximale (°C)
2. `cumul_pluie_24h` - Cumul de pluie sur 24h (mm)
3. `humidite_relative_air` - Humidité relative de l'air (%)
4. `vent_soutenu` - Vitesse du vent soutenu (km/h)
5. `pression_atmospherique` - Pression atmosphérique (hPa)
6. `anomalie_niveau_mer` - Anomalie du niveau de la mer (m)
7. `jours_sans_pluie` - Nombre de jours sans pluie
8. `humidite_des_sols` - Humidité des sols (%)

#### 4 Features ingénieurées :
9. `wind_log` = log(1 + vent) - Transformation logarithmique du vent
10. `press_def` = 1013.25 - pression - Déficit de pression
11. `is_extreme_wind` = 1 si vent ≥ 118 km/h, sinon 0 - Flag vent extrême
12. `sol_sature` = 1 si humidité_sol > 85% ou pluie ≥ 100mm, sinon 0 - Flag sol saturé

### 3. Architecture du Réseau de Neurones

```
Input (12 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64)  + BatchNorm + Dropout(0.2)
    ↓
┌─────────┬──────────┬──────────┐
↓         ↓          ↓
Dense(32) Dense(32)  Dense(32)
    ↓         ↓          ↓
Sigmoid   Sigmoid    Sigmoid
    ↓         ↓          ↓
Inondation Sécheresse Ouragan
```

- **Activation** : Swish (x * sigmoid(x)) pour les couches cachées
- **Sortie** : Sigmoid (probabilités 0-1) pour chaque classe
- **Loss** : Binary Crossentropy (classification multi-label)
- **Optimizer** : Adam avec learning rate 0.001

---

## Processus d'Entraînement

### 1. Données d'Entraînement

Le modèle utilise des données réelles depuis un fichier CSV :

```python
CFG = {
    "data_path": "data/climate_data_v5_expert.csv",  # Fichier de données
    ...
}
```

**Format requis du CSV :**
- **8 colonnes features** : `temperature_max`, `cumul_pluie_24h`, `humidite_relative_air`, `vent_soutenu`, `pression_atmospherique`, `anomalie_niveau_mer`, `jours_sans_pluie`, `humidite_des_sols`
- **3 colonnes labels** : `flood_risk`, `drought_risk`, `hurricane_risk` (0 ou 1)

Si le fichier n'existe pas, vous devez le créer avec vos données historiques.

### 2. Feature Engineering

```python
def engineer_features(df):
    d = df.copy()
    d["wind_log"] = np.log1p(d["vent_soutenu"])
    d["press_def"] = 1013.25 - d["pression_atmospherique"]
    d["is_extreme_wind"] = (d["vent_soutenu"] >= 119).astype(int)
    d["sol_sature"] = (d["humidite_des_sols"] > 85).astype(int)
    return d
```

### 3. Validation Croisée K-Fold (5 folds)

```
Dataset (25 000) 
    ↓
┌─────────┬─────────┬─────────┬─────────┬─────────┐
|  Fold 1 |  Fold 2 |  Fold 3 |  Fold 4 |  Fold 5 |
| (Train) | (Val)   |         |         |         |  → Modèle 1
├─────────┼─────────┼─────────┼─────────┼─────────┤
|         | (Train) | (Val)   |         |         |  → Modèle 2
├─────────┼─────────┼─────────┼─────────┼─────────┤
|         |         | (Train) | (Val)   |         |  → Modèle 3
├─────────┼─────────┼─────────┼─────────┼─────────┤
|         |         |         | (Train) | (Val)   |  → Modèle 4
├─────────┼─────────┼─────────┼─────────┼─────────┤
| (Val)   |         |         |         | (Train) |  → Modèle 5
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

**Par fold** :
- 80% entraînement, 20% validation
- StandardScaler ajusté sur le training set uniquement
- Early stopping (patience=15) pour éviter le surapprentissage

### 4. Calcul des Seuils Optimaux

Pour chaque fold et chaque classe :
```python
fpr, tpr, thresholds = roc_curve(y_val[:, i], y_prob[:, i])
best_threshold = thresholds[np.argmax(tpr - fpr)]
threshold = clip(best_threshold, 0.20, 0.60)  # Limites [0.2, 0.6]
```

### 5. Sélection du Meilleur Modèle

Le modèle avec la meilleure AUC (Area Under Curve) sur validation est conservé.

---

## Résultats de l'Entraînement

### Performance par Fold :

| Fold | Accuracy | AUC | Seuil Inondation | Seuil Sécheresse | Seuil Ouragan |
|------|----------|-----|------------------|------------------|---------------|
| 1 | 98.13% | 0.9749 | ~0.35 | ~0.35 | ~0.35 |
| 2 | 97.92% | 0.9713 | ~0.35 | ~0.35 | ~0.35 |
| 3 | 97.89% | 0.9681 | ~0.35 | ~0.35 | ~0.35 |
| 4 | 98.01% | 0.9712 | ~0.35 | ~0.35 | ~0.35 |
| 5 | 98.07% | 0.9707 | ~0.35 | ~0.35 | ~0.35 |

**Moyenne** : **98.00% d'accuracy**

---

## Fichiers Générés

```
saved_models/
├── model.keras          (1.3 MB) - Architecture et poids du modèle
├── scaler.pkl           (903 B)  - Paramètres de normalisation StandardScaler
├── meta.json            (287 B)  - Seuils optimaux et liste des features
└── evaluation_results.json - Métriques d'évaluation

model/plots/
├── training_curves.png    - Courbes de loss et AUC pendant l'entraînement
├── roc_curves.png         - Courbes ROC pour les 3 classes
└── confusion_matrices.png - Matrices de confusion (heatmap)
```

---

## Prévention du Surapprentissage

1. **Dropout** : 0.4, 0.3, 0.2 sur les couches cachées
2. **Batch Normalization** : Stabilise l'entraînement
3. **Early Stopping** : Arrêt si pas d'amélioration après 15 époques
4. **K-Fold Cross Validation** : Évaluation robuste sur 5 splits
5. **Limitation des seuils** : Clipping entre [0.2, 0.6]

---

## Prédiction en Production

```python
def predict(data_dict):
    # 1. Extraire les 8 features d'entrée
    # 2. Calculer les 4 features ingénieurées
    # 3. Normaliser avec le scaler sauvegardé
    # 4. Passer dans le modèle
    # 5. Appliquer les seuils optimaux
    # 6. Retourner les probabilités et alertes
```

---

## Auteur et Date

- **Script** : `model/train.py`
- **Date** : Avril 2025
- **Version** : v5 Expert (12 features)
- **Framework** : TensorFlow/Keras + scikit-learn
