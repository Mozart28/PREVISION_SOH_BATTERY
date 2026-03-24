# 🔋 Prédiction du State of Health (SoH) des batteries lithium-ion par LSTM

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras)
![R2](https://img.shields.io/badge/R²-0.9547-brightgreen)
![MAE](https://img.shields.io/badge/MAE-1.03%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Modèle de deep learning pour estimer le State of Health (SoH) de batteries lithium-ion à partir de séquences temporelles multivariées issues de cycles de décharge.

---

## 📋 Table des matières

- [Contexte](#-contexte)
- [Dataset](#-dataset)
- [Pipeline](#-pipeline)
- [Architecture](#-architecture-lstm)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Questions de réflexion](#-questions-de-réflexion)

---

## 🎯 Contexte

Le **State of Health (SoH)** est l'indicateur de référence pour quantifier la dégradation d'une batterie lithium-ion. Il représente le rapport entre la capacité actuelle et la capacité nominale initiale (exprimé en %).

```
SoH = 100% → Batterie neuve
SoH <  80% → Fin de vie conventionnelle
```

**Enjeux industriels :**
- ⚡ Véhicules électriques — estimation de l'autonomie réelle
- 🔧 Maintenance prédictive — anticiper les défaillances
- ♻️ Second life des batteries — évaluation objective avant réutilisation
- 🌍 IoT & stockage solaire — monitoring à distance

---

## 📊 Dataset

**Source :** [NASA Battery Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

| Caractéristique | Valeur |
|---|---|
| Observations | 29 180 |
| Batteries | 24 (B0005 → B0048) |
| Cycles max | 197 |
| Timesteps / cycle | 20 (fixe) |
| Valeurs manquantes | 0 |

### Variables

| Variable | Type | Description |
|---|---|---|
| `Voltage_measured` | float64 | Tension aux bornes (V) |
| `Current_measured` | float64 | Courant de décharge (A) |
| `Temperature_measured` | float64 | Température interne (°C) |
| `SoC` | float64 | State of Charge — niveau de charge (%) |
| `cycle_number` | int64 | Numéro du cycle |
| `battery_id` | string | Identifiant batterie |
| `SoH` | float64 | **Cible à prédire (%)** |

---

## 🔄 Pipeline

Le pipeline suit les **8 étapes officielles** de modélisation :

```
1. Analyse et vérification de la qualité des données
        ↓ Détection outliers (IQR) — conservés (physiquement réels)
2. Sélection des variables pertinentes
        ↓ Corrélation Pearson & Spearman + cycle_norm par batterie
3. Découpage en fenêtres temporelles glissantes
        ↓ Window=20, Stride=1 → ~16 000 séquences
4. Association fenêtre → SoH cible
        ↓ SoH du dernier timestep de la fenêtre
5. Split stratifié train / val / test
        ↓ 5 bins de SoH × 70/15/15% → toute la gamme couverte
6. Apprentissage LSTM
        ↓ Huber loss + EarlyStopping + ReduceLROnPlateau
7. Prédiction sur données inconnues
        ↓ Dénormalisation + suivi par batterie
8. Évaluation des performances
        ↓ MAE, RMSE, MAPE, R²
```

### 🔑 Transformation clé — `cycle_norm`

Le `cycle_number` brut est remplacé par `cycle_norm`, une normalisation **intra-batterie** :

```python
cycle_norm = (cycle - cycle_min) / (cycle_max - cycle_min)
```

**Pourquoi ?** Sans cette transformation, le modèle confondrait une batterie en fin de vie avec 11 cycles (B0047) avec une batterie en début de vie ayant 196 cycles (B0036). `cycle_norm` encode le **pourcentage de vie écoulé**, directement corrélé au SoH.

---

## 🧠 Architecture LSTM

```
Input  : (batch, 20, 5)          ← 20 timesteps × 5 features
   ↓
LSTM(128, return_sequences=True)  ← Extraction temporelle
Dropout(0.3) + BatchNorm
   ↓
LSTM(64, return_sequences=False)  ← Compression
Dropout(0.3) + BatchNorm
   ↓
Dense(32, ReLU)                   ← Représentation non-linéaire
Dropout(0.2)
   ↓
Dense(1, linéaire)                ← Prédiction SoH
```

**Hyperparamètres d'entraînement :**

| Paramètre | Valeur |
|---|---|
| Optimizer | Adam (lr₀ = 1e-3) |
| Loss | Huber (δ = 0.5) |
| Batch size | 64 |
| Epochs max | 100 |
| Early Stopping | patience = 15 |
| ReduceLROnPlateau | factor=0.5, patience=7 |

> **Pourquoi Huber ?** La MSE pousse le modèle à prédire la moyenne du SoH pour minimiser la perte globale (*collapse vers la moyenne*). Huber est robuste aux outliers tout en restant différentiable en 0, offrant un apprentissage plus équilibré.

---

## 📈 Résultats

| Métrique | Valeur | Interprétation |
|---|---|---|
| **MAE** | **1.03 %** | Erreur absolue moyenne < 1 point de SoH |
| **RMSE** | **1.49 %** | Robuste aux pics d'erreur |
| **MAPE** | **1.25 %** | Erreur relative moyenne |
| **R²** | **0.9547** | 96.4% de la variance expliquée |

**Diagnostic de variance :**
```
Std y_pred  : 6.726   Std y_true  : 6.990
Ratio       : 0.962   (idéal = 1.0)
Biais       : +0.05%  (quasi nul)
```

---

## ⚙️ Installation

```bash
# 1. Cloner le repo
git clone https://github.com/username/battery-soh-lstm.git
cd battery-soh-lstm

# 2. Créer l'environnement conda
conda create -n soh_env python=3.11 -y
conda activate soh_env

# 3. Installer les dépendances
conda install numpy pandas matplotlib scikit-learn joblib ipykernel -y
pip install tensorflow seaborn scipy

# 4. Installer le kernel Jupyter
python -m ipykernel install --user --name=soh_env --display-name "soh_env (Python 3.11)"
```

---

## 🚀 Utilisation

```bash
# Lancer Jupyter depuis le dossier projet
conda activate soh_env
jupyter notebook
```

Ouvrir `battery_soh_FINAL.ipynb` et sélectionner le kernel **soh_env (Python 3.11)**.

### Inférence sur un nouveau cycle

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Charger le modèle et les scalers
model    = load_model('battery_soh_lstm_FINAL.keras')
scaler_X = joblib.load('scaler_X_FINAL.pkl')
scaler_y = joblib.load('scaler_y_FINAL.pkl')

# Préparer une séquence (20 timesteps × 5 features)
# features = [Voltage, Current, Temperature, SoC, cycle_norm]
sequence  = np.array([...])                          # (20, 5)
seq_norm  = scaler_X.transform(sequence)             # normalisation
seq_input = seq_norm.reshape(1, 20, 5)               # (1, 20, 5)

# Prédiction
pred_norm = model.predict(seq_input)
pred_soh  = scaler_y.inverse_transform(pred_norm)[0][0]
print(f'SoH prédit : {pred_soh:.2f}%')
```

---

## 📁 Structure du projet

```
battery-soh-lstm/
│
├── 📓 battery_soh_FINAL.ipynb      ← Pipeline complet (8 étapes)
├── 🌐 battery_soh.html             ← Notebook rendu (sans Jupyter)
├── 📄 Rapport_SoH_LSTM_FINAL.docx  ← Rapport Word professionnel
│
├── 📦 battery_soh_lstm_FINAL.keras ← Modèle sauvegardé
├── 📦 scaler_X_FINAL.pkl           ← Scaler features
├── 📦 scaler_y_FINAL.pkl           ← Scaler cible
│
├── 📊 battery_health_dataset.csv   ← Dataset NASA
└── 📖 README.md
```

---

## ❓ Questions de réflexion

<details>
<summary><b>1. Pourquoi le SoC est-il une variable clé pour estimer le SoH ?</b></summary>

Le SoC mesure le niveau de charge instantané. Sa **dynamique de décharge** — la façon dont il évolue sur les 20 timesteps d'un cycle — change de manière caractéristique avec le vieillissement. Une batterie dégradée se décharge plus vite, sa courbe SoC(t) présente une pente plus accentuée. Le LSTM ne lit pas le SoC comme une valeur instantanée mais comme une **trajectoire temporelle** dont la forme encode implicitement l'état de santé.

</details>

<details>
<summary><b>2. Quel intérêt de découper un cycle en plusieurs fenêtres glissantes ?</b></summary>

Sans découpage : 1 459 séquences (une par cycle) — insuffisant pour un LSTM.
Avec fenêtre glissante (stride=1) : ~16 000 séquences — volume ×11.
Chaque fenêtre capture une portion différente de la dynamique de décharge, exposant le modèle à la diversité des patterns électriques caractéristiques de chaque niveau de SoH.

</details>

<details>
<summary><b>3. Que se passerait-il si la fenêtre était trop courte ou trop longue ?</b></summary>

**Trop courte** (ex: 3 timesteps) → contexte insuffisant, signal dominé par le bruit → underfitting sévère.

**Trop longue** (ex: 60 timesteps = 3 cycles) → la fenêtre chevauche plusieurs cycles avec des SoH différents → signal contradictoire → dégradation des performances.

**Optimale = 20 timesteps** → un cycle complet : cohérence temporelle + richesse informationnelle.

</details>

<details>
<summary><b>4. Quels risques de biais si les cycles sont mal répartis train/test ?</b></summary>

Un split chronologique naïf place les batteries neuves en train et les batteries dégradées en test. Le modèle est évalué sur des régimes jamais vus → R² négatif possible.

Le **split stratifié par bins de SoH** corrige ce biais en garantissant que toute la gamme de dégradation (70% → 122%) est représentée proportionnellement dans les trois ensembles.

</details>

<details>
<summary><b>5. Dans quels cas industriels ce modèle est-il pertinent ?</b></summary>

- **Véhicules électriques** — estimation de l'autonomie réelle, détection des cellules défaillantes
- **Maintenance prédictive** — anticipation des défaillances dans les systèmes de stockage
- **IoT & lampadaires solaires** — monitoring à distance sans technicien
- **Second life des batteries** — évaluation objective avant réutilisation
- **BMS embarqué** — inférence TensorFlow Lite directement dans le système de gestion

</details>

---

## 🔭 Perspectives

- **Feature engineering** — résistance interne (R = ΔV/ΔI), énergie de décharge (∫V·I dt), pente dV/dt
- **Architectures alternatives** — GRU, Transformer, CNN-LSTM hybride
- **Déploiement embarqué** — conversion TensorFlow Lite pour BMS temps réel
- **Généralisation multi-chimie** — validation sur batteries LFP, NMC

---

## 📚 Référence

NASA Prognostics Center of Excellence — *Battery Data Set*
> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

---

<div align="center">
  <sub>Projet Deep Learning — Prédiction SoH batteries lithium-ion | 2024</sub>
</div>
