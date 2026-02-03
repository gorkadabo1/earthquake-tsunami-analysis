# 🌊 Global Earthquake-Tsunami Analysis

A comprehensive data science study analyzing historical earthquake data to identify patterns linked to tsunami generation. The project combines exploratory analysis with statistical inference and machine learning to answer key research questions about earthquake-tsunami relationships.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Methods & Techniques](#methods--techniques)
- [Key Findings](#key-findings)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

## Overview

This project analyzes 782 earthquake events (2001-2022) to understand the factors that differentiate tsunami-generating earthquakes from those that don't produce tsunamis. The analysis employs a dual approach: **classification** (predicting tsunami occurrence) and **regression** (predicting earthquake magnitude).

The study reveals that while individual seismic characteristics like magnitude and depth show limited predictive power alone, combining multiple variables through machine learning achieves **94% AUC** in tsunami prediction.

## Research Questions

1. **What characteristics differentiate earthquakes that produce tsunamis from those that do not?**
2. **Are the strength or shallowness of earthquakes related to tsunami generation?**
3. **Are there natural patterns or distinct groups of earthquakes based on their seismic and geographic characteristics?**
4. **To what extent do seismic characteristics allow for the prediction of earthquake magnitude?**
5. **Is it possible to predict whether an earthquake will generate a tsunami based on its seismic and geographic characteristics?**

## Dataset

| Variable | Type | Description |
|----------|------|-------------|
| `magnitude` | Float | Richter scale energy released (**Regression target**) |
| `depth` | Float | Hypocenter depth in kilometers |
| `latitude` / `longitude` | Float | Epicenter coordinates (WGS84) |
| `cdi` | Integer | Community Decimal Intensity (0-9) |
| `mmi` | Integer | Modified Mercalli Intensity (1-9) |
| `sig` | Integer | Global significance/impact score |
| `nst` | Integer | Number of recording seismic stations |
| `dmin` | Float | Angular distance to nearest station |
| `gap` | Float | Azimuthal coverage gap |
| `tsunami` | Binary | Tsunami occurrence indicator (**Classification target**) |

**Dataset characteristics:**
- 782 samples, 13 variables
- No null values or duplicates
- Class distribution: 61.1% No Tsunami / 38.9% Tsunami

## Methods & Techniques

### Exploratory Data Analysis
- Distribution analysis with density histograms
- Geographic visualization with GeoPandas and Contextily
- Temporal trend analysis (2001-2022)
- Correlation heatmaps (Pearson)

### Statistical Testing
- **Normality tests:** Shapiro-Wilk
- **Group comparisons:** Mann-Whitney U test, Chi-square
- **Effect size:** Cliff's Delta
- **Multiple comparisons:** Benjamini-Hochberg FDR correction
- **Correlation analysis:** Pearson and Spearman coefficients

### Dimensionality Reduction
- **PCA:** 2D and 3D principal component analysis
- Explained variance analysis (9 components for 90% variance)

### Unsupervised Learning
- **K-Means Clustering** with Elbow method and Silhouette score optimization
- Geographic pattern identification in seismic zones

### Regression Models (Magnitude Prediction)
| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| Linear Regression | 0.340 | 0.17 | 0.27 |
| Ridge (α=1.0) | 0.341 | 0.16 | — |
| Lasso (α=0.01) | 0.341 | 0.17 | — |
| **Random Forest** | **0.303** | **0.34** | — |

### Classification Models (Tsunami Prediction)
| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression (L1) | 82.0% | 0.88 |
| Logistic Regression (L2) | 82.0% | 0.88 |
| Random Forest | 90.0% | 0.94 |
| **XGBoost** | **90.4%** | **0.939** |

## Key Findings

### 1. Differentiating Characteristics
Tsunami-generating earthquakes are primarily distinguished by **location and measurement context**, not by strength:
- Greater distance from recording stations (`dmin` ↑)
- Fewer stations tracking them (`nst` ↓)
- Lower latitudes (closer to equator)

### 2. Magnitude & Depth Relationship
Neither magnitude nor depth showed statistically significant differences between tsunami and non-tsunami events (Chi-square p > 0.05). This suggests the dataset lacks key physical variables (fault type, coastal distance, bathymetry).

### 3. Natural Clustering Patterns
K-Means clustering revealed groups explained primarily by:
- Geographic location
- Seismic network coverage
- Concentration along Pacific Ring of Fire

### 4. Magnitude Prediction Limitations
Linear models achieved only R² ≈ 0.17, indicating available variables explain very little magnitude variance. Random Forest improved to R² ≈ 0.34 but remains limited.

### 5. Tsunami Prediction Success
Classification models achieved excellent performance:
- **XGBoost: 90.4% accuracy, 0.939 AUC**
- Demonstrates that combining all variables enables high-precision prediction

## Results Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    TSUNAMI CLASSIFICATION                    │
├─────────────────────────────────────────────────────────────┤
│  Best Model: XGBoost                                        │
│  Accuracy: 90.4%  |  AUC: 0.939                            │
│  Key predictors: dmin, nst, Year, latitude, gap            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   MAGNITUDE REGRESSION                       │
├─────────────────────────────────────────────────────────────┤
│  Best Model: Random Forest                                  │
│  R²: 0.34  |  RMSE: 0.303                                  │
│  Key predictors: depth, latitude, cdi, longitude           │
│  Conclusion: Limited predictive capacity                    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
earthquake-tsunami-analysis/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── earthquake_data_tsunami.csv
│
├── notebooks/
│   └── earthquake_tsunami_analysis.ipynb
│
└── src/
    └── earthquake_tsunami_analysis.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/gorkadabo1/earthquake-tsunami-analysis.git
cd earthquake-tsunami-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Analysis

```python
# Using Jupyter Notebook
jupyter notebook notebooks/earthquake_tsunami_analysis.ipynb

# Or run the Python script directly
python src/earthquake_tsunami_analysis.py
```

### Quick Start Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/earthquake_data_tsunami.csv')

# Prepare features
features = ['depth', 'cdi', 'mmi', 'nst', 'dmin', 'gap', 'latitude', 'longitude']
X = df[features].dropna()
y = df.loc[X.index, 'tsunami']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

## Technologies Used

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, GeoPandas, Contextily
- **Statistical Analysis:** SciPy, Statsmodels
- **Machine Learning:** Scikit-learn, XGBoost
- **Dimensionality Reduction:** PCA (Scikit-learn)
- **Geographic Analysis:** GeoPandas, Shapely

## Author

**Gorka Dabó**

*This project was developed as part of the Data Science course in the Master's program in Data Analysis in Engineering (MADI) at Universidad de Navarra/Tecnun.*

## License

This project is licensed under the MIT License