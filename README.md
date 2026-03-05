# Chemical Toxicity Prediction Using Random Forest

This repository contains an end-to-end Machine Learning pipeline to predict whether a chemical compound is `Toxic` or `NonToxic` based on 1000+ extracted molecular features.

## Project Overview
Handling datasets with a massive amount of features (1200+) relative to sample size (171 rows) often leads to the "Curse of Dimensionality." This project combats this through aggressive feature selection and ensemble modeling. 

**Steps performed:**
1. **Exploratory Data Analysis (EDA):** Analyzed class imbalances.
2. **Preprocessing:** - Removed zero-variance (constant) features.
   - Removed highly correlated features (Pearson > 0.95).
   - Applied Standard Scaling.
3. **Feature Selection:** Utilized a Tree-Based feature selection algorithm (`SelectFromModel` with Random Forest) to reduce features from 641 to the top 257 most informative predictors.
4. **Modeling:** Trained a `RandomForestClassifier` with balanced class weights to predict toxicity.

## Repository Structure
- `data/`: Contains the input CSV file.
- `src/`: Contains the main Python scripts.
- `output/`: Auto-generated folder for saving EDA and Feature Importance plots.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Arigele/toxicity-prediction.git](https://github.com/Arigele/toxicity-prediction.git)
   cd toxicity-prediction
   ```
   2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use
   `venv\Scripts\activate`
   ```
   3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   4.  **Run the pipeline:**
   Make sure your dataset is placed in the `data/` folder and named exactly `data.csv ML2.csv`.
   ```bash
   cd src
   python main.py
   ```
   ## Results & Evaluation

The final Random Forest model achieved an overall accuracy of **68.6%**. 

Due to the limited sample size (171 rows) and class imbalance (115 Non-Toxic vs. 56 Toxic), the model exhibits a bias toward the majority class. 

**Classification Report Summary:**
- **Non-Toxic (Majority):** Performed very well, achieving a **96% recall** and an **81% F1-score**.
- **Toxic (Minority):** Struggled to properly identify minority samples, achieving a **9% recall** and a **15% F1-score**, though it maintained a 50% precision when it did predict toxicity.

**Key Takeaway:** The aggressive feature selection successfully reduced the dataset from over 1,200 features to 257 without causing the model to completely collapse. However, the lack of `Toxic` instances prevented the ensemble model from learning robust boundaries for the minority class.

## Future Work
To improve the recall of the `Toxic` class in future iterations, the following steps are recommended:
1. **Data Augmentation:** Implement **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically generate more `Toxic` training samples.
2. **Algorithm Exploration:** Test Gradient Boosting frameworks like XGBoost or LightGBM, which often handle imbalanced tabular data better than standard Random Forests.
3. **Hyperparameter Tuning:** Use `GridSearchCV` to optimize the tree depth, minimum samples per leaf, and decision thresholds to favor minority class recall.