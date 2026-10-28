# Online News Popularity Prediction

Machine learning project that predicts the number of social media shares of online news articles, comparing linear models, Random Forest and XGBoost on the UCI *Online News Popularity* dataset.

Developed individually as a university project for the course *Software Algorithms in Automatic Control Systems*, Faculty of Technical Sciences, University of Novi Sad.

## Problem
The dataset contains statistical features of about 39,000 articles published on Mashable over two years, such as word counts, number of links, images and videos, keyword statistics, topic (LDA) scores, sentiment and publication day. The goal is to predict the number of shares, which makes this a **regression** problem.

The target variable is highly skewed: most articles have a moderate number of shares, while a small number of viral articles have extremely high values.

## Approach
1. **Data preparation** – removed articles without content and the non-informative `url` column, checked for missing values and duplicates, and removed redundant features based on a correlation matrix
2. **Feature engineering** – created interaction features, e.g. keyword popularity combined with publication on weekends, keyword popularity relative to article age, internal link ratio and media density
3. **Target transformation** – `log(1 + shares)` to reduce the effect of extreme values
4. **Modeling** – 80/20 train/test split and comparison of:
   - Linear Regression, Ridge and Lasso (with standardized features)
   - Random Forest with hyperparameter tuning (`RandomizedSearchCV`, 3-fold cross-validation)
   - XGBoost with extended feature engineering and hyperparameter tuning
   - XGBoost trained on the 50 most important features
5. **Evaluation** – MAE, MSE/RMSE and R², reported on both the log scale and the original scale of shares

## Results
| Model | R² (log scale) | R² (original scale) |
|---|---|---|
| Linear models | 0.126 | ≈ −0.005 |
| Random Forest | 0.173 | negative |
| XGBoost | ≈ 0.18 | positive (modest) |

- Tree-based models captured nonlinear relationships better than linear models.
- On the original scale, linear models and Random Forest performed worse than predicting the mean, because they systematically underestimated viral articles.
- XGBoost was the only model with a positive R² on the original scale.

**Conclusion:** article popularity follows a power-law distribution, and virality often depends on external factors that are not present in the data, such as timing, mentions by public figures or external events. This limits how accurately the number of shares can be predicted from article features alone.

Full metrics, plots and discussion are available in the [project report (Serbian)](docs/project-report-sr.pdf).

## Getting started

### Requirements
- Python 3
- Libraries listed in `requirements.txt`

```bash
git clone https://github.com/marijelaveljkovic/online-news-popularity-prediction.git
cd online-news-popularity-prediction
pip install -r requirements.txt
```

### Run
First prepare the data, which creates `OnlineNewsPopularity_Cleaned.csv`:
```bash
python Ucitavanje.py
```
Then run any of the model scripts:
```bash
python LinearRegressionRidgeLasso.py
python RandomForest.py
python GradientBoosting.py
python XGBFeatureSelection.py
```
Hyperparameter search in the Random Forest and XGBoost scripts can take several minutes. Code comments and console output are in Serbian.

## Project structure
| File | Description |
|---|---|
| `Ucitavanje.py` | Data loading, cleaning, correlation analysis and export of the cleaned dataset |
| `LinearRegressionRidgeLasso.py` | Linear Regression, Ridge and Lasso |
| `RandomForest.py` | Random Forest with hyperparameter tuning |
| `GradientBoosting.py` | XGBoost with extended feature engineering |
| `XGBFeatureSelection.py` | XGBoost with selection of the top 50 features |
| `OnlineNewsPopularity.csv` | Original dataset |
| `docs/project-report-sr.pdf` | Project report (Serbian) |

## Dataset
K. Fernandes, P. Vinagre, P. Cortez, P. Sernadela. *Online News Popularity*. UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5NS3V

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · seaborn