import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

try:
    data = pd.read_csv('OnlineNewsPopularity_Cleaned.csv')
    print("Očišćeni dataset je uspešno učitan.")
    print(f"Oblik dataseta pre feature engineeringa: {data.shape}")
except FileNotFoundError:
    print("GREŠKA: Fajl 'OnlineNewsPopularity_Cleaned.csv' nije pronađen.")
    print("Molimo vas da prvo pokrenete skriptu za pripremu podataka ('Ucitavanje.py').")
    exit()

# Feature engineering
# Ideja je da je popularna tema objavljena subotom ili nedeljom
data['popular_topic_on_weekend'] = data['kw_avg_avg'] * data['is_weekend']
# Bolje su teme koje su i popularne i sveže
data['topic_freshness_factor'] = data['kw_avg_avg'] / (data['timedelta'] + 1)
# Visok udeo internih linkova može biti znak kvalitetnog sadrzaja koji vodi ka većem deljenju
data['internal_links_ratio'] = data['num_self_hrefs'] / (data['num_hrefs'] + 1)

# Subjektivni i emotivni naslovi izazivaju vecu reakciju
if 'title_subjectivity' in data.columns and 'abs_title_sentiment_polarity' in data.columns:
    data['title_impact_score'] = data['title_subjectivity'] * data['abs_title_sentiment_polarity']
# Format sadrzaja
if all(k in data.columns for k in ['num_imgs', 'num_videos', 'n_tokens_content']):
    data['media_density'] = ((data['num_imgs'] + data['num_videos']) / (data['n_tokens_content'] + 1)) * 100
# Da li je clanak umereno popularan ili ekspluzivno
if all(k in data.columns for k in ['kw_max_max', 'kw_avg_avg']):
    data['topic_volatility'] = data['kw_max_max'] / (data['kw_avg_avg'] + 1)

print(f"Novi oblik dataseta: {data.shape}")


X = data.drop('shares', axis=1)
y = data['shares']
y_log = np.log1p(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)
print(f"Podaci su podeljeni na trening ({len(X_train)} redova) i test ({len(X_test)} redova) set.")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01)
}

print("\n--- Treniranje i evaluacija modela ---")
for name, model in models.items():
    # Treniranje
    model.fit(X_train, y_train)

    # Predikcija (na log skali)
    y_pred_log = model.predict(X_test)
    y_test_log = y_test

    # Evaluacija na LOGARITAMSKOJ skali
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    mse_log = mean_squared_error(y_test_log, y_pred_log)
    r2_log = r2_score(y_test_log, y_pred_log)

    # Evaluacija na STVARNOJ skali
    y_test_original = np.expm1(y_test_log)
    y_pred_original = np.expm1(y_pred_log)
    mae_real = mean_absolute_error(y_test_original, y_pred_original)
    mse_real = mean_squared_error(y_test_original, y_pred_original)
    r2_real = r2_score(y_test_original, y_pred_original)

    print(f"\n--- Rezultati za: {name} ---")
    print("  Na LOGARITAMSKOJ skali:")
    print(f"    MAE: {mae_log:.4f}, MSE: {mse_log:.4f}, R²: {r2_log:.4f}")
    print("  Na STVARNOJ skali:")
    print(f"    MAE: {mae_real:.2f}, MSE: {mse_real:.2f}, R²: {r2_real:.4f}")


# Vizualizacija za Ridge model
ridge_model = models["Ridge Regression"]
y_pred_log_ridge = ridge_model.predict(X_test)
y_test_log = y_test
y_test_original = np.expm1(y_test_log)
y_pred_original_ridge = np.expm1(y_pred_log_ridge)

# Logaritamska skala
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.scatter(y_test_log, y_pred_log_ridge, alpha=0.3, s=15)
plt.plot([y_test_log.min(), y_test_log.max()],
         [y_test_log.min(), y_test_log.max()],
         'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel("Stvarne vrednosti log(1 + shares)")
plt.ylabel("Predikcije log(1 + shares)")
plt.title("Ridge Regression: Predikcija vs Stvarne vrednosti (Log Skala)")
plt.legend()
plt.grid(True)

# Stvarna skala
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_original_ridge, alpha=0.3, s=15)
plt.plot([0, y_test_original.max()],
         [0, y_test_original.max()],
         'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel("Stvarne vrednosti (shares)")
plt.ylabel("Predikcije (shares)")
plt.title("Ridge Regression: Predikcija vs Stvarne vrednosti (Stvarna Skala)")
plt.xlim(0, 40000)
plt.ylim(0, 40000)
plt.legend()
plt.grid(True)

plt.suptitle("Poređenje performansi na logaritamskoj i stvarnoj skali", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()