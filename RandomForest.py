import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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


# Odvajanje X i y
X = data.drop('shares', axis=1)
y = data['shares']

# Logaritamska transformacija ciljne varijable
y_log = np.log1p(y)

# Podela na trening i test set
# Za Random Forest NE SKALIRAMO X
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"Podaci su podeljeni na trening ({len(X_train)} redova) i test ({len(X_test)} redova) set.")

# Definišemo prostor parametara za pretragu
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 1.0] # 1.0 je isto sto i None za RandomForestRegressor
}

# Kreiramo RandomizedSearchCV objekat
# n_iter=20 znaci da ce testirati 20 nasumicnih kombinacija
# cv=3 znaci 3-fold unakrsna validacija
print("\n--- Započeta pretraga najboljih hiperparametara za Random Forest ---")

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=40,
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1 # Koristi sve dostupne procesore
)

# Pokretanje pretrage na trening podacima
rf_random.fit(X_train, y_train)

# Ispis najboljih pronadjenih parametara
print("\nPretraga završena.")
print("Najbolji pronađeni parametri:")
print(rf_random.best_params_)

# Kreiramo finalni model sa najboljim parametrima
best_rf_model = rf_random.best_estimator_

# Predikcija na test setu (predikcije su na log skali)
y_pred_log = best_rf_model.predict(X_test)

# Evaluacija na LOGARITAMSKOJ skali
y_test_log = y_test
mae_log = mean_absolute_error(y_test_log, y_pred_log)
mse_log = mean_squared_error(y_test_log, y_pred_log)
r2_log = r2_score(y_test_log, y_pred_log)

print("\n--- Rezultati finalnog modela (NA LOGARITAMSKOJ SKALI) ---")
print(f"Srednja apsolutna greška (MAE): {mae_log:.4f}")
print(f"Srednja kvadratna greška (MSE): {mse_log:.4f}")
print(f"R² koeficijent determinacije: {r2_log:.4f}")


# Evaluacija na STVARNOJ skali
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

mae_real = mean_absolute_error(y_test_original, y_pred_original)
mse_real = mean_squared_error(y_test_original, y_pred_original)
r2_real = r2_score(y_test_original, y_pred_original)

print("\n--- Rezultati finalnog modela (NA STVARNOJ SKALI) ---")
print(f"Srednja apsolutna greška (MAE): {mae_real:.2f}")
print(f"Srednja kvadratna greška (MSE): {mse_real:.2f}")
print(f"R² koeficijent determinacije: {r2_real:.4f}")


plt.figure(figsize=(8, 8))
plt.scatter(y_test_original, y_pred_original, alpha=0.3, s=15)
plt.plot([0, y_test_original.max()], [0, y_test_original.max()], 'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel("Stvarne vrednosti (shares)")
plt.ylabel("Predikcije (shares)")
plt.title("Random Forest: Predikcija vs Stvarne vrednosti (Stvarna Skala)")
plt.xlim(0, 40000)
plt.ylim(0, 40000)
plt.legend()
plt.grid(True)
plt.show()