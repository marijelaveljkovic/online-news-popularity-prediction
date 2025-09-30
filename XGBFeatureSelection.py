import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


try:
    data = pd.read_csv('OnlineNewsPopularity_Cleaned.csv')
    print("Očišćeni dataset je uspešno učitan.")
except FileNotFoundError:
    print("GREŠKA: Fajl 'OnlineNewsPopularity_Cleaned.csv' nije pronađen.")
    exit()

# Dodavanje svih osnovnih i naprednih atributa
print(f"\nOblik dataseta pre feature engineeringa: {data.shape}")
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

# 1. Interakcije Kanala (Channels) sa drugim Atributima
channels = ['lifestyle', 'entertainment', 'bus', 'socmed', 'tech', 'world']
for channel in channels:
    col_name = f'data_channel_is_{channel}'
    if col_name in data.columns:
        # Interakcija sa prosecnom popularnoscu kljucnih reci
        data[f'{channel}_interact_kw_avg'] = data[col_name] * data['kw_avg_avg']
        # Interakcija sa sentimentom naslova
        data[f'{channel}_interact_title_sent'] = data[col_name] * data['title_sentiment_polarity']

# 2. Napredniji Atributi Kljucnih Reci
data['kw_popularity_range'] = data['kw_max_max'] - data['kw_min_min']
data['kw_max_to_avg_ratio'] = data['kw_avg_max'] / (data['kw_avg_avg'] + 1)

# 3. Kombinovanje Sentiment Atributa
data['sentiment_title_vs_content'] = abs(data['title_sentiment_polarity'] - data['global_sentiment_polarity'])
data['combined_subjectivity'] = data['title_subjectivity'] * data['global_subjectivity']

# 4. Nelinearna transformacija 'timedelta'
data['timedelta_log'] = np.log1p(data['timedelta'])

print(f"Oblik dataseta nakon kompletnog feature engineeringa: {data.shape}")

# Priprema kompletnog seta podataka
X_full = data.drop('shares', axis=1)
y_log = np.log1p(data['shares'])

# Treniranje pocetnog modela za selekciju atributa
print("\n--- Korak 1: Treniranje početnog modela za selekciju atributa ---")
# Koristimo ceo set podataka za ovo, jer zelimo najstabilniju procenu vaznosti
initial_xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
initial_xgb_model.fit(X_full, y_log) # Treniramo na svim podacima

# Selekcija Top 50 atributa
print("\n--- Korak 2: Selekcija Top 50 najvažnijih atributa ---")
feature_importances = pd.Series(initial_xgb_model.feature_importances_, index=X_full.columns)
top_50_features = feature_importances.nlargest(50).index.tolist()

print("Top 50 pronađenih atributa:")
print(top_50_features)

# Kreiranje novog DataFrame-a koji sadrzi samo ove atribute
X_top50 = X_full[top_50_features]


# Treniranje i optimizacija finalnog modela na Top 50 atributa
print("\n--- Korak 3: Optimizacija finalnog modela na Top 50 atributa ---")
# Podela novog, manjeg skupa podataka
X_train, X_test, y_train, y_test = train_test_split(X_top50, y_log, test_size=0.2, random_state=42)
print(f"Podaci su podeljeni na trening ({len(X_train)} redova) i test ({len(X_test)} redova) set.")

param_grid = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_distributions=param_grid,
    n_iter=50,
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

print("Optimizacija u toku, ovo može potrajati...")
random_search.fit(X_train, y_train)

print("\nPretraga je završena.")
print("Najbolji pronađeni parametri:")
print(random_search.best_params_)

final_xgb_model = random_search.best_estimator_


# Evaluacija finalnog modela
y_pred_log = final_xgb_model.predict(X_test)

# Evaluacija na LOGARITAMSKOJ skali
mae_log = mean_absolute_error(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log = r2_score(y_test, y_pred_log)

print("\n--- Rezultati finalnog modela sa Top 50 Atributa (NA LOGARITAMSKOJ SKALI) ---")
print(f"Srednja apsolutna greška (MAE): {mae_log:.4f}")
print(f"Koren srednje kvadratne greške (RMSE): {rmse_log:.4f}")
print(f"R² koeficijent determinacije: {r2_log:.4f}")

# Evaluacija na STVARNOJ skali
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)

mae_real = mean_absolute_error(y_test_original, y_pred_original)
rmse_real = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
r2_real = r2_score(y_test_original, y_pred_original)

print("\n--- Rezultati finalnog modela sa Top 50 Atributa (NA STVARNOJ SKALI) ---")
print(f"Srednja apsolutna greška (MAE): {mae_real:.2f}")
print(f"Koren srednje kvadratne greške (RMSE): {rmse_real:.2f}")
print(f"R² koeficijent determinacije: {r2_real:.4f}")



plt.figure(figsize=(16, 8))

# Logaritamska Skala
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_log, alpha=0.3, s=15, color='darkviolet')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel("Stvarne vrednosti log(1 + shares)")
plt.ylabel("Predikcije log(1 + shares)")
plt.title("Predikcija vs Stvarne vrednosti (Log Skala)")
plt.legend()
plt.grid(True)

# Stvarna Skala
plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_original, alpha=0.3, s=15, color='crimson')
plt.plot([0, 50000], [0, 50000], 'r--', linewidth=2, label='Savršena predikcija')
plt.xlabel("Stvarne vrednosti (shares)")
plt.ylabel("Predikcije (shares)")
plt.title("Predikcija vs Stvarne vrednosti (Stvarna Skala)")
plt.xlim(0, 50000)
plt.ylim(0, 50000)
plt.legend()
plt.grid(True)

plt.suptitle("Performanse XGBoost Modela sa Selekcijom Top 50 Atributa", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()