import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bolji prikaz ispisa u konzoli
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Funkcija koja ucitava dataset, vrsi osnovne provere kvaliteta podataka i ispisuje izvestaj
def proveri_dataset(putanja_do_fajla):
    print("\n1. UČITAVANJE")
    try:
        # Ucitavanje, skipinitialspace=True za uklanjanje razmaka nakon zareza
        data = pd.read_csv(putanja_do_fajla, skipinitialspace=True)
        # Strip za uklanjanje razmaka iz naziva kolona
        data.columns = [col.strip() for col in data.columns]
        print(f"Dataset sa putanje '{putanja_do_fajla}' je uspešno učitan.")
        print("-" * 50)
    except FileNotFoundError:
        print(f"GREŠKA: Fajl nije pronađen na putanji: '{putanja_do_fajla}'. Proverite putanju i pokušajte ponovo.")
        return None

    # Informacije o datasetu
    print("\n2. OSNOVNE INFORMACIJE O DATASETU")
    print(f"Originalni oblik dataseta: {data.shape[0]} redova (članaka) i {data.shape[1]} kolona (atributa).")
    print("\nTipovi podataka i broj ne-null vrednosti po kolonama:")
    data.info()
    print("-" * 50)

    # Uklanjanje clanaka bez sadrzaja (jer to nisu clanci)
    print("\n3. ČIŠĆENJE PODATAKA: Uklanjanje članaka bez sadržaja")
    broj_redova_pre = data.shape[0]
    # Filtriramo redove gde je broj reci u sadrzaju veći od 0
    data = data[data['n_tokens_content'] > 0].copy()
    broj_redova_posle = data.shape[0]
    print(f"Uklonjeno je {broj_redova_pre - broj_redova_posle} redova gde je 'n_tokens_content' bio 0.")
    print(f"Novi oblik dataseta: {data.shape[0]} redova i {data.shape[1]} kolona.")
    print("-" * 50)

    # Kvalitet podataka
    print("\n4. PROVERE KVALITETA PODATAKA")
    # Provera nedostajucih vrednosti
    if data.isnull().sum().sum() == 0:
        print("-> Provera nedostajućih vrednosti: Nema nedostajućih (null) vrednosti.")
    else:
        print("-> Provera nedostajućih vrednosti: Pronađene su nedostajuće vrednosti:")
        print(data.isnull().sum()[data.isnull().sum() > 0])

    # Provera duplikata
    if 'url' in data.columns:
        if data['url'].duplicated().any():
            print(f"-> Provera duplikata: Pronađeno je {data['url'].duplicated().sum()} dupliranih URL-ova.")
        else:
            print("-> Provera duplikata: Nema dupliranih URL-ova u datasetu.")

        # Uklanjanje url kolone
        data = data.drop('url', axis=1)
        print("-> Kolona 'url' je uklonjena iz dataseta.")
    else:
        print("-> Kolona 'url' nije pronađena, provera duplikata preskočena.")

    # Lista kolona za uklanjanje zbog visoke korelacije sa drugim atributima
    kolone_za_uklanjanje = [
        # Grupa 1: Sadrzaj (zadrzavamo n_tokens_content)
        'n_non_stop_words',
        'n_non_stop_unique_tokens',

        # Grupa 2: Dani u nedelji (zadrzavamo is_weekend)
        'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
        'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday',

        # Grupa 3: Sentiment (zadržavamo global_... varijante)
        'rate_positive_words', 'rate_negative_words'
    ]

    # Proveravamo koje od ovih kolona zaista postoje u DataFrame-u
    postojece_kolone_za_uklanjanje = [kol for kol in kolone_za_uklanjanje if kol in data.columns]

    # Uklanjanje selektovanih kolona
    data = data.drop(columns=postojece_kolone_za_uklanjanje)

    print(
        f"-> Uklonjeno je {len(postojece_kolone_za_uklanjanje)} redundantnih kolona radi smanjenja multikolinearnosti.")

    print(f"Novi oblik dataseta: {data.shape[0]} redova i {data.shape[1]} kolona.")
    print("-" * 50)

    # Statisticki pregled
    print("\n5. STATISTIČKI PREGLED NUMERIČKIH ATRIBUTA")
    # .describe() daje pregled kao sto su srednja vrednost, standardna devijacija, min, max, itd.
    print(data.describe())
    print("-" * 50)

    # Analiza ciljne varijable
    print("\n6. ANALIZA CILJNE VARIJABLE ('shares')")
    print("Statistika za broj deljenja (shares):")
    print(data['shares'].describe())

    # Vizualizacija distribucije broja deljenja
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data['shares'], bins=100, kde=False)
    plt.title('Distribucija broja deljenja (shares)')
    plt.xlabel('Broj deljenja')
    plt.ylabel('Broj članaka')

    plt.subplot(1, 2, 2)
    log_shares = np.log1p(data['shares'])
    sns.histplot(log_shares, bins=50, kde=True)
    plt.title('Distribucija Log(1 + shares)')
    plt.xlabel('Logaritamski broj deljenja')
    plt.ylabel('Broj članaka')

    plt.suptitle('Analiza distribucije ciljne varijable', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("\nGrafički prikaz distribucije broja deljenja (originalne i log-transformisane vrednosti) je generisan.")
    print("-" * 50)

    # Matrica korelacije
    print("\n7. ANALIZA KORELACIJE (HEATMAP)")

    # Racunanje matrice
    correlation_matrix = data.corr()

    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, cmap='viridis')
    plt.title('Heatmap Korelacije Atributa', fontsize=18)
    plt.show()
    print("Heatmap je uspešno generisan.")

    # Izdvajanje i ispis korelacija sa ciljnom varijablom
    print("\nTop 15 najviše pozitivno korelisanih atributa sa 'shares':")
    print(correlation_matrix['shares'].sort_values(ascending=False).head(16)[1:])  # [1:] da preskocimo 'shares'
    print("\nTop 5 najviše negativno korelisanih atributa sa 'shares':")
    print(correlation_matrix['shares'].sort_values(ascending=True).head(5))
    print("-" * 50)

    return data

putanja_do_dataseta = 'OnlineNewsPopularity.csv'
data = proveri_dataset(putanja_do_dataseta)

if data is not None:
    print("\n\nAnaliza je završena. DataFrame 'data' je spreman za dalje korake.")

    ime_ociscenog_fajla = 'OnlineNewsPopularity_Cleaned.csv'
    data.to_csv(ime_ociscenog_fajla, index=False)
    print(f"Očišćeni dataset je sačuvan u fajl: '{ime_ociscenog_fajla}'")