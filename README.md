# Rozpoznawanie rzeczowników w słowniku Bolesława Chwaścińskiego

Mini-projekt zaliczeniowy z przedmiotu *Wprowadzenie do uczenia maszynowego i analizy danych dla studentów kierunków nieinformatycznych* (ZOOPLS.D200000.18111.26, dr inż. Joanna Świebocka-Więk), Uniwersytet Jagielloński.

**Autor:** Wojciech Fotek

## Cel projektu

Zbudowanie modelu uczenia nadzorowanego, który na podstawie cech leksykalnych i ortograficznych hasła rozpoznaje, czy dane hasło jest rzeczownikiem.

Rzeczowniki stanowią ok. 55% haseł i nie mają jednego silnego markera morfologicznego. Model musi więc opierać się na cechach kontrastowych: rozpoznaje rzeczownik raczej przez wykluczenie. Dlatego sama dokładność nie wystarcza do oceny modelu.

## Dane

Dane pochodzą z bazy leksykalnej przygotowanej w ramach mojej pracy licencjackiej pt. *An Analysis of Bolesław Chwaściński’s Kaboli Dictionary and Grammatical Sketch*, napisanej pod kierunkiem dr. Mateusza M. Kłagisza w Instytucie Orientalistyki UJ.

Baza liczy 1159 wyrównanych haseł z trzech rękopisów:

- MS 1 (polsko-dari, 1143 hasła);
- MS 2 (dari-polski, 995 haseł);
- RB (polsko-dari, 1067 haseł).

**Pełna baza (`data.csv`) nie jest publikowana w repozytorium.** W notebooku widoczne są przykładowe hasła.

### Opis kolumn

| Kolumna  | Opis |
|----------|------|
| `polish` | polskie hasło bazowe, głównie z MS 1 |
| `dari`   | ekwiwalent hasła bazowego w dari, głównie z MS 1 (zapis w transkrypcji, często bez znaków diakrytycznych) |
| `pos`    | część mowy |
| `sd`     | kod domeny semantycznej (1–11) wg klasyfikacji z mojej pracy licencjackiej; `---` dla haseł niesklasyfikowanych |

## Metody

- eksploracyjna analiza danych (rozkład części mowy, długość haseł);
- inżynieria cech kontrastowych (np. końcówki *-ć*, *się*, *-dan*/*-tan* wskazujące na czasownik, *-y*/*-i* wskazujące na przymiotnik);
- klasyfikator bazowy (`DummyClassifier`) jako punkt odniesienia;
- drzewo decyzyjne i algorytm kNN (ze standaryzacją cech);
- ocena: dokładność, walidacja krzyżowa, macierze pomyłek, precyzja, pełność i F1.

## Wyniki

Klasyfikator bazowy osiąga ok. 55% (zawsze „rzeczownik”). Drzewo decyzyjne (ok. 82%) i kNN (ok. 84%) przekraczają ten próg o 27–29 punktów procentowych, a walidacja krzyżowa (ok. 86%) potwierdza, że wynik nie zależy od jednego podziału danych.

## Pliki

- `klasyfikator.ipynb` – notebook z kodem i wynikami (wykresy, metryki);
- `klasyfikator.py` – ten sam kod jako skrypt.

## Wymagania

```
pandas
matplotlib
seaborn
scikit-learn
```

Do uruchomienia potrzebny jest plik `data.csv` w katalogu, z którego uruchamiany jest skrypt. Ponieważ baza nie jest publikowana, kodu nie da się uruchomić bez dostępu do niej.
