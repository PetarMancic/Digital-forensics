# Digital-forensics


# File System Metadata Analysis and Visualization Tool

##  Opšti opis projekta

Ovaj projekat predstavlja sveobuhvatan alat za analizu i vizualizaciju metapodataka fajlova u fajl sistemu, razvijen korišćenjem programskog jezika **Python**.
Alat je osmišljen sa ciljem da omogući sistematsku obradu velikog broja fajlova i da korisniku pruži jasan, intuitivan i vizuelno razumljiv prikaz strukture i ponašanja fajl sistema.

Analiza metapodataka fajlova ima široku primenu u oblastima kao što su **digitalna forenzika**, **administracija sistema**, **bezbednost informacija** i **analiza podataka**, jer omogućava otkrivanje neuobičajenih obrazaca, anomalija i potencijalno sumnjivih aktivnosti.

---

##  Ciljevi projekta

Glavni ciljevi ovog projekta su:

- prikupljanje i centralizacija metapodataka fajlova iz fajl sistema,
- analiza strukture fajlova na osnovu tipa, veličine i vremenskih atributa,
- vizuelna prezentacija rezultata analize kroz različite tipove grafika,
- identifikacija anomalija u fajl sistemu na osnovu statističkih metoda,
- omogućavanje lakšeg i bržeg razumevanja ponašanja fajl sistema.

---

##  Metapodaci koji se analiziraju

Alat prikuplja sledeće metapodatke za svaki fajl:

- **Naziv fajla**
- **Ekstenzija (tip fajla)**
- **Veličina fajla (u bajtovima)**
- **Vreme kreiranja fajla**
- **Vreme poslednje izmene**
- **Vreme poslednjeg pristupa**

Ovi podaci se skladište u strukturiranom formatu (pandas DataFrame) i predstavljaju osnovu za dalju analizu i vizualizaciju.

---

##  Vizualizacije i analiza

Ovaj alat sluzi kako bi generisao 4 razlicita grafika.

### 1️⃣ Distribucija fajlova po tipu

Distribucija fajlova po tipu prikazuje se pomoću **bar grafikona**, gde svaka kolona predstavlja određenu ekstenziju fajla, dok visina kolone označava ukupan broj fajlova tog tipa.
Takodje za distribuciju fajlova po tipu se generise "pie chart".

Ova vizualizacija omogućava:
- brz uvid u dominantne tipove fajlova,
- identifikaciju neuobičajenih ili retkih tipova fajlova,
- razumevanje strukture fajl sistema sa aspekta sadržaja.

---

### 2️⃣ Distribucija fajlova po veličini

Distribucija veličine fajlova prikazuje se pomoću **histograma**, gde su fajlovi grupisani u raspone veličina.

Ovakav prikaz omogućava:
- analizu raspodele veličine fajlova,
- identifikaciju velikih fajlova koji mogu zauzimati značajan prostor,
- uočavanje nepravilnosti u strukturi veličina fajlova.

---

### 3️⃣ Vremenska analiza fajlova

Vremenska analiza se vrši na osnovu vremena kreiranja i izmene fajlova i uključuje:

- **linijske grafike i vremenske serije** koje prikazuju broj fajlova kroz vreme,
- **kalendarske (heatmap) vizualizacije** koje omogućavaju pregled aktivnosti po danima i mesecima.

Ove vizualizacije pomažu u:
- razumevanju dinamike korišćenja fajlova,
- identifikaciji perioda povećane ili neuobičajene aktivnosti,
- analizi ponašanja korisnika ili sistema tokom vremena.

---

### 4️⃣ Detekcija anomalija

Detekcija anomalija se zasniva na statističkoj analizi metapodataka, pri čemu se identifikuju fajlovi koji značajno odstupaju od prosečnih vrednosti. Prosečna vrednost se ogleda u količini broja pristupa nekom fajlu u toku dana. 
Ukoliko se previse puta pristupalo 

Primeri anomalija uključuju:
- fajlove čija je veličina znatno veća od prosečne vrednosti,
- neuobičajene vremenske obrasce izmena ili pristupa.

Anomalije se vizualno ističu pomoću **scatter grafika**, čime se omogućava njihovo brzo i jasno prepoznavanje.

---

## 🛠️ Tehnologije i alati

U okviru projekta korišćene su sledeće tehnologije i biblioteke:

- **Python** – osnovni programski jezik
- **pandas** – obrada i analiza podataka
- **matplotlib** – statičke vizualizacije
- **plotly** – interaktivne vizualizacije sa prikazom tačnih vrednosti pri prelasku mišem

---

## ▶️ Pokretanje projekta

### Preduslovi
Potrebno je imati instaliran Python (verzija 3.9 ili novija).

### Instalacija zavisnosti
```bash
pip install pandas matplotlib plotly
```
Takodje potrebno bilo koje razvojno okruženje u kojem bismo mogli implementirati resenje. U mom slucaju ja sam koristio Visual Studio Code.

Sto se tice pokretanja samog programa:

```bash
python test.py --path --out-dir --visualize --detect-iqr

--path ---> predstavlja putanju do foldera koji bismo zeleli da analiziramo
--out-dir--> predstavlja putanju do foldera gde bismo zeleli da skladistimo rezultat funkcije (slike grafikona )
--visualize -> boolean vrednost kojom kazemo da li zelimo da iscrtamo grafike ili ne
--detect-iqr -> boolean vrednost kojom kazemo da li zelimo da se iscrta grafik anomalije

primer kako sam ja pozivao funkciju je :
python test.py "C\Master\DigitalForensics" "outputDirectory" --visualize --detect-iqr
```


