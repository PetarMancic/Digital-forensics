#!/usr/bin/env python3
"""file_metadata_analyzer.py

Alat za analizu i vizualizaciju metapodataka fajlova u fajl sistemu.

Funkcionalnosti:
- Skeniranje direktorijuma i prikupljanje metapodataka: ime fajla, ekstenzija, veličina, vreme kreiranja, modifikacije i pristupa
- Analiza podataka koristeći pandas
- Vizualizacija: distribucija fajlova po tipu, veličini i vremenskim atributima, kalendarska vizualizacija
- Detekcija anomalija (IQR i opcionalno IsolationForest)
- CLI interfejs za jednostavno korišćenje

Biblioteke: pandas, numpy, matplotlib, plotly, scikit-learn (opciono)"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def human_readable_size(nbytes):
    if nbytes == 0:
        return '0 B'
    sizes = ['B','KB','MB','GB','TB']
    i = int(np.floor(np.log(nbytes)/np.log(1024)))
    p = np.power(1024, i)
    s = round(nbytes / p, 2)
    return f'{s} {sizes[i]}'


def scan_directory(root_path, include_hidden=False):
    metadata = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            filenames = [f for f in filenames if not f.startswith('.')]
        for fname in filenames:
            fpath = Path(dirpath)/fname
            try:
                st = fpath.stat()
                metadata.append({
                    'path': str(fpath),
                    'name': fname,
                    'extension': fpath.suffix.lower().lstrip('.') or 'no_ext',
                    'size_bytes': st.st_size,
                    'created': datetime.fromtimestamp(st.st_ctime),
                    'modified': datetime.fromtimestamp(st.st_mtime),
                    'accessed': datetime.fromtimestamp(st.st_atime)
                })
            except Exception as e:
                print(f'Warning: failed to read {fpath}: {e}')
    return pd.DataFrame(metadata)


def detect_anomalies_iqr(df, column='size_bytes'):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    mask = (df[column]<lower) | (df[column]>upper)
    return df[mask]


def detect_anomalies(df, out_dir):
    mean_size = df['size_bytes'].mean()
    std_size = df['size_bytes'].std()
    threshold = mean_size + 2*std_size  # fajlovi > 2 std od proseka

    plt.figure(figsize=(8,5))
    # Svi fajlovi
    plt.scatter(df['name'], df['size_bytes'], color='blue')
    # Anomalije
    plt.scatter(
        df[df['size_bytes'] > threshold]['name'],
        df[df['size_bytes'] > threshold]['size_bytes'],  # ispravljeno ovde
        color='red',
        label='Anomalije'
    )
    plt.title("Detekcija anomalija po veličini fajla")
    plt.xlabel("Fajl")
    plt.ylabel("Veličina (Bajti)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    


#def plot_distributions(df, out_dir):
    # Path(out_dir).mkdir(parents=True, exist_ok=True)
    # # Distribucija po tipu
    # plt.figure(figsize=(10,6))
    # df['extension'].value_counts().head(20).plot(kind='bar')
    # plt.title('Top 20 fajl ekstenzija')
    # plt.savefig(Path(out_dir)/'type_distribution.png')
    # plt.close()

    # # Histogram veličine
    # plt.figure(figsize=(10,6))
    # plt.hist(np.log1p(df['size_bytes']), bins=50)
    # plt.title('Histogram veličine fajlova (log scale)')
    # plt.savefig(Path(out_dir)/'size_histogram.png')
    # plt.close()

    # # Boxplot
    # plt.figure(figsize=(8,4))
    # plt.boxplot(df['size_bytes'], vert=False)
    # plt.title('Boxplot veličine fajlova')
    # plt.savefig(Path(out_dir)/'size_boxplot.png')
    # plt.close()

    # # Kalendarska heatmap (modifikacije po danu)
    # df['mod_day'] = df['modified'].dt.date
    # count_day = df.groupby('mod_day').size().reset_index(name='count')
    # fig = px.bar(count_day, x='mod_day', y='count', title='Broj modifikacija po danu')
    # fig.write_html(Path(out_dir)/'calendar_heatmap.html')

def plot_file_types(df, out_dir):
    print("Distribucija fajlova po tipu")
    type_counts = df['extension'].value_counts()
    
    plt.figure(figsize=(8,5))
    type_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Distribucija fajlova po tipu")
    plt.xlabel("Tip fajla")
    plt.ylabel("Broj fajlova")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


  

def plot_file_sizes(df, out_dir):
    # Distribucija po veličini fajla (histogram)
    plt.figure(figsize=(8,5))
    plt.hist(df['size_bytes'], bins=10, color='salmon', edgecolor='black')
    plt.title("Distribucija fajlova po veličini")
    plt.xlabel("Veličina fajla (Bajti)")
    plt.ylabel("Broj fajlova")
    plt.show()


def plot_modification_time(df, out_dir):
    df['modified_month'] = df['modified'].dt.to_period('M')

    print("plot modification time ")
    counts = df['modified_month'].value_counts().sort_index()

    plt.figure(figsize=(10,4))
    counts.plot()
    plt.title('Broj izmenjenih fajlova kroz vreme')
    plt.xlabel('Vreme')
    plt.ylabel('Broj fajlova')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'file_modifications_over_time.png')
    plt.close()


import plotly.express as px

# def plot_calendar_heatmap(df, out_dir):
#     df['date'] = df['modified'].dt.date
#     daily = df.groupby('date').size().reset_index(name='count')

#     fig = px.density_heatmap(
#         daily,
#         x='date',
#         y=['activity'],
#         z='count',
#         title='Kalendarska vizualizacija aktivnosti fajlova'
#     )

#     fig.write_html(Path(out_dir) / 'calendar_activity.html')


def plot_calendar_heatmap(df, out_dir):
    # Pretvori u datetime
    print("Uso sam u calendar heatmap")
    df['created'] = pd.to_datetime(df['created'])

    # Min i max datum
    min_date = df['created'].min().date()
    max_date = df['created'].max().date()
    
    # Kreiraj DataFrame sa brojem fajlova po danu
    daily_counts = df.groupby(df['created'].dt.date).size().reset_index(name='count')

    # Popuni sve datume između min i max
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    daily_counts = daily_counts.set_index('created').reindex(all_dates, fill_value=0)
    daily_counts.index.name = 'date'
    daily_counts.reset_index(inplace=True)

    # Napravi matricu za plot (red = sedmica, kolona = dan)
    daily_counts['week'] = daily_counts['date'].dt.isocalendar().week
    daily_counts['weekday'] = daily_counts['date'].dt.weekday  # Mon=0 ... Sun=6

    # Pivot tabela: red = weekday, kolona = week
    heatmap_data = daily_counts.pivot(index='weekday', columns='week', values='count')

    plt.figure(figsize=(20,4))
    im = plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', origin='lower')

    # X i Y ticks
    plt.yticks(ticks=np.arange(7), labels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    plt.xticks(ticks=np.arange(0, heatmap_data.shape[1], 4))  # svaka 4. nedelja
    plt.xlabel(f'Nedelje od {min_date} do {max_date}')
    plt.colorbar(im, label='Broj fajlova')
    plt.title('Kalendarska heatmap po danima')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(out_dir)/'calendar_heatmap_dynamic.png', dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Alat za analizu i vizualizaciju metapodataka fajlova')
    parser.add_argument('--path', type=str, required=True, help='Putanja do direktorijuma')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Folder za izlazne fajlove')
    parser.add_argument('--visualize', action='store_true', help='Prikaži grafičke vizualizacije')
    parser.add_argument('--detect-iqr', action='store_true', help='Detekcija anomalija veličine fajlova')
    parser.add_argument('--include-hidden', action='store_true', help='Uključi skrivene fajlove')
    args = parser.parse_args()

    df = scan_directory(args.path, include_hidden=args.include_hidden)
    print(f'Pronađeno {len(df)} fajlova.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir/'metadata.csv', index=False)

    print(f'Metapodaci sačuvani u {args.out_dir}/metadata.csv')

    if args.detect_iqr:
        anomalies = detect_anomalies_iqr(df)
        anomalies.to_csv(Path(args.out_dir)/'anomalies.csv', index=False)
        print(f'Detekovano {len(anomalies)} anomalija. Sačuvano u anomalies.csv')

    if args.visualize:
        #plot_distributions(df, args.out_dir)
        print("visualize je true")
        plot_file_types(df, args.out_dir)
        plot_file_sizes(df, args.out_dir)
        plot_modification_time(df, args.out_dir)
        # plot_calendar_heatmap(df,args.out_dir)
        # detect_anomalies(df, args.out_dir)
        print(f'Grafički prikazi sačuvani u {args.out_dir}/')

if __name__ == '__main__':
    main()
