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


def plot_distributions(df, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Distribucija po tipu
    plt.figure(figsize=(10,6))
    df['extension'].value_counts().head(20).plot(kind='bar')
    plt.title('Top 20 fajl ekstenzija')
    plt.savefig(Path(out_dir)/'type_distribution.png')
    plt.close()

    # Histogram veličine
    plt.figure(figsize=(10,6))
    plt.hist(np.log1p(df['size_bytes']), bins=50)
    plt.title('Histogram veličine fajlova (log scale)')
    plt.savefig(Path(out_dir)/'size_histogram.png')
    plt.close()

    # Boxplot
    plt.figure(figsize=(8,4))
    plt.boxplot(df['size_bytes'], vert=False)
    plt.title('Boxplot veličine fajlova')
    plt.savefig(Path(out_dir)/'size_boxplot.png')
    plt.close()

    # Kalendarska heatmap (modifikacije po danu)
    df['mod_day'] = df['modified'].dt.date
    count_day = df.groupby('mod_day').size().reset_index(name='count')
    fig = px.bar(count_day, x='mod_day', y='count', title='Broj modifikacija po danu')
    fig.write_html(Path(out_dir)/'calendar_heatmap.html')


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
        plot_distributions(df, args.out_dir)
        print(f'Grafički prikazi sačuvani u {args.out_dir}/')

if __name__ == '__main__':
    main()
