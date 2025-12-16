#!/usr/bin/env python3
"""file_metadata_analyzer.py

Alat za analizu i vizualizaciju metapodataka fajlova u fajl sistemu.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# --------------------------------------------------
# POMOĆNE FUNKCIJE
# --------------------------------------------------

def scan_directory(root_path, include_hidden=False):
    metadata = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            filenames = [f for f in filenames if not f.startswith('.')]

        for fname in filenames:
            fpath = Path(dirpath) / fname
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

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[(df[column] < lower) | (df[column] > upper)]


# --------------------------------------------------
# VIZUALIZACIJE
# --------------------------------------------------

def plot_file_types(df, out_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(df['extension'].value_counts())
    plt.title('Distribucija fajlova prema tipu')
    plt.xlabel('Ekstenzija')
    plt.ylabel('Broj fajlova')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'file_types.png')
    plt.close()


def plot_file_sizes(df, out_dir):
    sizes = df['size_bytes']

    plt.figure(figsize=(10, 6))

    bins = np.logspace(
        np.log10(sizes.min()),
        np.log10(sizes.max()),
        50
    )

    plt.hist(sizes, bins=bins)
    plt.xscale('log')

    plt.xlabel('Veličina fajla (B) – log skala')
    plt.ylabel('Broj fajlova')
    plt.title('Distribucija veličine fajlova')

    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'file_sizes.png', dpi=200)
    plt.close()


def plot_modification_time(df, out_dir):
    df['modified_month'] = df['modified'].dt.to_period('M')
    counts = df['modified_month'].value_counts().sort_index()

    plt.figure(figsize=(12, 4))
    counts.plot()
    plt.title('Broj izmenjenih fajlova kroz vreme')
    plt.xlabel('Vreme')
    plt.ylabel('Broj fajlova')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'file_modifications_over_time.png')
    plt.close()


def plot_file_sizes_interactive(df, out_dir):
    fig = px.histogram(
        df,
        x='size_bytes',
        nbins=50,
        log_x=True,
        title='Distribucija veličine fajlova (log skala)',
        labels={'size_bytes': 'Veličina fajla (B)'}
    )

    fig.write_html(f"{out_dir}/file_sizes.html")

def plot_calendar_heatmap(df, out_dir):
    df['date'] = df['modified'].dt.date
    daily = df.groupby('date').size().reset_index(name='count')

    fig = px.density_heatmap(
        daily,
        x='date',
        y='count',
        z='count',
        title='Kalendarska vizualizacija aktivnosti fajlova'
    )

    fig.write_html(Path(out_dir) / 'calendar_activity.html')


def plot_distributions(df, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    plot_file_types(df, out_dir)
    plot_file_sizes(df, out_dir)
    plot_modification_time(df, out_dir)
    plot_calendar_heatmap(df, out_dir)
    plot_file_sizes_interactive(df,out_dir)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Alat za analizu i vizualizaciju metapodataka fajlova'
    )
    parser.add_argument('--path', type=str, required=True, help='Putanja do direktorijuma')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Folder za izlazne fajlove')
    parser.add_argument('--visualize', action='store_true', help='Generiši grafičke vizualizacije')
    parser.add_argument('--detect-iqr', action='store_true', help='Detekcija anomalija veličine fajlova')
    parser.add_argument('--include-hidden', action='store_true', help='Uključi skrivene fajlove')

    args = parser.parse_args()

    print(args.path)
    print(args.out_dir)
    print(args.visualize)
    print(args.detect_iqr)
    print(args.include_hidden)

    df = scan_directory(args.path, include_hidden=args.include_hidden)
    print(f'Pronađeno {len(df)} fajlova.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / 'metadata.csv', index=False)
    print(f'Metapodaci sačuvani u {out_dir / "metadata.csv"}')

    if args.detect_iqr:
        anomalies = detect_anomalies_iqr(df)
        anomalies.to_csv(out_dir / 'anomalies.csv', index=False)
        print(f'Detekovano {len(anomalies)} anomalija. Sačuvano u anomalies.csv')

    if args.visualize:
        plot_distributions(df, out_dir)
        print(f'Grafički prikazi sačuvani u {out_dir}')


if __name__ == '__main__':
    main()
