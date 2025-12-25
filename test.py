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
import seaborn as sns

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    threshold = mean_size + 2*std_size  

    plt.figure(figsize=(8,5))
    plt.scatter(df['name'], df['size_bytes'], color='blue')
    plt.scatter(
        df[df['size_bytes'] > threshold]['name'],
        df[df['size_bytes'] > threshold]['size_bytes'],  
        color='red',
        label='Anomalije'
    )
    plt.title("Detekcija anomalija po veličini fajla")
    plt.xlabel("Fajl")
    plt.ylabel("Veličina (Bajti)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    

def plot_file_sizes(df, out_dir):
    """Analiza veličine fajlova: histogram + pie chart + tabela statistika"""

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec

    # Stil
    sns.set_style("whitegrid")

    # Figura + GridSpec (2 reda x 2 kolone)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(
        2, 2,
        height_ratios=[3, 1],   
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.2
    )

    
    ax1 = fig.add_subplot(gs[0, 0])   
    ax2 = fig.add_subplot(gs[0, 1])   
    table_ax = fig.add_subplot(gs[1, 1])  

    n_bins = min(20, max(5, len(df) // 5))

    sns.histplot(
        data=df,
        x='size_bytes',
        bins=n_bins,
        kde=True,
        color='lightseagreen',
        edgecolor='teal',
        linewidth=1,
        alpha=0.7,
        ax=ax1
    )

    sns.kdeplot(
        data=df,
        x='size_bytes',
        color='darkred',
        linewidth=2,
        ax=ax1
    )

    ax1.set_title(
        f"Distribucija veličina fajlova (N={len(df)})",
        fontsize=14, fontweight='bold', pad=15
    )
    ax1.set_xlabel("Veličina u bajtima", fontsize=11)
    ax1.set_ylabel("Broj fajlova", fontsize=11)

    stats = {
        'mean': df['size_bytes'].mean(),
        'median': df['size_bytes'].median(),
        'q1': df['size_bytes'].quantile(0.25),
        'q3': df['size_bytes'].quantile(0.75)
    }

    colors_line = {'mean': 'red', 'median': 'green', 'q1': 'blue', 'q3': 'orange'}
    labels_line = {'mean': 'Prosek', 'median': 'Medijana', 'q1': 'Q1', 'q3': 'Q3'}

    for stat, val in stats.items():
        ax1.axvline(
            val,
            color=colors_line[stat],
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label=f"{labels_line[stat]}: {val:,.0f} B"
        )

    ax1.legend(loc='upper right')

    # ================= PIE CHART =================
    size_categories = pd.cut(
        df['size_bytes'],
        bins=[0, 1024, 10240, 102400, 1024000, float('inf')],
        labels=['<1 KB', '1–10 KB', '10–100 KB', '100–1000 KB', '>1 MB']
    )

    size_counts = size_categories.value_counts().sort_index()
    pie_colors = sns.color_palette("pastel", len(size_counts))

    wedges, texts, autotexts = ax2.pie(
        size_counts.values,
        labels=size_counts.index,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax2.set_title(
        "Udeo fajlova po veličinskim kategorijama",
        fontsize=14, fontweight='bold', pad=20
    )

    legend_labels = [
        f'{label}: {size_counts[label]} fajlova'
        for label in size_counts.index
    ]

    ax2.legend(
        wedges,
        legend_labels,
        title="Kategorije",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10
    )

    # ================= TABELA STATISTIKA =================
    table_ax.axis("off")

    table_data = [
        ["Ukupno fajlova", f"{len(df):,}"],
        ["Ukupna veličina (MB)", f"{df['size_bytes'].sum()/1024/1024:.2f}"],
        ["Prosek (B)", f"{df['size_bytes'].mean():,.0f}"],
        ["Medijana (B)", f"{df['size_bytes'].median():,.0f}"],
        ["Std dev (B)", f"{df['size_bytes'].std():,.0f}"],
        ["Min (B)", f"{df['size_bytes'].min():,.0f}"],
        ["Max (B)", f"{df['size_bytes'].max():,.0f}"],
    ]

    stats_table = table_ax.table(
        cellText=table_data,
        colLabels=["Statistika", "Vrednost"],
        cellLoc="left",
        loc="center"
    )

    stats_table.auto_set_font_size(False)
    stats_table.set_fontsize(10)
    stats_table.scale(1, 1.5)

    for (row, col), cell in stats_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#f2e5c4")

    # ================= NASLOV + SAVE =================
    plt.suptitle(
        "ANALIZA VELIČINE FAJLOVA",
        fontsize=18, fontweight='bold', y=0.98
    )

    if out_dir:
        plt.savefig(
            f"{out_dir}/file_size_analysis_complete.png",
            dpi=150,
            bbox_inches='tight'
        )

    plt.show()


def plot_file_sizes1(df, out_dir):
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.gridspec import GridSpec

    # ================= PRIPREMA PODATAKA =================
    bins = [0, 1024, 10_240, 102_400, 1_024_000, float('inf')]
    labels = ['<1 KB', '1–10 KB', '10–100 KB', '100 KB–1 MB', '>1 MB']

    df['size_category'] = pd.cut(df['size_bytes'], bins=bins, labels=labels)
    category_counts = df['size_category'].value_counts().sort_index()

    avg_size = df['size_bytes'].mean()

    # ================= FIGURA =================
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    # ================= BAR CHART + PROSEK =================
    ax_bar.bar(
        category_counts.index,
        category_counts.values,
        color='#8ecae6',
        edgecolor='black'
    )

    for i, count in enumerate(category_counts.values):
        ax_bar.text(
            i,              
            count + 1,      
            str(count),     
            ha='center',    
            va='bottom',    
            fontsize=10,
            fontweight='bold'
        )    

    ax_bar.set_title("Raspodela fajlova po veličinskim kategorijama", fontsize=14, fontweight='bold')
    ax_bar.set_xlabel("Kategorija veličine")
    ax_bar.set_ylabel("Broj fajlova")
    ax_bar.legend()

    # ================= PIE CHART =================
    ax_pie.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )

    ax_pie.set_title("Procentualni udeo fajlova po veličini", fontsize=14, fontweight='bold')

    # ================= TABELA STATISTIKA =================
    ax_table.axis("off")

    table_data = [
        ["Ukupno fajlova", f"{len(df)}"],
        ["Ukupna veličina (MB)", f"{df['size_bytes'].sum() / 1024 / 1024:.2f}"],
        ["Prosečna veličina (KB)", f"{avg_size / 1024:.2f}"],
        ["Minimalna veličina (B)", f"{df['size_bytes'].min():,}"],
        ["Maksimalna veličina (MB)", f"{df['size_bytes'].max() / 1024 / 1024:.2f}"]
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metod", "Vrednost"],
        cellLoc="left",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # ================= NASLOV + SAVE =================
    plt.suptitle(
        "ANALIZA VELIČINE FAJLOVA",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    if out_dir:
        plt.savefig(
            f"{out_dir}/file_size_overview.png",
            dpi=200,
            bbox_inches='tight'
        )

    plt.show()


def plot_file_types(df, out_dir):
    print("Distribucija fajlova po tipu")
    
    type_counts = df['extension'].value_counts()
    
    threshold = len(df) * 0.02  
    small_categories = type_counts[type_counts < threshold]
    
    if len(small_categories) > 0:
        other_count = small_categories.sum()
        type_counts = type_counts[type_counts >= threshold]
        type_counts['Ostalo'] = other_count
    
    total = len(df)
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
 
    sns.barplot(x=type_counts.index, y=type_counts.values,
                hue=type_counts.index,  
                palette="Set2", 
                legend=False,  
                ax=ax1)
    
    ax1.set_title("Bar Chart - Distribucija po tipu", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Tip fajla", fontsize=11)
    ax1.set_ylabel("Broj fajlova", fontsize=11)
    
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    
    for i, v in enumerate(type_counts.values):
        percentage = v/total*100
        ax1.text(i, v + max(type_counts.values)*0.02,
                f"{v}\n({percentage:.1f}%)", 
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # Dodaj horizontalnu liniju za prosek
    avg_count = total / len(type_counts)
    ax1.axhline(y=avg_count, color='red', linestyle='--', 
                alpha=0.7, linewidth=1.5, label=f'Prosek: {avg_count:.1f}')
    ax1.legend(loc='upper right')
    
    # ============= PLOT 2: PIE CHART =============
    colors = sns.color_palette("Set3", len(type_counts))
    wedges, texts, autotexts = ax2.pie(type_counts.values, 
                                       labels=type_counts.index,
                                       colors=colors,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       textprops={'fontsize': 10})
    
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax2.set_title("Pie Chart - Udeo po tipu", fontsize=14, fontweight='bold')
    
   
    legend_labels = [f'{label}: {type_counts[label]} ({type_counts[label]/total*100:.1f}%)' 
                    for label in type_counts.index]
    ax2.legend(wedges, legend_labels, 
               title="Tipovi fajlova",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1),
               fontsize=9)
    
    # ============= STATISTIKA ISPOD PIE CHARTA =============
    
    # Napravi tekst sa statistikama
    textstr = f"""Statistike tipova fajlova:
    {'='*40}
    Ukupno fajlova: {total:,}
    Različitih tipova: {len(type_counts)}"""

    
    
    # Dodaj tekst ispod pie charta
    fig.text(0.75, 0.18, textstr, fontsize=9,
             ha='center',  
             va='top',     
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8,
                      edgecolor='navy', linewidth=1.5))
    
   
    plt.suptitle(f"DISTRIBUCIJA FAJLOVA PO TIPU", 
                 fontsize=18, fontweight='bold', y=0.98)
    
   
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if out_dir:
        plt.savefig(f"{out_dir}/file_types_distribution.png", 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    



def caltplott(out_dir):
    import pandas as pd
    import matplotlib.pyplot as plt
    import calplot
    from pathlib import Path

    # 1. Učitaj dnevnu statistiku (po accessed)
    df_days = groupBydays(out_dir)

    if df_days is None or df_days.empty:
        print("⚠️ Nema podataka za crtanje calplot-a")
        return

    # 2. Priprema za calplot (OBAVEZNA konverzija u datetime)
    df_days['access_date'] = pd.to_datetime(df_days['access_date'], errors='coerce')
    df_days = df_days.dropna(subset=['access_date'])

    series = df_days.set_index('access_date')['access_count']

    # 3. Crtanje
    fig, ax = calplot.calplot(
        series,
        cmap='Blues',
        figsize=(16, 6),
        suptitle='Kalendar poslednjeg pristupa fajlovima (po danima)'
    )

    # 4. Snimanje
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(out_dir) / 'calendar_accessed_calplot.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"✅ Grafik sačuvan: {output_path}")




#ovo je stari groupByDays
def groupBydays(filePath):
    # Load the CSV file with explicit date parsing
    df = pd.read_csv(filePath + "\\metadata.csv",
        # Automatically parse the 'accessed' column during loading
        parse_dates=['accessed'],
        # The datetime format is YYYY-MM-DD HH:MM:SS
        date_format='%Y-%m-%d %H:%M:%S'
    )

    # Verify that the 'accessed' column is a datetime type
    print(f"Column 'accessed' has data type: {df['accessed'].dtype}")

    # Count rows with and without valid dates to diagnose
    na_count = df['accessed'].isna().sum()
    print(f"Number of rows with invalid/NA dates: {na_count}")
    print(f"Total rows in original data: {len(df)}")

    # Check a few sample values to confirm
    print("\nSample of 'accessed' column values:")
    print(df['accessed'].head())

    # Remove any rows where 'accessed' is invalid (should be none if parsing succeeded)
    df_clean = df.dropna(subset=['accessed'])

    # Extract the date part (year-month-day) from the timestamp
    df_clean['access_date'] = df_clean['accessed'].dt.date

    # Count accesses per day using groupby
    daily_accesses = df_clean.groupby('access_date').size().reset_index(name='access_count')

    # Create a complete date range to fill missing days
    start_date = df_clean['access_date'].min()
    end_date = df_clean['access_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Merge with your daily counts to fill in zeros for days with no accesses
    all_dates_df = pd.DataFrame({'access_date': all_dates.date})
    daily_accesses_full = all_dates_df.merge(
        daily_accesses,
        on='access_date',
        how='left'
    ).fillna(0)

    # Ensure counts are integers
    daily_accesses_full['access_count'] = daily_accesses_full['access_count'].astype(int)

    # Sort by date for a cleaner output
    daily_accesses_full = daily_accesses_full.sort_values('access_date')

    # Save to a new CSV file
    daily_accesses_full.to_csv("daily_accesses.csv", index=False)

    print(f"\nAnalysis complete. Dates range from {start_date} to {end_date}.")
    print(f"Days analyzed: {len(daily_accesses_full)}")
    print("Sample of results:")
    print(daily_accesses_full.head())

    return daily_accesses_full



import pandas as pd
import calplot
import matplotlib.pyplot as plt
from pathlib import Path

def plot_with_calplot(df, out_dir, date_col='accessed'):
    df[date_col] = pd.to_datetime(df[date_col])

    # Broj pristupa po danu
    daily = (
        df.groupby(df[date_col].dt.date)
          .size()
          .rename('count')
    )

    # calplot zahteva DateTimeIndex
    daily.index = pd.to_datetime(daily.index)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # GitHub-style calendar
    fig, ax = calplot.calplot(
        daily,
        cmap='Greens',
        figsize=(16, 6),
        suptitle='GitHub-style kalendar pristupa fajlovima'
    )

    plt.savefig(Path(out_dir) / 'github_calendar_calplot.png', dpi=200)
    plt.show()




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
    output_file = out_dir / 'metadata.csv'
    df.to_csv(output_file, index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')


    print(f'Metapodaci sačuvani u {args.out_dir}/metadata.csv')

    if args.detect_iqr:
        anomalies = detect_anomalies_iqr(df)
        anomalies.to_csv(Path(args.out_dir)/'anomalies.csv', index=False)
        print(f'Detekovano {len(anomalies)} anomalija. Sačuvano u anomalies.csv')

    if args.visualize:
        print("visualize je true")
        plot_file_types(df, args.out_dir)
        plot_file_sizes1(df, args.out_dir)
        caltplott(args.out_dir)
        #plot_github_style_calendar(df,args.out_dir)
        detect_anomalies(df, args.out_dir)
        print(f'Grafički prikazi sačuvani u {args.out_dir}/')

if __name__ == '__main__':
    main()
