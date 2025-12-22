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

def plot_github_style_calendar(df, out_dir, date_col='accessed'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    df[date_col] = pd.to_datetime(df[date_col])

    # Broj pristupa po danu
    daily = (
        df.groupby(df[date_col].dt.date)
          .size()
          .rename("count")
          .reset_index()
          .rename(columns={date_col: "date"})
    )

    daily['date'] = pd.to_datetime(daily['date'])

    # Popuni sve dane
    all_days = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(all_days, fill_value=0)
    daily.index.name = 'date'
    daily.reset_index(inplace=True)

    
    daily['weekday'] = daily['date'].dt.weekday      
    daily['week'] = (
        daily['date'] - pd.to_timedelta(daily['weekday'], unit='D')
    ).dt.isocalendar().week

   
    heatmap = daily.pivot(index='weekday', columns='week', values='count')

    plt.figure(figsize=(18, 4))
    im = plt.imshow(heatmap, aspect='auto', cmap='Greens')

    # Y ose: dani u nedelji
    plt.yticks(
        ticks=np.arange(7),
        labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    )

    plt.xlabel('Weeks')
    plt.title('GitHub-style calendar heatmap – pristup fajlovima')

    cbar = plt.colorbar(im)
    cbar.set_label('Broj pristupa')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'github_calendar_heatmap.png', dpi=200)
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
        plot_modification_time(df, args.out_dir)
        plot_github_style_calendar(df,args.out_dir)
        detect_anomalies(df, args.out_dir)
        print(f'Grafički prikazi sačuvani u {args.out_dir}/')

if __name__ == '__main__':
    main()
