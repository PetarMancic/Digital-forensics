import pandas as pd
import calplot
import matplotlib.pyplot as plt

def proba():
    df = pd.read_csv(r"C:\FAX\MASTER\Digitalna forenzika\Digital-forensics\OsnovnePodaci\metadata.csv", parse_dates=["accessed"])

    # izdvajanje samo datuma (bez vremena)
    df["access_date"] = df["accessed"].dt.date

    # broj pristupa po datumu
    daily_activity = df.groupby("access_date").size()

    # calplot zahteva Series sa DateTimeIndex
    daily_activity.index = pd.to_datetime(daily_activity.index)
    daily_activity = daily_activity.sort_index()

    custom_colorscale= [
        (0.0, "white"),
        (0.5,"#FFDDC1"),
        (1.0, "#FF5733")
    ]

    # crtanje kalendarskog heatmap-a
    calplot.calplot(
        daily_activity,
        cmap='YlGn',
        suptitle='Aktivnost pristupa fajlovima po danima 2025',
        figsize=(16, 4)
    )

    # sačuvaj graf u PNG fajl
    plt.savefig("aktivnost_po_danima.png", dpi=300, bbox_inches='tight')

    # prikaži graf
    plt.show()


import pandas as pd

def count_file_access_per_day(csv_path):
    """
    Učitava CSV sa metapodacima i vraća listu tuple (datum, broj pristupa fajlovima tog dana)
    """
    # Učitaj CSV
    df = pd.read_csv(csv_path)
    
    # Pretvori kolonu 'accessed' u datetime
    df['accessed'] = pd.to_datetime(df['accessed'])
    
    # Grupisanje po datumu i brojanje koliko puta je pristupano
    daily_counts = df.groupby(df['accessed'].dt.date).size()
    
    # Pretvori u listu tuple (datum, broj_pristupa)
    result = list(daily_counts.items())
    
    # Sortiraj po datumu
    result.sort(key=lambda x: x[0])
    
    return result

# Primer korišćenja
counts = count_file_access_per_day("metadata.csv")
for date, num in counts[:10]:  # prikaži prvih 10 dana
    print(date, num)




def main():
   # proba()
    count_file_access_per_day("D:\Fax\Digital-forensics\Petar\metadata.csv")

if __name__ == '__main__':
    main()
