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

def main():
    proba()

if __name__ == '__main__':
    main()
