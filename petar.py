import pandas as pd
import calplot
import matplotlib.pyplot as plt
import pandas as pd
def fja():

    df = pd.read_csv("C:\\FAX\\MASTER\\Digitalna forenzika\\Digital-forensics\\24122025\\metadata.csv")

    # Pretvaranje 'accessed' kolone u datetime, sa dayfirst=True i errors='coerce' za nevalidne datume
    df['accessed'] = pd.to_datetime(df['accessed'], dayfirst=True, errors='coerce')

    # Uklanjanje redova gde 'accessed' nije validan datum
    df = df.dropna(subset=['accessed'])

    # Kreiranje kolone koja sadrži samo datum
    df['access_date'] = df['accessed'].dt.date

    # Grupisanje po datumu i brojanje pristupa
    daily_accesses = df.groupby('access_date').size().reset_index(name='access_count')

    # Kreiranje kompletne liste datuma od najranijeg do najnovijeg
    start_date = df['access_date'].min()
    end_date = df['access_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date)

    print(start_date)
    print(end_date)

    # Merge sa postojećim podacima da bi svi datumi bili prisutni
    all_dates_df = pd.DataFrame(all_dates, columns=['access_date'])
    all_dates_df['access_date'] = all_dates_df['access_date'].dt.date
    daily_accesses_full = all_dates_df.merge(daily_accesses, on='access_date', how='left').fillna(0)

    # Pretvaranje broja pristupa u integer
    daily_accesses_full['access_count'] = daily_accesses_full['access_count'].astype(int)

    # Upis u CSV fajl
    daily_accesses_full.to_csv("daily_accesses.csv", index=False)


import pandas as pd
from datetime import datetime

def fja1():
    # Load the CSV file with explicit date parsing
    df = pd.read_csv(
        "C:\\FAX\\MASTER\\Digitalna forenzika\\Digital-forensics\\24122025\\metadata.csv",
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


def main():
   fja1()

if __name__ == '__main__':
    main()
