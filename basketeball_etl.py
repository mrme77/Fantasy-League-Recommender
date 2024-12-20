import pandas as pd
from typing import List, Dict, Any
import time

def etl_basketball_data(players: List[Dict[str, Any]], stats: List[Dict[str, Any]], output_file: str = 'basketball_data.csv') -> pd.DataFrame:
    """
    ETL function to process basketball data:
    1. Convert players and stats to DataFrames
    2. Join the data
    3. Save to CSV
    
    Args:
        players: List of player dictionaries
        stats: List of stats dictionaries
        output_file: Name of output CSV file
    
    Returns:
        DataFrame with combined data
    """
    print("\nStarting ETL process...")
    start_time = time.time()
    
    # Convert to DataFrames
    print("Converting data to DataFrames...")
    df_players = pd.DataFrame(players)
    df_stats = pd.DataFrame(stats)
    
    # Merge DataFrames
    print("Joining player and stats data...")
    df_combined = df_stats.merge(
        df_players,
        on='player_id',
        how='left'
    )
    
    # Reorder columns for better readability
    columns_order = [
        'player_id', 'player_name', 'archetype', 'season', 'team',
        'games_played', 'minutes', 'points', 'rebounds', 'assists',
        'steals', 'blocks', 'fg_pct', 'fg3_pct', 'ft_pct'
    ]
    df_combined = df_combined[columns_order]
    
    # Save to CSV
    print(f"Saving data to {output_file}...")
    df_combined.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"ETL process completed in {round(end_time - start_time, 2)} seconds")
    print(f"Total records: {len(df_combined)}")
    
    return df_combined

# Example usage:
if __name__ == "__main__":
    # Initialize API and get data
    api = BasketballAPI()
    players = api.get_players()
    stats = api.get_stats()
    
    # Run ETL process
    df_combined = etl_basketball_data(players, stats)
    
    # Display sample of the data
    print("\nSample of combined data:")
    print(df_combined.head())
    
    # Display basic statistics
    print("\nData summary:")
    print(f"Total unique players: {df_combined['player_id'].nunique()}")
    print(f"Seasons covered: {df_combined['season'].nunique()}")
    print(f"Total teams: {df_combined['team'].nunique()}")
    print("\nArchetype distribution:")
    print(df_combined['archetype'].value_counts())