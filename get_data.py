from BasketballAPI import BasketBallAPI
from basketball_etl import etl_basketball_data


if __name__ == "__main__":
    # Initialize API and get data
    api = BasketBallAPI()
    players = api.get_players()
    stats = api.get_stats()

    
    df_combined = etl_basketball_data(players, stats)

    print("\nSample of combined data:")
    print(df_combined.head())

    print('----------------------------------')
    print('--------Summary Statistics--------')
    print("")
    print("\nData summary:")
    print(f"Total unique players: {df_combined['player_id'].nunique()}")
    print(f"Seasons covered: {df_combined['season'].nunique()}")
    print(f"Total teams: {df_combined['team'].nunique()}")
    print("\nArchetype distribution:")
    print(df_combined['archetype'].value_counts())
