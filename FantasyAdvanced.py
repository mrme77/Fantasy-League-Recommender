# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# class FantasyReccomenderAdvanced:
#     def __init__(self, data, unavailable_players=None):
#         """
#         Initializes the FantasyReccomenderAdvanced with player statistics data and unavailable players.
        
#         Args:
#             data (pd.DataFrame): A DataFrame containing players' statistics.
#             unavailable_players (list): A list of player_ids that are unavailable for selection.
#         """
#         self.data = data
#         self.key_metrics = [
#             'points', 'rebounds', 'assists', 'steals', 'blocks',
#             'fg_pct', 'fg3_pct', 'ft_pct'
#         ]
#         self.weights = {
#             'points': 0.25,
#             'rebounds': 0.15,
#             'assists': 0.15,
#             'steals': 0.125,
#             'blocks': 0.125,
#             'fg_pct': 0.10,
#             'fg3_pct': 0.05,
#             'ft_pct': 0.05
#         }
#         self.unavailable_players = unavailable_players if unavailable_players else []

#     def create_rank_based_recommendations(self, top_n=5):
#         """
#         Creates rank-based recommendations using key basketball metrics, excluding unavailable players.

#         Args:
#             top_n (int): The number of top players to recommend based on composite scores.
        
#         Returns:
#             pd.DataFrame: A DataFrame sorted by composite scores in descending order,
#                           with only the best score for each player.
#         """
#         # Dropping rows with NaN values in the key metrics
#         data = self.data.dropna(subset=self.key_metrics)

#         # Normalizing Metrics
#         scaler = MinMaxScaler()
#         normalized_stats = pd.DataFrame(
#             scaler.fit_transform(data[self.key_metrics]),
#             columns=self.key_metrics
#         )

#         # Calculating Composite Score
#         composite_scores = pd.Series(0, index=normalized_stats.index)
#         for metric, weight in self.weights.items():
#             composite_scores += normalized_stats[metric] * weight

#         # Rankings DataFrame
#         initial_rankings = pd.DataFrame({
#             'player_id': data['player_id'],
#             'player_name': data['player_name'],
#             'team': data['team'],
#             'composite_score': composite_scores,
#             'traded': data['traded']
#         })

#         initial_rankings['max_composite_score'] = initial_rankings.groupby('player_id')['composite_score'].transform('max')

#         best_scores = initial_rankings[initial_rankings['composite_score'] == initial_rankings['max_composite_score']]

#         best_scores = best_scores.drop(columns=['max_composite_score'])

#         # Exclude unavailable players
#         if self.unavailable_players:
#             best_scores = best_scores[~best_scores['player_id'].isin(self.unavailable_players)]

#         # Sorting by Composite Score in Descending Order
#         rankings = best_scores.sort_values('composite_score', ascending=False).reset_index(drop=True)

#         return rankings.head(top_n)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Optional

class FantasyRecommenderAdvanced:
    def __init__(self, data: pd.DataFrame, unavailable_players: Optional[List[int]] = None):
        """
        Initialize Fantasy Basketball Recommender
        
        Args:
            data: DataFrame with player statistics
            unavailable_players: List of player_ids that are unavailable
        """
        self.data = data.copy()  # Create copy to avoid modifications to original
        self.key_metrics: List[str] = [
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
        ]
        self.weights: Dict[str, float] = {
            'points': 0.25,
            'rebounds': 0.15,
            'assists': 0.15,
            'steals': 0.125,
            'blocks': 0.125,
            'fg_pct': 0.10,
            'fg3_pct': 0.05,
            'ft_pct': 0.05
        }
        self.unavailable_players = unavailable_players if unavailable_players else []
        
    def create_rank_based_recommendations(self, top_n: int = 5) -> pd.DataFrame:
        """
        Create rank-based recommendations using key basketball metrics
        
        Args:
            top_n: Number of top players to recommend
            
        Returns:
            DataFrame with top recommended players
        """
        # Create copy of data and drop NaN values
        working_data = self.data.dropna(subset=self.key_metrics).copy()
        
        # Normalize metrics using MinMaxScaler
        scaler = MinMaxScaler()
        normalized_stats = pd.DataFrame(
            scaler.fit_transform(working_data[self.key_metrics]),
            columns=self.key_metrics,
            index=working_data.index
        )
        
        # Calculate composite score
        composite_scores = pd.Series(0.0, index=normalized_stats.index)
        for metric, weight in self.weights.items():
            composite_scores += normalized_stats[metric] * weight
            
        # Create rankings DataFrame with required columns
        rankings = pd.DataFrame({
            'player_id': working_data['player_id'],
            'player_name': working_data['player_name'],
            'team': working_data['team'],
            'composite_score': composite_scores
        })
        
        # Get best scores for each player
        best_scores = (
            rankings.loc[rankings.groupby('player_id')['composite_score']
            .idxmax()]
            .reset_index(drop=True)
        )
        
        # Filter out unavailable players
        if self.unavailable_players:
            best_scores = best_scores[
                ~best_scores['player_id'].isin(self.unavailable_players)
            ]
        
        # Sort and return top N players
        return (
            best_scores
            .sort_values('composite_score', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
    
    def get_player_stats(self, player_id: int) -> pd.DataFrame:
        """
        Get detailed stats for a specific player
        
        Args:
            player_id: Player's ID
            
        Returns:
            DataFrame with player's statistics
        """
        return (
            self.data[self.data['player_id'] == player_id]
            [['player_name', 'team'] + self.key_metrics]
            .reset_index(drop=True)
        )
    
    def compare_players(self, player_ids: List[int]) -> pd.DataFrame:
        """
        Compare statistics of multiple players
        
        Args:
            player_ids: List of player IDs to compare
            
        Returns:
            DataFrame with compared statistics
        """
        return (
            self.data[self.data['player_id'].isin(player_ids)]
            [['player_name', 'team'] + self.key_metrics]
            .reset_index(drop=True)
        )

# Example usage
def main():
    # Load your data
    df = pd.read_csv('basketball_data.csv')
    
    # Initialize recommender
    recommender = FantasyRecommenderAdvanced(df)
    
    # Get top 5 recommendations
    top_players = recommender.create_rank_based_recommendations(top_n=5)
    print("\nTop 5 Recommended Players:")
    print(top_players)
    
    # Compare specific players
    player_comparison = recommender.compare_players([1, 2, 3])  # Example player IDs
    print("\nPlayer Comparison:")
    print(player_comparison)

if __name__ == "__main__":
    main()
