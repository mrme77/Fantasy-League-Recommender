import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FantasyRecommenderAdvanced:
    def __init__(self, data):
        """
        Initializes the FantasyRecommenderAdvanced with player statistics data.
        
        Args:
            data (pd.DataFrame): A DataFrame containing players' statistics.
        """
        self.data = data
        self.key_metrics = [
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
        ]
        self.weights = {
            'points': 0.25,
            'rebounds': 0.15,
            'assists': 0.15,
            'steals': 0.125,
            'blocks': 0.125,
            'fg_pct': 0.10,
            'fg3_pct': 0.05,
            'ft_pct': 0.05
        }

    def create_rank_based_recommendations(self, top_n=5, exclude_players=None):
        """
        Creates rank-based recommendations using key basketball metrics, 
        with optional exclusion of specific players provided by the user.

        Args:
            top_n (int): The number of top players to recommend.
            exclude_players (list, optional): List of player IDs to exclude from recommendations.
                                              If None, no players are excluded.

        Returns:
            pd.DataFrame: A DataFrame sorted by composite scores in descending order,
                          filtered to exclude specified players.
        """
        # Dropping rows with NaN values in key metrics
        data = self.data.dropna(subset=self.key_metrics)

        # Exclude players if a list is provided
        if exclude_players:
            data = self.data.dropna(subset=self.key_metrics)
            data = data[~data['player_id'].isin(exclude_players)]
            print(f"Excluding players with IDs: {exclude_players}")

        # Normalize metrics
        scaler = MinMaxScaler()
        normalized_stats = pd.DataFrame(
            scaler.fit_transform(data[self.key_metrics]),
            columns=self.key_metrics
        )

        # Calculate composite scores
        composite_scores = pd.Series(0, index=normalized_stats.index)
        for metric, weight in self.weights.items():
            composite_scores += normalized_stats[metric] * weight

        # Create rankings DataFrame
        rankings = pd.DataFrame({
            'player_id': data['player_id'],
            'player_name': data['player_name'],
            'team': data['team'],
            'composite_score': composite_scores,
            'traded': data['traded']
        })

        # Sort by composite score in descending order
        rankings = rankings.sort_values('composite_score', ascending=False).reset_index(drop=True)

        return rankings.head(top_n)
