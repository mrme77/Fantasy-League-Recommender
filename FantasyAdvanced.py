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

    def create_rank_based_recommendations(self, unavailable_players=None, top_n=5):
        """
        Creates rank-based recommendations using key basketball metrics, excluding unavailable players.

        Args:
            unavailable_players (list): List of player IDs to exclude from recommendations.
            top_n (int): The number of top players to recommend based on composite scores.

        Returns:
            pd.DataFrame: A DataFrame sorted by composite scores in descending order,
                          excluding unavailable players.
        """
        # Dropping rows with NaN values in the key metrics
        data = self.data.dropna(subset=self.key_metrics)

        # Normalizing Metrics
        scaler = MinMaxScaler()
        normalized_stats = pd.DataFrame(
            scaler.fit_transform(data[self.key_metrics]),
            columns=self.key_metrics
        )

        # Calculating Composite Score
        composite_scores = pd.Series(0, index=normalized_stats.index)
        for metric, weight in self.weights.items():
            composite_scores += normalized_stats[metric] * weight

        # Rankings DataFrame
        initial_rankings = pd.DataFrame({
            'player_id': data['player_id'],
            'player_name': data['player_name'],
            'team': data['team'],
            'composite_score': composite_scores,
            'traded': data['traded']
        })

        # Keeping only the best score for each player
        initial_rankings['max_composite_score'] = initial_rankings.groupby('player_id')['composite_score'].transform('max')
        best_scores = initial_rankings[initial_rankings['composite_score'] == initial_rankings['max_composite_score']]
        best_scores = best_scores.drop(columns=['max_composite_score'])

        # Sorting by Composite Score in Descending Order
        rankings = best_scores.sort_values('composite_score', ascending=False).reset_index(drop=True)

        # Exclude unavailable players if provided
        if unavailable_players:
            rankings = rankings[~rankings['player_id'].isin(unavailable_players)]

        return rankings.head(top_n)
