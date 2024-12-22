import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FantasyRecommenderBasic:
    def __init__(self, data):
        """
        Initializes the FantasyRecommenderBasic class.

        Args:
            data (pd.DataFrame): A DataFrame containing player statistics.
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
            'steals': 0.12,
            'blocks': 0.12,
            'fg_pct': 0.10,
            'fg3_pct': 0.05,
            'ft_pct': 0.05
        }

    def _normalize_metrics(self):
        """
        Normalizes the key metrics using MinMax scaling.

        Returns:
            pd.DataFrame: Normalized player statistics.
        """
        scaler = MinMaxScaler()
        return pd.DataFrame(
            scaler.fit_transform(self.data[self.key_metrics]),
            columns=self.key_metrics
        )

    def _calculate_composite_scores(self, normalized_stats):
        """
        Calculates composite scores for each player based on normalized metrics and weights.

        Args:
            normalized_stats (pd.DataFrame): Normalized statistics for each player.

        Returns:
            pd.Series: Composite scores for each player.
        """
        composite_scores = pd.Series(0, index=normalized_stats.index)
        for metric, weight in self.weights.items():
            composite_scores += normalized_stats[metric] * weight
        return composite_scores

    def create_rank_based_recommendations(self, top_n=5):
        """
        Creates rank-based recommendations using key basketball metrics.

        Args:
            top_n (int): The number of top recommendations to return.

        Returns:
            pd.DataFrame: A DataFrame sorted by composite scores in descending order.
        """
        # Normalize metrics
        normalized_stats = self._normalize_metrics()

        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(normalized_stats)

        # Create rankings DataFrame
        rankings_df = pd.DataFrame({
            'player_id': self.data['player_id'],
            'player_name': self.data['player_name'],
            'season': self.data['season'],
            'team': self.data['team'],
            'composite_score': composite_scores,
            'traded': self.data['traded']
        })

        # Keep only the best composite score for each player
        best_scores = rankings_df.loc[rankings_df.groupby('player_id')['composite_score'].idxmax()]

        # Sort by composite score in descending order
        rankings = best_scores.sort_values('composite_score', ascending=False).reset_index(drop=True)

        return rankings.head(top_n)
