import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class svd_approach:
    def __init__(self, data: pd.DataFrame, unavailable_players: List[int] = None, n_components: int = 8):
        """
        Initialize Fantasy Basketball Recommender using Singular Value Decomposition (SVD)
        
        Args:
            data: DataFrame with player statistics, including 'player_id', 'player_name', 'team', and 'archetype'
            unavailable_players: List of player IDs to exclude from recommendations
            n_components: Number of components for SVD model
        """
        self.data = data.copy()
        self.unavailable_players = unavailable_players if unavailable_players else []
        self.n_components = n_components
        self.key_metrics = [
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
        ]
        
        if 'player_id' not in self.data.columns or 'season' not in self.data.columns:
            raise ValueError("Data must include 'player_id' and 'season' columns.")
        
        self.manual_split_data()
        self.prepare_data()

    def set_unavailable_players(self, player_ids: List[int]):
        """Update the list of unavailable players"""
        self.unavailable_players = player_ids
        print(f"Unavailable players set: {self.unavailable_players}")

    def get_unavailable_players(self) -> List[int]:
        """Get the list of unavailable players"""
        return self.unavailable_players

    def manual_split_data(self,test_size=0.2):
        """
        Manually split data ensuring all player IDs are in both sets.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
        """
        unique_players = self.data['player_id'].unique()
        train_idx, test_idx = [], []

        for player_id in unique_players:
            player_data = self.data[self.data['player_id'] == player_id]
            # Ensure there's enough data to split
            if len(player_data) > 1:
                split_point = int(len(player_data) * (1 - test_size))
                train_idx.extend(player_data.index[:split_point])
                test_idx.extend(player_data.index[split_point:])
            else:
                # If only one data point, assign to training set
                train_idx.extend(player_data.index)

        self.train_data = self.data.loc[train_idx]
        self.test_data = self.data.loc[test_idx]

    def prepare_data(self) -> None:
        """Prepare data for SVD model"""
        new_df = self.train_data.dropna(subset=self.key_metrics)
        aggregated_df = (
            new_df.groupby('player_id')[self.key_metrics].mean().reset_index()
        )

        aggregated_df = aggregated_df.merge(
            new_df[['player_id', 'player_name', 'team', 'archetype']].drop_duplicates(),
            on='player_id',
            how='left'
        ).dropna()

        self.processed_data = aggregated_df.copy()
        self.scaler = MinMaxScaler()
        self.scaled_features = self.scaler.fit_transform(self.processed_data[self.key_metrics])
        self.df = self.processed_data
        self.fit_svd()

    def fit_svd(self):
        """Fit SVD model to the data"""
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd_features = self.svd_model.fit_transform(self.scaled_features)

    def get_similarity_scores(self, player_id: int) -> pd.DataFrame:
        """Calculate similarity scores between given player and all others"""
        player_idx = self.processed_data[self.processed_data['player_id'] == player_id].index[0]
        player_features = self.svd_features[player_idx].reshape(1, -1)
        similarities = cosine_similarity(self.svd_features, player_features).flatten()

        similarity_df = pd.DataFrame({
            'player_id': self.processed_data['player_id'],
            'player_name': self.processed_data['player_name'],
            'team': self.processed_data['team'],
            'similarity_score': similarities,
            'archetype': self.processed_data['archetype']
        })
        return similarity_df.sort_values('similarity_score', ascending=False)

    def get_recommendations(self, player_id: int, top_n: int = 5) -> pd.DataFrame:
        """Get player recommendations based on similarity to given player"""
        similarities = self.get_similarity_scores(player_id)
        if self.unavailable_players:
            print(f"Filtering out unavailable players: {self.unavailable_players}")
            similarities = similarities[~similarities['player_id'].isin(self.unavailable_players)]
        similarities = similarities[similarities['player_id'] != player_id]
        recommendations = similarities.drop_duplicates(subset='player_id')
        return recommendations.head(top_n)
    
    def precisionK(self, player_id: int, top_n: int = 5) -> float:
        """Calculate precision at K for player recommendations"""
        recommendations = set(self.get_recommendations(player_id, top_n)['player_id'])
        if not recommendations:
            return 0.0

        player_archetype = self.df[self.df['player_id'] == player_id]['archetype'].iloc[0]
        if not player_archetype:
            return 0.0

        if 'elite' in player_archetype:
            elite_ids = set(self.df[self.df['archetype'] != 'regular']['player_id'])
            precision = len(recommendations & elite_ids) / len(recommendations)
        else:
            elite_ids = set(self.df[self.df['archetype'] != 'regular']['player_id'])
            regular_ids = set(self.df[self.df['archetype'] == 'regular']['player_id'])
            elite_recs = len(recommendations & elite_ids)
            regular_recs = len(recommendations & regular_ids)
            precision = (regular_recs / len(recommendations)) - 0.1 if elite_recs == 0 else \
                        (elite_recs + regular_recs) / len(recommendations)
            precision = max(0, precision)

        return precision * 100

