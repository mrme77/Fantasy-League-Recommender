import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Tuple

class FantasyRecommenderRF:
    def __init__(self, data: pd.DataFrame, n_estimators: int = 100):
        """
        Initialize Fantasy Basketball Recommender using Random Forest
        
        Args:
            data: DataFrame with player statistics
            n_estimators: Number of trees in the random forest
        """
        self.data = data.copy()
        self.key_metrics = [
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
        ]
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        self.unavailable_players = set()
        self._prepare_data()
        
    # def _prepare_data(self) -> None:
    #     """Prepare data for Random Forest model"""
    #     # Drop NaN values and scale features
    #     self.processed_data = self.data.dropna(subset=self.key_metrics).copy()
    #     self.scaled_features = self.scaler.fit_transform(
    #         self.processed_data[self.key_metrics]
    #     )
        
    #     # Create player archetypes using statistics
    #     self.rf_model.fit(
    #         self.scaled_features,
    #         self.processed_data['archetype']
    #     )
        
    #     # Set df as processed_data for easier reference
    #     self.df = self.processed_data
    def _prepare_data(self) -> None:
        """Prepare data for Random Forest model"""
        # Cleaning dataset by removing rows with NaN values in the key metrics
        new_df = self.data.dropna(subset=self.key_metrics)

        # Aggregating player stats by averaging across seasons
        aggregated_df = (
        new_df.groupby('player_id')[['points', 'rebounds', 'assists', 'steals', 
                                     'blocks', 'fg_pct', 'fg3_pct', 'ft_pct']]
        .mean()
        .reset_index()
    )

        # Add other non-statistical columns that should remain constant for each player (assuming the same for all seasons)
        aggregated_df = aggregated_df.merge(
        new_df[['player_id', 'player_name', 'team', 'archetype']].drop_duplicates(),
        on='player_id',
        how='left'
    )

        # Ensure there are no NaN values left
        aggregated_df = aggregated_df.dropna(subset=self.key_metrics)

        # Scale features using MinMaxScaler
        self.processed_data = aggregated_df.copy()
        self.scaled_features = self.scaler.fit_transform(self.processed_data[self.key_metrics])

        # Create player archetypes using statistics
        self.rf_model.fit(self.scaled_features, self.processed_data['archetype'])

        # Set df as processed_data for easier reference
        self.df = self.processed_data

        
    def get_similarity_scores(self, player_id: int) -> pd.DataFrame:
        """
        Calculate similarity scores between given player and all others
        
        Args:
            player_id: ID of the target player
            
        Returns:
            DataFrame with similarity scores for all players
        """
        # Get player features
        player_idx = self.processed_data[
            self.processed_data['player_id'] == player_id
        ].index[0]
        player_features = self.scaled_features[player_idx].reshape(1, -1)
        
        # Get probabilities for player archetype
        player_probs = self.rf_model.predict_proba(player_features)
        
        # Calculate similarities using probability distributions
        all_probs = self.rf_model.predict_proba(self.scaled_features)
        similarities = np.dot(all_probs, player_probs.T).flatten()
        
        # Create similarity DataFrame
        similarity_df = pd.DataFrame({
            'player_id': self.processed_data['player_id'],
            'player_name': self.processed_data['player_name'],
            'team': self.processed_data['team'],
            'similarity_score': similarities,
            'archetype': self.processed_data['archetype']
        })
        
        return similarity_df.sort_values('similarity_score', ascending=False)
    
    def get_recommendations(self, 
                            player_id: int, 
                            top_n: int = 5) -> pd.DataFrame:
        """
        Get player recommendations based on similarity to given player
        
        Args:
            player_id: ID of the target player
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with top similar players
        """
        similarities = self.get_similarity_scores(player_id)
        
        # Exclude unavailable players
        if self.unavailable_players:
            similarities = similarities[
                ~similarities['player_id'].isin(self.unavailable_players)
            ]
        
        # Exclude the query player
        similarities = similarities[
            similarities['player_id'] != player_id
        ]
        
        # Drop duplicate player recommendations
        recommendations = similarities.drop_duplicates(subset='player_id')
        
        return recommendations.head(top_n)
    
    def update_unavailable_players(self, player_ids: List[int]) -> None:
        """Update set of unavailable players"""
        self.unavailable_players.update(player_ids)
    
    def precisionK(self, player_id, recommendations, top_n=5):
        """
        Calculate precision at K for player recommendations.
        """
        recommender_results = set(recommendations['player_id'])

        if len(recommender_results) == 0:
            return 0.0

        # Get player archetype
        player_archetype = self.df[self.df['player_id'] == player_id]['archetype'].iloc[0]
        if not player_archetype:
            return 0.0

        if 'elite' in player_archetype:
            elite_players_ids = self.df[self.df['archetype'] != 'regular']['player_id'].unique()
            Kmetric = len(recommender_results.intersection(set(elite_players_ids))) / len(recommender_results)
        else:
            elite_players_ids = self.df[self.df['archetype'] != 'regular']['player_id'].unique()
            regular_players_ids = self.df[self.df['archetype'] == 'regular']['player_id'].unique()
            elite_recs = len(recommender_results.intersection(set(elite_players_ids)))
            regular_recs = len(recommender_results.intersection(set(regular_players_ids)))
            Kmetric = (regular_recs / len(recommender_results)) - 0.1 if elite_recs == 0 else \
                  (elite_recs + regular_recs) / len(recommender_results)
            Kmetric = max(0, Kmetric)

        return Kmetric * 100

