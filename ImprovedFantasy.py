import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FantasyRecommender:
    def __init__(self):
        """Initialize the recommender with an empty set of selected players"""
        self.selected_players = set()
        
    def create_rank_based_recommendations(self, stats_df):
        """Create rank-based recommendations using key basketball metrics"""
        # 1. Select Key Metrics
        key_metrics = [
            'points',
            'rebounds',
            'assists',
            'steals',
            'blocks',
            'fg_pct',
            'fg3_pct',
            'ft_pct'
        ]
        
        # 2. Normalize Metrics
        scaler = MinMaxScaler()
        normalized_stats = pd.DataFrame(
            scaler.fit_transform(stats_df[key_metrics]),
            columns=key_metrics
        )
        
        # 3. Assign Weights to Metrics
        weights = {
            'points': 0.25,
            'rebounds': 0.15,
            'assists': 0.15,
            'steals': 0.125,
            'blocks': 0.125,
            'fg_pct': 0.10,
            'fg3_pct': 0.05,
            'ft_pct': 0.05
        }
        
        # 4. Calculate Composite Score
        composite_scores = pd.Series(0, index=normalized_stats.index)
        for metric, weight in weights.items():
            composite_scores += normalized_stats[metric] * weight
        
        # 5. Create Final Rankings
        rankings = pd.DataFrame({
            'player_id': stats_df['player_id'],
            'player_name': stats_df['player_name'],
            'composite_score': composite_scores,
            'traded': stats_df['traded']
        }).sort_values('composite_score', ascending=False)
        
        return rankings
    
    def select_player(self, player_input):
        """
        Mark player(s) as selected. Can handle both single player ID and list of player IDs.
        Args:
            player_input: Union[int, List[int]] - Either a single player ID or a list of player IDs
        """
        try:
            if isinstance(player_input, (list, tuple, set)):
                # Ensure the input is iterable and update the selected players set
                self.selected_players.update(player_input)
            else:
                # Handle single player ID
                self.selected_players.add(player_input)
        except Exception as e:
            print(f"Error selecting player(s): {e}")
            raise

    def get_recommendations(self, rankings_df, n_recommendations=5, traded=None):
        """
        Get recommendations excluding all selected players
        Args:
            rankings_df: DataFrame containing player rankings
            n_recommendations: int, number of recommendations to return
            traded: bool or None, filter by traded status
        Returns:
            DataFrame with top N recommended players
        """
        # Filter out selected players
        available_mask = ~rankings_df['player_id'].isin(self.selected_players)
        
        # Apply traded filter if specified
        if traded is not None:
            filtered_rankings = rankings_df[available_mask & (rankings_df['traded'] == traded)]
        else:
            filtered_rankings = rankings_df[available_mask]
            
        return filtered_rankings.head(n_recommendations)
    
    def get_selected_players(self):
        """Return list of selected player IDs"""
        return list(self.selected_players)