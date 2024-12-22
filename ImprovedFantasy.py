# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# class FantasyRecommender:
#     def __init__(self):
#         """Initialize the recommender with an empty set of selected players"""
#         self.selected_players = set()
        
#     def create_rank_based_recommendations(self, stats_df):
#         """Create rank-based recommendations using key basketball metrics"""
#         # 1. Select Key Metrics
#         key_metrics = [
#             'points',
#             'rebounds',
#             'assists',
#             'steals',
#             'blocks',
#             'fg_pct',
#             'fg3_pct',
#             'ft_pct'
#         ]
        
#         # 2. Normalize Metrics
#         scaler = MinMaxScaler()
#         normalized_stats = pd.DataFrame(
#             scaler.fit_transform(stats_df[key_metrics]),
#             columns=key_metrics
#         )
        
#         # 3. Assign Weights to Metrics
#         weights = {
#             'points': 0.25,
#             'rebounds': 0.15,
#             'assists': 0.15,
#             'steals': 0.12,
#             'blocks': 0.12,
#             'fg_pct': 0.10,
#             'fg3_pct': 0.05,
#             'ft_pct': 0.05,
#             'traded': 0.01
#         }
        
#         # 4. Calculate Composite Score
#         composite_scores = pd.Series(0, index=normalized_stats.index)
#         for metric, weight in weights.items():
#             composite_scores += normalized_stats[metric] * weight
        
#         # 5. Create Final Rankings
#         rankings = pd.DataFrame({
#             'player_id': stats_df['player_id'],
#             'player_name': stats_df['player_name'],
#             'composite_score': composite_scores,
#             'traded': stats_df['traded']
#         }).sort_values('composite_score', ascending=False)
        
#         return rankings
    
#     def select_player(self, player_input):
#         """
#         Mark player(s) as selected. Can handle both single player ID and list of player IDs.
#         Args:
#             player_input: Union[int, List[int]] - Either a single player ID or a list of player IDs
#         """
#         try:
#             if isinstance(player_input, (list, tuple, set)):
#                 # Ensure the input is iterable and update the selected players set
#                 self.selected_players.update(player_input)
#             else:
#                 # Handle single player ID
#                 self.selected_players.add(player_input)
#         except Exception as e:
#             print(f"Error selecting player(s): {e}")
#             raise

#     def get_recommendations(self, rankings_df, n_recommendations=5, traded=None):
#         """
#         Get recommendations excluding all selected players
#         Args:
#             rankings_df: DataFrame containing player rankings
#             n_recommendations: int, number of recommendations to return
#             traded: bool or None, filter by traded status
#         Returns:
#             DataFrame with top N recommended players
#         """
#         # Filter out selected players
#         available_mask = ~rankings_df['player_id'].isin(self.selected_players)
        
#         # Apply traded filter if specified
#         if traded is not None:
#             filtered_rankings = rankings_df[available_mask & (rankings_df['traded'] == traded)]
#         else:
#             filtered_rankings = rankings_df[available_mask]
            
#         return filtered_rankings.head(n_recommendations)
    
#     def get_selected_players(self):
#         """Return list of selected player IDs"""
#         return list(self.selected_players)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class FantasyRecommender:
    """
    Improved FantasyRecommender Class with Proper Handling of Duplicate Players
    across different seasons and teams.
    """

    def __init__(self):
        """
        Initializes the FantasyRecommender with an empty set of selected players.
        """
        self.selected_players = set()

    def create_rank_based_recommendations(self, stats_df):
        """
        Creates rank-based recommendations using key basketball metrics.

        Args:
            stats_df (pd.DataFrame): A DataFrame containing players' statistics.

        Returns:
            pd.DataFrame: A DataFrame sorted by composite scores in descending order,
                          with duplicates removed (keeping the highest score for each player).
        """
        # 1. Select Key Metrics
        key_metrics = [
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
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
            'steals': 0.12,
            'blocks': 0.12,
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
            'season': stats_df['season'],
            'team': stats_df['team'],
            'composite_score': composite_scores,
            'traded': stats_df['traded']
        })

        # Drop duplicate player_id entries, keeping the one with the highest composite score
        rankings = rankings.sort_values('composite_score', ascending=False)
        rankings = rankings.drop_duplicates(subset='player_id', keep='first')

        return rankings

    def get_recommendations(self, rankings_df, n_recommendations=5, traded=None):
        """
        Retrieves the top N recommendations, optionally filtered by traded status,
        but excluding already selected players.

        Args:
            rankings_df (pd.DataFrame): DataFrame of player rankings.
            n_recommendations (int, optional): Number of recommendations to return. Default is 5.
            traded (bool, optional): Filter players by traded status. Default is None.

        Returns:
            pd.DataFrame: A DataFrame containing the top N recommended players.
        """
        # Apply traded filter if specified
        if traded is not None:
            filtered_rankings = rankings_df[rankings_df['traded'] == traded]
        else:
            filtered_rankings = rankings_df

        # Exclude already selected players
        filtered_rankings = filtered_rankings[~filtered_rankings['player_id'].isin(self.selected_players)]

        # Return top N recommendations
        return filtered_rankings.head(n_recommendations)

    def select_player(self, player_ids):
        """
        Marks one or more players as selected by adding their IDs to the selected_players set.

        Args:
            player_ids (list or int): A list of player IDs or a single player ID to be selected.
        """
        if not isinstance(player_ids, list):
            player_ids = [player_ids]
        self.selected_players.update(player_ids)

    def reset_selected_players(self):
        """
        Resets the selected players set.
        """
        self.selected_players.clear()

    def get_selected_players(self):
        """
        Returns the list of player IDs that have already been selected.

        Returns:
            list: A list of player IDs in the selected_players set.
        """
        return list(self.selected_players)