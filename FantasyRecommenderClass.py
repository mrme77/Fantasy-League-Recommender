from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class FantasyRecommender:
    """
FantasyRecommender Class

This class provides a rank-based recommendation system for selecting players in a fantasy basketball league. 
It ranks players based on a weighted composite score calculated using their key performance metrics. The system 
allows users to make recommendations, select players, and retrieve updated rankings that exclude already 
selected players.

Attributes:
    selected_players (set): A set to keep track of players that have already been selected.

Methods:
    __init__():
        Initializes the FantasyRecommender with an empty set of selected players.

    create_rank_based_recommendations(stats_df):
        Creates rank-based recommendations for players using key basketball metrics.
        
        Args:
            stats_df (pd.DataFrame): A DataFrame containing players' statistics, including the following columns:
                - 'points', 'rebounds', 'assists', 'steals', 'blocks',
                  'fg_pct' (field goal percentage), 'fg3_pct' (three-point percentage), 'ft_pct' (free throw percentage),
                  'player_id', and 'traded'.
        
        Returns:
            pd.DataFrame: A DataFrame sorted by composite scores in descending order. The columns include:
                - 'player_id': Unique identifier for players.
                - 'player_name': Name of the player.
                - 'composite_score': Weighted score based on player metrics.
                - 'traded': Boolean indicating whether the player has been traded.

    get_recommendations(rankings_df, n_recommendations=5, traded=None):
        Retrieves the top N player recommendations, filtered by traded status and excluding already selected players.
        
        Args:
            rankings_df (pd.DataFrame): DataFrame of player rankings created by `create_rank_based_recommendations`.
            n_recommendations (int, optional): The number of recommendations to return. Default is 5.
            traded (bool, optional): If specified, filters players based on their traded status. Default is None.
        
        Returns:
            pd.DataFrame: A DataFrame containing the top N recommended players.

    select_player(player_id):
        Marks a player as selected by adding their player ID to the `selected_players` set.
        
        Args:
            player_id (int): Unique identifier for the player being selected.

    get_selected_players():
        Returns the list of player IDs that have already been selected.
        
        Returns:
            list: A list of player IDs in the `selected_players` set.

"""

    def __init__(self):
        self.selected_players = set()  # Keep track of selected players
        
    def create_rank_based_recommendations(self, stats_df):
        """
        Create rank-based recommendations using key basketball metrics
        """
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
            'player_name': stats_df['player_name'],  # Add player name
            'composite_score': composite_scores,
            'traded': stats_df['traded']
        }).sort_values('composite_score', ascending=False)
        
        return rankings
    
    def get_recommendations(self, rankings_df, n_recommendations=5, traded=None):
        """
        Get top N recommendations, optionally filtered by traded status,
        excluding already selected players
        """
        # Create mask for unselected players
        unselected_mask = ~rankings_df['player_id'].isin(self.selected_players)
        
        # Apply traded filter if specified
        if traded is not None:
            filtered_rankings = rankings_df[unselected_mask & (rankings_df['traded'] == traded)]
        else:
            filtered_rankings = rankings_df[unselected_mask]
            
        return filtered_rankings.head(n_recommendations)
    
    def select_player(self, player_id):
        """
        Mark a player as selected
        """
        self.selected_players.add(player_id)
        
    def get_selected_players(self):
        """
        Get list of selected players
        """
        return list(self.selected_players)