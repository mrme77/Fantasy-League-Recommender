import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class svd_approach:
    @staticmethod
    def svd_augmentated(df, top_n=5):
        """
        Perform Singular Value Decomposition (SVD) on player performance data
        and return the SVD matrix and cosine similarity matrix.
        Args:
            df (pd.DataFrame): Dataframe containing player performance data.
            top_n (int): Number of recommended players to return (default is 5).
        Returns:
            tuple: SVD matrix, cosine similarity matrix, player performance matrix.
        """
        # Cleaning dataset by removing rows with NaN values
        new_df = df.dropna(subset=[
            'points', 'rebounds', 'assists', 'steals', 'blocks',
            'fg_pct', 'fg3_pct', 'ft_pct'
        ])
        
        # Aggregating player stats by averaging across seasons
        aggregated_df = (
            new_df.groupby('player_id')[['points', 'rebounds', 'assists', 'steals', 
                                         'blocks', 'fg_pct', 'fg3_pct', 'ft_pct']]
            .mean()
            .reset_index()
        )
        
        player_performance_matrix = aggregated_df.set_index('player_id')
        
        # Normalizing data
        scaler = StandardScaler()
        normalized_matrix = scaler.fit_transform(player_performance_matrix)
        
        # Applying SVD
        svd = TruncatedSVD(n_components=3, random_state=42)
        svd_matrix = svd.fit_transform(normalized_matrix)
        
        # Calculating cosine similarity
        cosine_sim = cosine_similarity(svd_matrix)
        return svd_matrix, cosine_sim, player_performance_matrix

    @staticmethod
    def recommend_players(player_id, cosine_sim, player_performance_matrix, top_n=5):
        """
        Recommend the top N players similar to the given player ID.
        """
        if player_id not in player_performance_matrix.index:
            raise ValueError("Player ID not found in the dataset.")
        
        # Get similarity scores for the player
        idx = player_performance_matrix.index.tolist().index(player_id)
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort scores by similarity, excluding the player itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        recommended_players = [player_performance_matrix.index[i[0]] for i in sim_scores]
        return recommended_players

    @staticmethod
    def precisionK(player_id, df, cosine_sim, player_performance_matrix, top_n=5):
        """
        Calculate precision at K for player recommendations.
        """
        recommender_results = set(
            svd_approach.recommend_players(player_id, cosine_sim, player_performance_matrix, top_n)
        )
        if len(recommender_results) == 0:
            return 0.0
        
        # Get player archetype
        player_archetype = df[df['player_id'] == player_id]['archetype'].iloc[0]
        if not player_archetype:
            return 0.0
        
        if 'elite' in player_archetype:
            elite_players_ids = df[df['archetype'] != 'regular']['player_id'].unique()
            Kmetric = len(recommender_results.intersection(set(elite_players_ids))) / len(recommender_results)
        else:
            elite_players_ids = df[df['archetype'] != 'regular']['player_id'].unique()
            regular_players_ids = df[df['archetype'] == 'regular']['player_id'].unique()
            elite_recs = len(recommender_results.intersection(set(elite_players_ids)))
            regular_recs = len(recommender_results.intersection(set(regular_players_ids)))
            Kmetric = (regular_recs / len(recommender_results)) - 0.1 if elite_recs == 0 else \
                      (elite_recs + regular_recs) / len(recommender_results)
            Kmetric = max(0, Kmetric)
        
        return Kmetric * 100
