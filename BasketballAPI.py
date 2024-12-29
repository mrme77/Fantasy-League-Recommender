import random
import time
from faker import Faker
from typing import List, Dict, Any

class BasketballAPI:
    def __init__(self, players_per_season=390):
        """
A basketball analytics API that simulates and generates player and team data across multiple seasons.

This API creates a synthetic basketball dataset with the following key features:
- Generates 390 players per season across 5 seasons (2019-2024)
- Assigns players to 8 teams with detailed team information (city, state, creation date)
- Categorizes players into archetypes (elite_scorer, elite_defender, elite_playmaker, regular)
- Simulates realistic player statistics based on their archetype
- Tracks player movement between teams with a 15% movement probability
- Includes performance metrics like points, rebounds, assists with appropriate ranges
- Generates team details including unique IDs, locations, and franchise history

Returns:
    Various methods to access player stats, team information, and season data
    with built-in analysis capabilities for tracking player movements and performance.
"""
        print("Initializing Basketball API...")
        # Basic parameters
        self.players_per_season = players_per_season
        self.elite_percentage = 0.05
        self.movement_probability = 0.15
        self.faker = Faker()
        Faker.seed(12345)  
        
        
        self.seasons = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
        self.teams = ['SkyBirds', 'VBADragons', 'SparkiFY', 'KeyBoardWarriors', 'UnitedWeCode', 'HtmlERS', 'LutammS', 'BIsupremE']
        # Players and their attributes
        self._initialize_players()
        self._assign_teams_by_season()
        print("Initialization complete!")

    def _initialize_players(self):
        """Generate exactly 390 players and assign their attributes"""
        print("Generating players...")
        self.players = []
        unique_names = set()
        
        while len(unique_names) < self.players_per_season:
            name = f"{self.faker.first_name()} {self.faker.last_name()}"
            if name not in unique_names:
                unique_names.add(name)
                self.players.append(name)
        
        # Generate number of elite players
        num_elite = int(self.players_per_season * self.elite_percentage)
        
        print(f"Assigning player archetypes ({num_elite} players per elite category)...")
        # Randomly selecting elite players
        all_players = set(self.players)
        self.elite_scorers = set(random.sample(list(all_players), num_elite))
        remaining_players = all_players - self.elite_scorers
        
        self.elite_defenders = set(random.sample(list(remaining_players), num_elite))
        remaining_players = remaining_players - self.elite_defenders
        
        self.elite_playmakers = set(random.sample(list(remaining_players), num_elite))
        self.regular_players = remaining_players - self.elite_playmakers
        
        # Player dictionary with attributes
        self.player_data = {}
        for idx, player in enumerate(self.players):
            archetype = 'regular'
            if player in self.elite_scorers:
                archetype = 'elite_scorer'
            elif player in self.elite_defenders:
                archetype = 'elite_defender'
            elif player in self.elite_playmakers:
                archetype = 'elite_playmaker'
            
            self.player_data[player] = {
                'player_id': idx,
                'archetype': archetype,
                'movement_prone': random.random() < self.movement_probability
            }

    def _assign_teams_by_season(self):
        """Assign teams to players for each season"""
        print("Assigning teams for each season...")
        self.player_teams = {player: {} for player in self.players}
        
        # Team assignment
        for player in self.players:
            self.player_teams[player][self.seasons[0]] = random.choice(self.teams)
        
        # Subsequent seasons assignments
        for i in range(1, len(self.seasons)):
            current_season = self.seasons[i]
            previous_season = self.seasons[i-1]
            
            for player in self.players:
                previous_team = self.player_teams[player][previous_season]
                
                if self.player_data[player]['movement_prone'] and random.random() < 0.3:
                    
                    new_team = random.choice([t for t in self.teams if t != previous_team])
                    self.player_teams[player][current_season] = new_team
                else:
                    # player team consistency
                    self.player_teams[player][current_season] = previous_team

    def _generate_stats(self, player: str, season: str) -> Dict[str, Any]:
        """Generate statistics based on player archetype, with random missing data"""
        archetype = self.player_data[player]['archetype']
        base_stats = {
            'games_played': random.randint(20, 82),
            'minutes': round(random.uniform(12, 38), 1)
        }
        
        stats = {}
        
        if archetype == 'elite_scorer':
            stats = {
                'points': round(random.uniform(20, 35), 1),
                'rebounds': round(random.uniform(1, 8), 1),
                'assists': round(random.uniform(2, 6), 1),
                'steals': round(random.uniform(0, 2), 1),
                'blocks': round(random.uniform(0, 1), 1),
                'fg_pct': round(random.uniform(0.450, 0.650), 3),
                'fg3_pct': round(random.uniform(0.350, 0.450), 3),
                'ft_pct': round(random.uniform(0.800, 0.950), 3)
            }
        elif archetype == 'elite_defender':
            stats = {
                'points': round(random.uniform(8, 15), 1),
                'rebounds': round(random.uniform(5, 12), 1),
                'assists': round(random.uniform(1, 4), 1),
                'steals': round(random.uniform(1.5, 3), 1),
                'blocks': round(random.uniform(1, 3), 1),
                'fg_pct': round(random.uniform(0.350, 0.550), 3),
                'fg3_pct': round(random.uniform(0.250, 0.380), 3),
                'ft_pct': round(random.uniform(0.650, 0.850), 3)
            }
        elif archetype == 'elite_playmaker':
            stats = {
                'points': round(random.uniform(15, 25), 1),
                'rebounds': round(random.uniform(3, 7), 1),
                'assists': round(random.uniform(8, 12), 1),
                'steals': round(random.uniform(1, 2.5), 1),
                'blocks': round(random.uniform(0, 1), 1),
                'fg_pct': round(random.uniform(0.400, 0.550), 3),
                'fg3_pct': round(random.uniform(0.350, 0.420), 3),
                'ft_pct': round(random.uniform(0.800, 0.900), 3)
            }
        else:  # regular players
            tier = random.choices(['rotation', 'bench', 'deep_bench'], 
                                weights=[0.5, 0.3, 0.2])[0]
            
            if tier == 'rotation':
                stats = {
                    'points': round(random.uniform(8, 18), 1),
                    'rebounds': round(random.uniform(3, 8), 1),
                    'assists': round(random.uniform(2, 5), 1),
                    'steals': round(random.uniform(0.5, 1.8), 1),
                    'blocks': round(random.uniform(0.3, 1.2), 1),
                    'fg_pct': round(random.uniform(0.400, 0.520), 3),
                    'fg3_pct': round(random.uniform(0.300, 0.380), 3),
                    'ft_pct': round(random.uniform(0.700, 0.850), 3)
                }
            elif tier == 'bench':
                stats = {
                    'points': round(random.uniform(5, 12), 1),
                    'rebounds': round(random.uniform(2, 5), 1),
                    'assists': round(random.uniform(1, 3), 1),
                    'steals': round(random.uniform(0.3, 1.2), 1),
                    'blocks': round(random.uniform(0.2, 0.8), 1),
                    'fg_pct': round(random.uniform(0.380, 0.480), 3),
                    'fg3_pct': round(random.uniform(0.280, 0.360), 3),
                    'ft_pct': round(random.uniform(0.650, 0.800), 3)
                }
            else:  # deep_bench
                stats = {
                    'points': round(random.uniform(2, 8), 1),
                    'rebounds': round(random.uniform(1, 4), 1),
                    'assists': round(random.uniform(0.5, 2), 1),
                    'steals': round(random.uniform(0.2, 0.8), 1),
                    'blocks': round(random.uniform(0.1, 0.5), 1),
                    'fg_pct': round(random.uniform(0.350, 0.450), 3),
                    'fg3_pct': round(random.uniform(0.250, 0.330), 3),
                    'ft_pct': round(random.uniform(0.600, 0.750), 3)
                }

        # Randomly introducing missing values
        missing_percentage = 0.05  
        stats_with_missing = {key: (value if random.random() > missing_percentage else None) 
                             for key, value in stats.items()}
        
        return {**base_stats, **stats_with_missing}

    def get_players(self) -> List[Dict[str, Any]]:
        """Method to get player data"""
        print("\nLoading player data...")
        time.sleep(5)  
        
        players = [
            {
                'player_id': data['player_id'],
                'player_name': player,
                'archetype': data['archetype']
            }
            for player, data in self.player_data.items()
        ]
        
        print(f"Player data loaded successfully! ({len(players)} players)")
        return players

    def get_stats(self) -> List[Dict[str, Any]]:
        """Method to get player statistics"""
        stats_data = []
        
        print("\nBeginning statistics data load...")
        for season in self.seasons:
            print(f"\nLoading data for season {season}...")
            time.sleep(15)  
            
            season_stats = []
            for player in self.players:
                stats = self._generate_stats(player, season)
                season_stats.append({
                    'player_id': self.player_data[player]['player_id'],
                    'season': season,
                    'team': self.player_teams[player][season],
                    **stats
                })
            
            stats_data.extend(season_stats)
            print(f"Loaded {len(season_stats)} player records for season {season}")
        
        print(f"\nAll statistics loaded successfully! ({len(stats_data)} total records)")
        return stats_data


if __name__ == "__main__":
    # Simulating API with timing information
    start_time = time.time()
    
    print("Starting data generation process...")
    api = BasketballAPI()
    
    print("\nFetching player data...")
    players = api.get_players()
    
    print("\nFetching statistics...")
    stats = api.get_stats()
    
    end_time = time.time()
    total_time = round(end_time - start_time)
    minutes = total_time // 60
    seconds = total_time % 60
    
    print(f"\nProcess completed in {minutes} minutes and {seconds} seconds")