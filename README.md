# Basketball Fantasy League Recommender Project
![Picture](image.png)
### Overview

This repo attempts to provide a comprehensive educational project that demonstrates the integration of data engineering and machine learning concepts. The project showcases three key components: API development, ETL (Extract, Transform, Load) processes, and recommender systems. Specifically, we leverage a custom-built BasketballAPI that generates fictional basketball player data and statistics across multiple seasons by simulating the `nba_api` which is an API Client for www.nba.com. This data is then processed through a Python-based ETL pipeline that cleanly transforms and stores the information in a CSV format. 

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which provides a structured approach through six key phases:
1. **Business Understanding**: Defining the need for a fantasy basketball recommender system
2. **Data Understanding**: Analyzing the BasketballAPI data structure and statistical patterns
3. **Data Preparation**: ETL processes to clean and transform the raw API data
4. **Modeling**: Implementing various recommender system algorithms
5. **Evaluation**: Testing the validity of the recommendations generated.6. **Deployment**: Creating a usable baseline framework for fantasy basketball managers.

While the project encompasses multiple technical aspects, this notebook primarily focuses on building and evaluating different recommender system techniques to suggest similar players based on their statistical performance and characteristics. This practical approach allows us to explore real-world applications of data science while working with a controlled, yet realistic dataset that simulates five seasons of basketball data across 390 players and multiple teams.

### Prerequisites

1. Git
2. Python 3.7+ (3.11+ preferred)
3. VS Code Editor (or any other IDE)


The following modules are required: 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from FantasySVD import svd_approach
from FantasyRF import FantasyRecommenderRF
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from BasicRec import FantasyReccomenderBasic
from FantasyAdvanced import FantasyRecommenderAdvanced
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 800)
%matplotlib inline
```

### Deployment
<ul> <b>BaskteballAPI.py</b>: A synthetic basketball data generation API that simulates player statistics and team dynamics across five seasons (2019-2024). The system creates realistic player profiles with different archetypes (elite/regular), manages team assignments, and generates performance metrics.</ul>

<ul> <b>basketball_etl.py</b>: An ETL (Extract, Transform, Load) function that processes basketball player and statistics data into a unified DataFrame. The function merges player information with their performance statistics and exports the combined data to a CSV file, while tracking processing time and record counts.</ul>

<ul><b>DataAnalysis.ipynb</b>: It is a Jupyter Notebook where we perform analysis on the data generated using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. We start with an EDA (Exploratory Data Analysis) then run and evaluate different types of recommender systems.</ul>

<ul><b>BasicRec.py</b>A fantasy basketball recommender system class that uses MinMaxScaler to normalize player statistics and calculate weighted composite scores based on key performance metrics. The system processes player data through normalized statistics (points, rebounds, assists, etc.) to generate personalized recommendations with a demonstrated.</ul>

<ul><b>FantasyAdvanced.py</b>An enanched fantasy basketball recommender system class that leverages MinMaxScaler normalization to evaluate and rank players based on weighted performance metrics (points, rebounds, assists, etc.). The system handles player unavailability.</ul>

<ul><b>FantasySVD.py</b>An enanched basketball player recommendation system class using Singular Value Decomposition (SVD) for dimensionality reduction and cosine similarity for player comparisons. It suggests similar players based on normalized performance metrics and archetypes.</ul>

<ul><b>FantasyRF.py</b>An enanched fantasy basketball recommender system class using Random Forest classification to analyze player statistics and generate personalized recommendations matching similar player archetypes and performance patterns.</ul>

Other files:
- image.png: Readme file intro photo created with AI.
- precision_hist.png: Readme file supporting picture.

The model is turned into a pickle file and used in the web application. However, since the file exceeds GitHub's file size limit, it will not be shared to avoid commit issues.
### Resources

- [Recommender Engine with SVD](https://machinelearningmastery.com/using-singular-value-decomposition-to-build-a-recommender-system/)
- [What are Recommender Systems?](https://www.geeksforgeeks.org/what-are-recommender-systems/)
- [Embeddings](https://vickiboykis.com/what_are_embeddings/)
### Outputs
Below are 2 screenshots displaying the web application.


### Acknowledgment
I would like to acknowledge Stackoverflow, You.com for its generative AI models, and ChatGPT as instrumental aids in the development of this project.
