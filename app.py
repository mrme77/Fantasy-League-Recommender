from flask import Flask, render_template, request, flash, url_for, send_from_directory
import pickle
import pandas as pd
import os
from werkzeug.serving import make_server
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.urandom(24)

def load_model():
    """Load the pickled model with error handling"""
    try:
        with open('fantasy_rec_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        logger.error("Model file 'fantasy_rec_model.pkl' not found")
        raise Exception("Model file 'fantasy_rec_model.pkl' not found")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")

# Load the model when the application starts
try:
    fantasy_rec = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error initializing application: {str(e)}")
    fantasy_rec = None

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# def shutdown_server():
#     """Shutdown the Flask server"""
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()

# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     """Handle shutdown requests"""
#     try:
#         shutdown_server()
#         logger.info("Server shutting down...")
#         return 'Server shutting down...'
#     except Exception as e:
#         logger.error(f"Error during shutdown: {str(e)}")
#         return str(e), 500
import os,signal
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Handle shutdown requests"""
    logger.info("Server shutting down...")
    os.kill(os.getpid(), signal.SIGINT)  # Send SIGINT to the current process
    return 'Server shutting down...'

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle main page requests"""
    if fantasy_rec is None:
        flash("Error: Model not loaded properly", "error")
        return render_template('error.html', error="Model not loaded properly")

    # Initialize variables
    initial_recommendations = None
    updated_recommendations = None
    precision_initial = None
    precision_updated = None
    unavailable_players = []
    unavailable_message = None

    try:
        # Get unique player IDs from the dataset
        player_ids = fantasy_rec.get_player_ids()
        logger.debug(f"Retrieved {len(player_ids)} player IDs")

        if request.method == 'POST':
            try:
                # Get and validate the selected player_id
                player_id = int(request.form.get('player_id'))
                logger.info(f"Processing request for player ID: {player_id}")
                
                # Get initial recommendations
                initial_recommendations = fantasy_rec.get_recommendations(player_id, top_n=5)
                precision_initial = fantasy_rec.precisionK(player_id, initial_recommendations)
                logger.debug(f"Initial recommendations generated with precision: {precision_initial}")

                # Process unavailable players
                unavailable_players = request.form.getlist('unavailable_players')
                if unavailable_players:
                    unavailable_players = [int(pid) for pid in unavailable_players]
                    fantasy_rec.update_unavailable_players(unavailable_players)
                    logger.info(f"Updated unavailable players: {unavailable_players}")
                    
                    # Get updated recommendations
                    updated_recommendations = fantasy_rec.get_recommendations(player_id, top_n=5)
                    precision_updated = fantasy_rec.precisionK(player_id, updated_recommendations)
                    unavailable_message = f"Unavailable players: {', '.join(map(str, unavailable_players))}"
                    logger.debug(f"Updated recommendations generated with precision: {precision_updated}")

            except ValueError as ve:
                logger.error(f"Invalid player ID: {str(ve)}")
                flash("Invalid player ID provided", "error")
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                flash(f"Error processing request: {str(e)}", "error")

        return render_template(
            'index.html',
            player_ids=player_ids,
            initial_recommendations=initial_recommendations,
            precision_initial=precision_initial,
            updated_recommendations=updated_recommendations,
            precision_updated=precision_updated,
            unavailable_players=unavailable_players,
            unavailable_message=unavailable_message
        )

    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error=str(e))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    logger.warning(f"Page not found: {request.url}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)