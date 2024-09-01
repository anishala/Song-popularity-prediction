from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import gzip

app = Flask(__name__)

# Load the pre-trained model, scaler, and encoders
def load_model_files():
    try:
        with open('scaler.pkl', 'rb') as scaler_file, \
             gzip.open('random_forest_model_compressed.pkl.gz', 'rb') as model_file:
            return {
                'scaler': pickle.load(scaler_file),
                'model': pickle.load(model_file)
            }
    except FileNotFoundError as e:
        print(f"Error: Required model file not found. {str(e)}")
        return None
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        return None

model_components = load_model_files()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('indexs.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and return the prediction result."""
    if not model_components:
        return jsonify({"error": "Model components not loaded properly"}), 500

    try:
        # Extract and convert form data to appropriate types
        form_data = {
            'playlist_genre': request.form.get('playlist_genre', ''),
            'playlist_subgenre': request.form.get('playlist_subgenre', ''),
            'danceability': float(request.form.get('danceability', 0)),
            'energy': float(request.form.get('energy', 0)),
            'key': float(request.form.get('key', 0)),
            'loudness': float(request.form.get('loudness', 0)),
            'mode': int(request.form.get('mode', 0)),
            'speechiness': float(request.form.get('speechiness', 0)),
            'acousticness': float(request.form.get('acousticness', 0)),
            'valence': float(request.form.get('valence', 0)),
            'tempo': float(request.form.get('tempo', 0)),
            'duration_s': float(request.form.get('duration_s', 0))
        }

        # Convert form data into a DataFrame
        input_data = pd.DataFrame([form_data])
    
     
      

        input_data_scaled = model_components['scaler'].transform(input_data)

        # Make the prediction
        prediction = model_components['model'].predict(input_data_scaled)[0]

        # Decode the prediction
        result = "Popular" if prediction == 1 else "Not Popular"

        # Render the result page with the prediction
        return render_template('results.html', prediction=result)

    except ValueError as e:
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
