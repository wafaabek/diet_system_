from flask import Flask, request, render_template, jsonify, session
import joblib
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

more_details_button = False

# Load the pre-trained KNN model for supervised
knn = joblib.load('knn_supervised_model_.pkl')
# Load the vectorizer
vectorizer = joblib.load('supervise_vectorizer_.pkl')

# Load the pre-trained KNN model  for unsupervised
knn_similar =joblib.load('knn_unsupervised_model_.pkl')

preprocessor_similar =joblib.load('unsupervised_preprossesor_.pkl')

# Load the dataset
data = pd.read_csv('cleaned_recipes_.csv')

recipes_limited = data.head(6)  # Show only the first 6 recipes

# Nutritional columns to be used for recommendations
nutritional_features = [
    "Calories", "ProteinContent", "FatContent", "CarbohydrateContent"
]

# Scale nutritional features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[nutritional_features])

def fetch_image(recipe_name , ingredients):
    url = "Your-URL"
    API_KEY = "YOUR-API-KEY"
    SEARCH_ENGINE_ID = "ID-SEARCH-ENGINE"

    search_query = f"{recipe_name} {ingredients}"

    params = {
        "q": search_query,
        "cx": SEARCH_ENGINE_ID,
        "key": API_KEY,
        "searchType": "image",
        "num": 1
    }

    try:
        # Effectuer la requête
        response = requests.get(url, params=params)

        # Vérifier le code de statut HTTP
        if response.status_code != 200:
            print(f"Erreur HTTP: {response.status_code}")
            print(f"Message: {response.text}")
            return None

        # Vérifier si la réponse contient du JSON
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Erreur : La réponse ne contient pas de JSON valide.")
            print(f"Contenu brut de la réponse : {response.text}")
            return None

        # Vérifier si des résultats existent dans le JSON
        if "items" in data and len(data["items"]) > 0:
            image_url = data["items"][0]["link"]  # URL de la première image
            return image_url
        else:
            print("Aucune image trouvée pour cette recette.")
            return None

    except requests.RequestException as e:
        print(f"Erreur de requête : {e}")
        return None

@app.route('/')
def home():
    return render_template("index.html", recipes=recipes_limited.to_dict("records"))






@app.route("/recommend/<int:recipe_id>")
def recommend(recipe_id):
    """
    Recommend similar recipes based on the selected recipe ID.
    """
    
    # Find the index of the selected recipe in the dataset
    selected_index = data.index[data["RecipeId"] == recipe_id].tolist()[0]
    similar_recipes = find_similar_recipes(recipe_id, data, knn_similar, preprocessor_similar, n_neighbors=10)

    return render_template("recommendations.html", selected_recipe=data.iloc[selected_index], recommendations=similar_recipes)

################################################################################################

@app.route('/predict', methods=['POST'])
def predict():
    ingredients = request.form['ingredients']
    user_vector = vectorizer.transform([ingredients])
    prediction = knn.predict(user_vector)[0]

    # Get the most similar recipe indices
    _, indices = knn.kneighbors(user_vector)
    
    top_5_indices = [int(index) for index in indices[0][:6]]  # Convert to Python int
    session['remaining_recipes'] = top_5_indices  # Save indices in session
    session['prediction'] = prediction  # Save prediction in session for reuse

    return get_recipe(prediction)






@app.route('/next_recipe', methods=['POST'])
def next_recipe():
    prediction = session.get('prediction', 'Unknown')  # Retrieve stored prediction
    return get_recipe(prediction)


def get_recipe(prediction):
    if 'remaining_recipes' not in session:
        return render_template(
            'index.html',
            prediction_text=f'No recipes available.'
        )

    # Get the first recipe index from the session
    recipe_index = session['remaining_recipes'].pop(0)  # Remove the first element
    session['remaining_recipes'].append(recipe_index)  # Add it to the end
    session.modified = True  # Notify Flask that the session has changed

    most_similar_recipe = data.iloc[recipe_index]
    recipe_name = most_similar_recipe['Name']
    
    # Make sure to join ingredient parts as a single string
    image_url = fetch_image(recipe_name, ' '.join(most_similar_recipe['RecipeIngredientParts']))

    # Recipe details to display
    recipe_details = {
        'Name': recipe_name,
        'CookTime': most_similar_recipe['CookTime'],
        'Images': image_url,
        'RecipeCategory': most_similar_recipe['RecipeCategory'],
        'RecipeIngredientQuantities': most_similar_recipe['RecipeIngredientQuantities'],
        'RecipeIngredientParts': most_similar_recipe['RecipeIngredientParts'],
        'AggregatedRating': f"{most_similar_recipe['AggregatedRating']:.2f}",
        'RecipeInstructions': most_similar_recipe['RecipeInstructions'],
        'RecipeServings': int(most_similar_recipe['RecipeServings']),
    }

    extended_details = {
        'Calories': f"{most_similar_recipe['Calories']:.2f}",
        'FatContent': f"{most_similar_recipe['FatContent']:.2f}",
        'SaturatedFatContent': f"{most_similar_recipe['SaturatedFatContent']:.2f}",
        'CholesterolContent': f"{most_similar_recipe['CholesterolContent']:.2f}",
        'SodiumContent': f"{most_similar_recipe['SodiumContent']:.2f}",
        'CarbohydrateContent': f"{most_similar_recipe['CarbohydrateContent']:.2f}",
        'FiberContent': f"{most_similar_recipe['FiberContent']:.2f}",
        'SugarContent': f"{most_similar_recipe['SugarContent']:.2f}",
        'ProteinContent': f"{most_similar_recipe['ProteinContent']:.2f}"
    }

    return render_template(
        'index.html',
        prediction_text=f'The recipe is: {prediction}',
        recipe_details=recipe_details,
        extended_details=extended_details
    )

def find_similar_recipes(recipe_id, data, knn_model, preprocessor, n_neighbors=10):
    """
    Find similar recipes to a given recipe.

    Args:
        recipe_id (int): Index of the recipe to find similar recipes for.
        data (DataFrame): Original dataset.
        knn_model (NearestNeighbors): Trained KNN model.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        n_neighbors (int): Number of similar recipes to return.

    Returns:
        DataFrame: Similar recipes with their distances.
    """
    # Preprocess the recipe features
    recipe_features = preprocessor.transform(data.iloc[[recipe_id]])
    
    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(recipe_features, n_neighbors=n_neighbors)
    
    # Retrieve similar recipes
    similar_recipes = data.iloc[indices[0]].copy()
    similar_recipes['Distance'] = distances[0]
    
    return similar_recipes

if __name__ == "__main__":
    app.run(debug=True)
