from flask import Flask, request, render_template, jsonify, session
import joblib
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import numpy as np


app = Flask(__name__)
app.secret_key = 'malek key'  # Required for session management

more_details_button = False

# Load the pre-trained KNN model for supervised
#knn = joblib.load('knn_supervised_model_.pkl')
# Load the vectorizer
#vectorizer = joblib.load('supervise_vectorizer_.pkl')

# Load the pre-trained KNN model  for unsupervised
#knn_similar =joblib.load('knn_unsupervised_model_.pkl')
  
#preprocessor_similar =joblib.load('unsupervised_preprossesor_.pkl')

# Charger les modèles dans Flask


# Function to load KMeans model and KNN model
def load_models():
   
    kmeans = joblib.load(r'model\kmeans_model.joblib')
    knn_diet = joblib.load(r'model\knn_diet.joblib')
    
    return kmeans, knn_diet




# Load the dataset
data = pd.read_csv('cleaned_recipes_.csv')

recipes_limited = data.sample(n=6)

def fetch_image(recipe_name , ingredients):
    url = "https://cse.google.com/cse.js?cx=b604b290643ae4f55"
    API_KEY = "AIzaSyCcPP5R23o7DdTVxCdLvBIwXKqm_ullXG4"
    SEARCH_ENGINE_ID = "b604b290643ae4f55"

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
    selected_indexes = data.index[data["RecipeId"] == recipe_id].tolist()

    if not selected_indexes:
        return "Recipe not found", 404  # Handle the case when no recipe matches the given ID

    selected_index = selected_indexes[0]

    # Find the row for the selected recipe
    recipe_row = data.iloc[selected_index]
    recipe_row['Images'] = fetch_image(recipe_row['Name'],recipe_row['RecipeIngredientParts'])
    # Find similar recipes using the k-NN model
    distances, indices = knn_similar.kneighbors(preprocessor_similar.transform(data.iloc[[selected_index]]), n_neighbors=10)

    # Get the recommended recipes
    similar_recipes = data.iloc[indices[0]].to_dict("records")
    # Fetch images for each of the recommended recipes
    for recipe in similar_recipes:
        # Fetch the image URL for each recommended recipe
        recipe_name = recipe['Name']
        ingredients = recipe['RecipeIngredientParts']  # Join ingredients if it's a list
        recipe['Images'] = fetch_image(recipe_name, ingredients)  # Add image URL to the recipe dict
    return render_template("recommendations.html", selected_recipe=recipe_row, recommendations=similar_recipes)


################################################################################################
# Fonction pour gérer les valeurs aberrantes
# Classifier les recettes par type de repas (Petit-déjeuner, Déjeuner, Dîner)


# Charger les données de recettes

# Lecture du fichier CSV contenant les recettes
recipes = pd.read_csv('cleaned_recipes_.csv')

# Fonction pour détecter et gérer les outliers


# Gestion des outliers
# Gérer les colonnes numériques
def validate_numeric_columns(recipes, features):
    for feature in features:
        recipes[feature] = pd.to_numeric(recipes[feature], errors='coerce')
    recipes = recipes.dropna(subset=features)
    return recipes

# Gérer les outliers
recipes = pd.read_csv('cleaned_recipes_.csv')

# **Validation et gestion des données**
def validate_numeric_columns(recipes, features):
    for feature in features:
        recipes[feature] = pd.to_numeric(recipes[feature], errors='coerce')
    recipes = recipes.dropna(subset=features)
    return recipes

def handle_outliers(recipes, features):
    z_scores = (recipes[features] - recipes[features].mean()) / recipes[features].std()
    outliers = (z_scores.abs() > 3).any(axis=1)
    return recipes[~outliers]

# **Classification des recettes par type de repas**
def classify_recipes(recipes):
    breakfast_keywords = [
        'Breakfast', 'Scones', 'Smoothies', 'Oatmeal', 'Breads', 'Frozen Desserts',
        'Breakfast Eggs', 'Pancakes', 'Croissants', 'Cereals', 'Yogurt', 'Coffee',
        'Juices', 'Tarts', 'Muffins', 'Fruit', 'Toast', 'Bagels'
    ]
    lunch_dinner_keywords = [
        'Lunch/Snacks', 'Dinner', 'Sandwich', 'Salad', 'Soup', 'Stew', 'Chicken',
        'Wrap', 'Pasta', 'Roast', 'Fish', 'Rice', 'Vegetable', 'Beans', 'Meat'
    ]

    def classify_category(category):
        if isinstance(category, str):
            if any(keyword in category for keyword in breakfast_keywords):
                return 'Breakfast'
            elif any(keyword in category for keyword in lunch_dinner_keywords):
                return 'Lunch_Dinner'
        return 'Other'

    recipes['MealType'] = recipes['RecipeCategory'].apply(classify_category)
    return recipes

# **Clustering des recettes**
def cluster_recipes(recipes, n_clusters=3):
    features = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(recipes[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    recipes['Cluster'] = kmeans.fit_predict(scaled_data)
    return recipes, kmeans, scaler

# **Filtrer les recettes par calories**
def filter_by_calories(recipes, target_calories, tolerance=150):
    return recipes[(recipes['Calories'] >= target_calories - tolerance) & 
                   (recipes['Calories'] <= target_calories + tolerance)]

# **Recommander des recettes**
def recommend_recipes(query, kmeans, scaler, recipes, n_neighbors=5):
    features = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']
    query_scaled = scaler.transform(query)
    cluster = kmeans.predict(query_scaled)[0]
    cluster_recipes = recipes[recipes['Cluster'] == cluster]
    cluster_recipes = filter_by_calories(cluster_recipes, query[0][0], tolerance=50)

    if cluster_recipes.empty:
        return pd.DataFrame({'Message': ['Aucune recette disponible pour ce cluster.']})
    
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(cluster_recipes[features])
    distances, indices = knn.kneighbors(query_scaled)

    recommendations = cluster_recipes.iloc[indices[0]].reset_index(drop=True)
    recommendations['CalorieDifference'] = (recommendations['Calories'] - query[0][0]).abs()
    return recommendations.sort_values(by='CalorieDifference')[['Name', 'Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']].reset_index(drop=True)

# **Calculer les besoins caloriques**
def calculate_daily_calories(weight, height, age, sex, goal_weight):
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == 'male' else -161)
    daily_calories = bmr + (goal_weight - weight) * 500 / abs(goal_weight - weight if goal_weight != weight else 1)
    return daily_calories

# **Calculer les macronutriments**
def calculate_macronutrients(daily_calories):
    protein = (daily_calories * 0.15) / 4
    fat = (daily_calories * 0.25) / 9
    carbs = (daily_calories * 0.60) / 4
    return protein, fat, carbs

# **Diviser les calories par repas**
def distribute_calories(daily_calories):
    return daily_calories * 0.25, daily_calories * 0.40, daily_calories * 0.35

# **Route principale**
@app.route('/diet', methods=['GET', 'POST'])
def indexo():
    # Validation et prétraitement
    features = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']
    global recipes
    recipes = validate_numeric_columns(recipes, features)
    recipes = handle_outliers(recipes, features)
    recipes = classify_recipes(recipes)
    recipes, kmeans, scaler = cluster_recipes(recipes)

    if request.method == 'POST':
        # Récupérer les informations utilisateur
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        sex = request.form['sex']
        goal_weight = float(request.form['goal_weight'])

        # Calcul des besoins caloriques et macronutriments
        # Calculer les besoins caloriques et macronutriments
        daily_calories = calculate_daily_calories(weight, height, age, sex, goal_weight)
        protein_grams, fat_grams, carb_grams = calculate_macronutrients(daily_calories)

# Répartition des besoins par repas
        breakfast_calories, lunch_calories, dinner_calories = distribute_calories(daily_calories)
        breakfast_macros = calculate_macronutrients(breakfast_calories)
        lunch_macros = calculate_macronutrients(lunch_calories)
        dinner_macros = calculate_macronutrients(dinner_calories)

# Recommandations pour chaque repas
        breakfast_recipes = recommend_recipes(
        np.array([[breakfast_calories, breakfast_macros[1], breakfast_macros[0], breakfast_macros[2]]]), 
        kmeans, scaler, recipes[recipes['MealType'] == 'Breakfast']
)
        lunch_recipes = recommend_recipes(
        np.array([[lunch_calories, lunch_macros[1], lunch_macros[0], lunch_macros[2]]]), 
        kmeans, scaler, recipes[recipes['MealType'] == 'Lunch_Dinner']
)
        dinner_recipes = recommend_recipes(
        np.array([[dinner_calories, dinner_macros[1], dinner_macros[0], dinner_macros[2]]]), 
        kmeans, scaler, recipes[recipes['MealType'] == 'Lunch_Dinner']
)

        # Rendu de la page
        return render_template(
    'diet.html',
    daily_calories=round(daily_calories, 2),
    protein_grams=round(protein_grams, 2),
    fat_grams=round(fat_grams, 2),
    carb_grams=round(carb_grams, 2),
    formatted_macronutrients=f"""
        Besoins journaliers en macronutriments :
        - Protéines : {round(protein_grams, 2)} g
        - Graisses : {round(fat_grams, 2)} g
        - Glucides : {round(carb_grams, 2)} g
    """,
    breakfast_macros=breakfast_macros,
    lunch_macros=lunch_macros,
    dinner_macros=dinner_macros,
    breakfast_calories=breakfast_calories,
    lunch_calories=lunch_calories,
    dinner_calories=dinner_calories,
    breakfast=breakfast_recipes.to_dict(orient='records'),
    lunch=lunch_recipes.to_dict(orient='records'),
    dinner=dinner_recipes.to_dict(orient='records')
)

    return render_template('diet.html')








@app.route('/next_recipe', methods=['POST'])
def next_recipe():
    prediction = session.get('prediction', 'Unknown')  # Retrieve stored prediction
    return get_recipe(prediction)


##################################




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
        extended_details=extended_details,
        recipes=recipes_limited.to_dict("records")
    )

def find_similar_recipes(recipe_id, data, knn_model, preprocessor, n_neighbors=10):
    """
    Find similar recipes to a given recipe.
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
