import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.express as px

class EnhancedRecipeRecommender:
    def __init__(self, df):
        self.df = df.copy()  # Create a copy of original dataframe
        self.original_df = df.copy()  # Keep an unscaled version
        self.scaler = MinMaxScaler()
        
        # Normalize numerical features for similarity calculation
        numerical_features = ['rating', 'total_time', 'calories', 'protein_g']
        self.df[numerical_features] = self.df[numerical_features].fillna(0)
        self.scaled_features = self.scaler.fit_transform(self.df[numerical_features])
        self.df[numerical_features] = self.scaled_features
        
        # Prepare ingredient vectors
        self.df['ingredients_str'] = self.df['parsed_ingredients'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ' '.join(eval(x))
        )
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['ingredients_str'])
        
        # Create category embeddings
        self.df['category_encoded'] = pd.get_dummies(self.df['category']).values.tolist()

    def search_recipes(self, query=None, ingredients_list=None):
        if query:
            mask = (
                self.original_df['title'].str.contains(query, case=False) |
                self.original_df['ingredients_str'].str.contains(query, case=False)
            )
            return self.original_df[mask]
        
        if ingredients_list:
            mask = pd.Series(True, index=self.original_df.index)
            for ingredient in ingredients_list:
                mask &= self.original_df['ingredients_str'].str.contains(ingredient, case=False)
            return self.original_df[mask]
        
        return self.original_df
        
    def get_recommendations(self, recipe_idx, n_recommendations=5, 
                          weight_ingredients=0.6,
                          weight_rating=0.2,
                          weight_nutrition=0.1,
                          weight_category=0.1,
                          max_cooking_time=None,
                          dietary_restrictions=None):
        
        # Get ingredient similarity
        ingredient_sim = cosine_similarity(
            self.tfidf_matrix[recipe_idx:recipe_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get rating similarity
        rating_sim = 1 - np.abs(
            self.df['rating'].values - self.df['rating'].iloc[recipe_idx]
        )
        
        # Get nutritional similarity
        nutrition_features = ['calories', 'protein_g']
        nutrition_sim = 1 - np.mean([
            np.abs(self.df[feat].values - self.df[feat].iloc[recipe_idx])
            for feat in nutrition_features
        ], axis=0)
        
        # Get category similarity
        category_sim = cosine_similarity(
            [self.df['category_encoded'].iloc[recipe_idx]],
            self.df['category_encoded'].tolist()
        ).flatten()
        
        # Combine similarities with weights
        final_sim = (
            weight_ingredients * ingredient_sim +
            weight_rating * rating_sim +
            weight_nutrition * nutrition_sim +
            weight_category * category_sim
        )
        
        # Apply filters
        mask = np.ones(len(self.df), dtype=bool)
        if max_cooking_time:
            mask &= self.original_df['total_time'] <= max_cooking_time
        if dietary_restrictions:
            for restriction in dietary_restrictions:
                mask &= ~self.original_df['ingredients_str'].str.contains(restriction, case=False)
        
        final_sim[~mask] = -1
        
        # Get indices of top similar recipes
        similar_indices = np.argsort(final_sim)[::-1][1:n_recommendations+1]
        
        # Return recommendations with image column included
        recommendations = self.original_df.iloc[similar_indices][
            ['title', 'category', 'rating', 'total_time', 'calories', 'protein_g', 'image',
             'parsed_ingredients', 'clean_instructions']  # Include all necessary columns
        ].copy()
        recommendations['similarity_score'] = final_sim[similar_indices]
        
        return recommendations

def display_recipe_card(recipe_data, show_full_details=False):
    """Display a recipe in a card format with improved image handling"""
    try:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Improved image handling logic
            image_url = recipe_data.get('image', None)
            if (image_url and pd.notna(image_url) and 
                isinstance(image_url, str) and 
                image_url.strip() != '' and 
                image_url.lower() != 'not available'):
                try:
                    st.image(image_url, use_column_width=True)
                except Exception as e:
                    st.image("default_recipe_image.jpg", use_column_width=True)
                    if show_full_details:
                        st.warning("Unable to load recipe image")
            else:
                st.image("default_recipe_image.jpg", use_column_width=True)
        
        with col2:
            st.subheader(f"{recipe_data['title']}")
            st.write(f"**Category:** {recipe_data['category']}")
            
            # Use simple text instead of nested columns
            st.write(f"⭐ Rating: {recipe_data['rating']:.1f}/5")
            st.write(f"⏲️ Time: {int(recipe_data['total_time'])} mins")
            st.write(f"🔥 Calories: {int(recipe_data['calories'])}")

        if show_full_details:
            with st.expander("📝 Ingredients", expanded=True):
                ingredients = eval(recipe_data['parsed_ingredients']) if isinstance(recipe_data['parsed_ingredients'], str) else recipe_data['parsed_ingredients']
                for ingredient in ingredients:
                    st.write(f"• {ingredient}")
            
            with st.expander("👩‍🍳 Instructions", expanded=True):
                if 'clean_instructions' in recipe_data and recipe_data['clean_instructions']:
                    instructions = recipe_data['clean_instructions'].split('.')
                    for i, instruction in enumerate(instructions, 1):
                        if instruction.strip():
                            st.write(f"{i}. {instruction.strip()}")
            
            with st.expander("🍎 Nutritional Information", expanded=False):
                nutrition_cols = [
                    'calories', 'carbohydrates_g', 'sugars_g', 'fat_g', 
                    'cholesterol_mg', 'protein_g', 'dietary_fiber_g', 
                    'sodium_mg', 'calcium_mg', 'iron_mg'
                ]
                
                col1, col2 = st.columns(2)
                for i, nutrient in enumerate(nutrition_cols):
                    if nutrient in recipe_data:
                        nutrient_name = nutrient.replace('_', ' ').title()
                        value = recipe_data[nutrient]
                        if pd.notna(value):
                            if 'mg' in nutrient:
                                formatted_value = f"{value:.1f} mg"
                            elif 'g' in nutrient:
                                formatted_value = f"{value:.1f} g"
                            else:
                                formatted_value = f"{value:.1f}"
                            
                            with col1 if i % 2 == 0 else col2:
                                st.metric(nutrient_name, formatted_value)
    except Exception as e:
        st.error(f"Error displaying recipe card: {str(e)}")

@st.cache_data
def load_data():
    return pd.read_csv('transformed_df.csv')

def create_streamlit_app():
    st.title("🍳 Recipe Recommendation System")
    
    # Load data and initialize recommender
    df = load_data()
    recommender = EnhancedRecipeRecommender(df)
    
    # Initialize session state
    if 'selected_recipe' not in st.session_state:
        st.session_state.selected_recipe = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'ingredients_input' not in st.session_state:
        st.session_state.ingredients_input = ''
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "🔍 Search by Name"
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        col4, col5 = st.columns([1, 1])
        with col4:
            if st.button("🔍 Search by Name"):
                st.session_state.selected_tab = "🔍 Search by Name"
        with col5:
            if st.button("🥗 Search by Ingredients"):
                st.session_state.selected_tab = "🥗 Search by Ingredients"
    
    # Display content based on selected tab
    if st.session_state.selected_tab == "🔍 Search by Name":
        search_query = st.text_input("Enter recipe name:", key="name_search", value=st.session_state.search_query)
        
        if search_query != st.session_state.search_query:
            st.session_state.selected_recipe = None
            st.session_state.search_query = search_query

        if search_query:
            filtered_df = recommender.search_recipes(query=search_query)
            st.write(f"Found {len(filtered_df)} matching recipes")
            if not filtered_df.empty:
                selected_recipe = st.selectbox(
                    "Select a recipe:", 
                    filtered_df['title'].tolist(), 
                    key="name_select"
                )
                st.session_state.selected_recipe = selected_recipe
            else:
                st.warning("No recipes found. Please try a different search term.")
    
    elif st.session_state.selected_tab == "🥗 Search by Ingredients":
        ingredients_input = st.text_input(
            "Enter ingredients (comma-separated):", 
            key="ing_search", 
            value=st.session_state.ingredients_input
        )
        
        if ingredients_input != st.session_state.ingredients_input:
            st.session_state.selected_recipe = None
            st.session_state.ingredients_input = ingredients_input

        if ingredients_input:
            ingredients_list = [i.strip() for i in ingredients_input.split(',')]
            filtered_df = recommender.search_recipes(ingredients_list=ingredients_list)
            st.write(f"Found {len(filtered_df)} matching recipes")
            if not filtered_df.empty:
                selected_recipe = st.selectbox(
                    "Select a recipe:", 
                    filtered_df['title'].tolist(), 
                    key="ing_select"
                )
                st.session_state.selected_recipe = selected_recipe
            else:
                st.warning("No recipes found with these ingredients. Please try different ingredients.")
    
    # Display Selected Recipe
    st.header("Selected Recipe")
    if st.session_state.selected_recipe:
        recipe_idx = df[df['title'] == st.session_state.selected_recipe].index[0]
        selected_recipe = df.iloc[recipe_idx]
        display_recipe_card(selected_recipe, show_full_details=True)
    else:
        st.info("No recipe selected. Please search for or select a recipe.")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Recommendation Settings")
        
        st.subheader("Adjust Importance Weights")
        weight_ingredients = st.slider("🥗 Ingredients Similarity", 0.0, 1.0, 0.6)
        weight_rating = st.slider("⭐ Rating Similarity", 0.0, 1.0, 0.2)
        weight_nutrition = st.slider("🍎 Nutritional Similarity", 0.0, 1.0, 0.1)
        weight_category = st.slider("📑 Category Similarity", 0.0, 1.0, 0.1)
        
        st.subheader("Filters")
        max_cooking_time = st.number_input(
            "⏲️ Maximum Cooking Time (minutes)", 
            min_value=0, 
            value=120
        )
        
        dietary_restrictions = st.multiselect(
            "🚫 Dietary Restrictions",
            ["dairy", "nuts", "gluten", "meat", "seafood"]
        )
    
    # Get and Display Recommendations
    st.header("👨‍🍳 Recommended Recipes")
    if st.session_state.selected_recipe:
        recipe_idx = df[df['title'] == st.session_state.selected_recipe].index[0]
        recommendations = recommender.get_recommendations(
            recipe_idx,
            n_recommendations=6,
            weight_ingredients=weight_ingredients,
            weight_rating=weight_rating,
            weight_nutrition=weight_nutrition,
            weight_category=weight_category,
            max_cooking_time=max_cooking_time,
            dietary_restrictions=dietary_restrictions,
        )
        
        # Display Recommendations
        for i in range(0, len(recommendations), 2):
            col1, col2 = st.columns(2)
            with col1:
                if i < len(recommendations):
                    recipe = recommendations.iloc[i]
                    display_recipe_card(recipe)
            with col2:
                if i + 1 < len(recommendations):
                    recipe = recommendations.iloc[i + 1]
                    display_recipe_card(recipe)
    else:
        st.info("Please select a recipe to see recommendations.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Recipe Recommender",
        page_icon="🍳",
        layout="wide"
    )
    create_streamlit_app()