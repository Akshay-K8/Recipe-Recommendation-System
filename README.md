# Recipe Recommendation System 🍳

## Project Overview 📝
The **Recipe Recommendation System** is an AI-powered web application designed to provide personalized recipe recommendations. Users can search for recipes based on their preferences, available ingredients, and nutritional needs. The system leverages machine learning techniques to enhance user experience by delivering relevant, tailored results.

## Features 🌟
- **Search Recipes**:
  - 🔍 By recipe name.
  - 🥗 By ingredient list.
- **Personalized Recommendations**:
  - 📊 Suggests recipes based on similarity in ingredients, ratings, and nutritional profiles.
  - ⏲️ Includes filters for cooking time and dietary restrictions.
- **Recipe Details**:
  - 🖼️ Displays a detailed recipe card with image, ingredients, instructions, and nutritional information.
- **Adjustable Recommendation Weights**:
  - ⚖️ Allows users to adjust the importance of ingredients, ratings, nutrition, and category in recommendations.
- **User-Friendly UI**:
  - 💻 Built using Streamlit for an intuitive and interactive interface.

## Tech Stack 💻
- **Programming Language**: Python 🐍
- **Libraries**:
  - **Data Processing**: Pandas, NumPy 📊
  - **Machine Learning**: Scikit-learn 🤖
  - **Web Framework**: Streamlit 🌐
  - **Visualization**: Plotly 📈

## Recommendation Logic 🧠
### Data Preprocessing 🧹
- **Normalization**: Numerical features like ratings, total cooking time, calories, and protein are scaled using Min-Max Scaling.
- **Ingredient Vectorization**: Ingredients are converted into text vectors using TF-IDF for similarity comparison.
- **Category Encoding**: Recipe categories are one-hot encoded to compute similarity.

### Algorithm ⚙️
- **Ingredient Similarity**: Calculated using cosine similarity on TF-IDF vectors.
- **Rating Similarity**: Based on the absolute difference between recipe ratings.
- **Nutritional Similarity**: Computes the difference in calories and protein content.
- **Category Similarity**: Uses cosine similarity on encoded category vectors.
- **Weighted Scoring**: Final similarity scores combine all factors based on user-defined weights.

## How It Works 🔍
1. **Search Recipes**: Users can input a recipe name or a list of ingredients.
2. **Filter Results**: Apply filters such as maximum cooking time or dietary restrictions.
3. **View Recipe Details**: Explore comprehensive details about selected recipes.
4. **Get Recommendations**: See similar recipes based on the selected recipe and adjust weights for fine-tuning.

## Setup Instructions ⚙️
### Prerequisites 📋
- Python 3.7 or above 🐍
- Required libraries (can be installed via `requirements.txt`):
  ```bash
  pip install pandas numpy scikit-learn streamlit plotly
  ```

### Steps to Run 🏃‍♂️
1. Clone the repository:
   ```bash
   git clone https://github.com/Akshay-K8/Recipe-Recommendation-System.git
   cd Recipe-Recommendation-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Access the app at `http://localhost:8501` in your browser.

## Future Enhancements 🚀
- **Real-Time Feedback**: Incorporate user feedback loops to improve recommendations.
- **Ingredient Substitution**: Suggest alternative ingredients for unavailable items.
- **Advanced Models**:
  - Reinforcement learning for hyper-personalization.
  - Knowledge graphs to better understand ingredient relationships.
- **Chatbot Integration**: Add an NLP-based interface for conversational interactions.

## Acknowledgments 🙌
This project is inspired by the potential of AI to enhance everyday tasks like cooking and meal planning. Special thanks to open-source libraries and datasets that made this project possible.

---

Feel free to fork this repository and contribute to its development. Your feedback and suggestions are always welcome! 😊
