Name : Keerthana

Company : Cognifyz Where Data Meets Intelligence

Ref. : CTI/A1/C121336

Domain : Machine Learning

Duration : 30/03/2025 and culminate on 30/04/2025

Overview of the Project


![Screenshot (25)](https://github.com/user-attachments/assets/cea47894-4903-458f-8201-f42e59ee6300)

![Screenshot (26)](https://github.com/user-attachments/assets/4292c1cf-20f9-48c5-937a-97aa8d75556f)

![Screenshot (27)](https://github.com/user-attachments/assets/85e40374-a375-49e5-8005-d22fc85bfd45)


Project : Predict Restaurant Rating

🍽️ Restaurant Rating Predictor

This web app predicts the aggregate rating of a restaurant using machine learning based on various restaurant-related features from a dataset. Built with Streamlit, it provides an interactive interface to upload a dataset, train a model, and visualize feature importances.

🚀 Features

Upload your own CSV restaurant dataset

Automatic data cleaning and preprocessing

Encoding of categorical variables

Train/test split and feature scaling

Model training using Random Forest Regressor

Evaluation using MSE and R-squared metrics

Visual insights into top features affecting ratings

🧠 Machine Learning Workflow

Data Preprocessing:

Drops irrelevant columns like Restaurant ID, Address, etc.

Fills missing numerical values with median

Fills missing categorical values with mode

Encodes categorical columns using one-hot encoding

Model Training:

Splits data into train/test sets (80/20)

Applies standard scaling

Trains a RandomForestRegressor on training data

Model Evaluation:

Evaluates predictions using Mean Squared Error (MSE) and R-squared (R²)

Displays the top 10 most influential features using bar charts

📂 File Structure

bash
Copy
Edit
├── app.py              # Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
🛠️ Installation

🔧 1. Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/restaurant-rating-predictor.git
cd restaurant-rating-predictor

📦 2. Install Requirements

bash
Copy
Edit
pip install -r requirements.txt

▶️ 3. Run the App

bash
Copy
Edit
streamlit run app.py
📋 Requirements
Python 3.7+

pandas

numpy

seaborn

matplotlib

scikit-learn

streamlit

📝 Sample Dataset Format
Your dataset should include columns such as:

Cuisines

Price range

Has Table booking

Has Online delivery

Aggregate rating (target column)

📊 Output
📈 Model performance metrics

🔍 Top 10 features influencing restaurant ratings

💡 Future Enhancements
Allow model selection (Linear, Decision Tree, etc.)

Support for multiple datasets

Export trained model for deployment

🤝 Contributing
Feel free to fork the repo and submit pull requests for improvements.
