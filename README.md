#🚢 Logistic Regression on Titanic Dataset with Streamlit Deployment
This project uses logistic regression to predict survival on the Titanic dataset based on passenger features such as class, age, gender, and more. It also includes a Streamlit app to allow user-friendly model interaction.

🧠 What is Logistic Regression?
Logistic Regression is a supervised machine learning algorithm used for binary classification problems. Unlike linear regression, which predicts continuous outputs, logistic regression predicts probabilities that map to binary classes (e.g., Survived or Not Survived).

🔸 Mathematical Formula:
𝑃
(
𝑌
=
1
)
=
1
1
+
𝑒
−
(
𝛽
0
+
𝛽
1
𝑋
1
+
…
+
𝛽
𝑛
𝑋
𝑛
)
P(Y=1)= 
1+e 
−(β 
0
​
 +β 
1
​
 X 
1
​
 +…+β 
n
​
 X 
n
​
 )
 
1
​
 
Where:

𝑃
(
𝑌
=
1
)
P(Y=1): Probability of the positive class (e.g., survival)

𝛽
0
β 
0
​
 : Intercept

𝛽
𝑛
β 
n
​
 : Coefficients for the features

Output is interpreted as a probability, which is then classified using a threshold (typically 0.5)

📁 Dataset
File Used: Titanic_train.csv

Target Variable: Survived (0 = No, 1 = Yes)

Features Considered:

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, etc.

🧪 Implementation Steps
1. Data Loading and Inspection
Read the dataset and checked its structure using .head(), .info(), and .describe().

Verified presence of null values and variable types.

2. Data Cleaning and Preprocessing
Handled missing values (e.g., filled or dropped nulls in Age, Embarked)

Converted categorical variables (Sex, Embarked) using one-hot encoding

Selected relevant numerical and encoded features for modeling

3. Model Building
Split data into training and testing sets using train_test_split

Trained a Logistic Regression model using sklearn.linear_model.LogisticRegression

Evaluated model using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📊 Why Logistic Regression?
It's efficient and interpretable for binary classification tasks

Useful for estimating the probability of class membership

Well-suited for datasets like Titanic where the output is binary (survived or not)

🌐 Streamlit Integration
To make the model accessible via a web interface, a Streamlit app was created:

Accepts user input for key features (Age, Pclass, Fare, etc.)

Applies the same preprocessing steps

Uses the trained logistic regression model to predict survival

Returns a probability of survival and a classification

This allows non-technical users to interact with the model and explore predictions visually.

🛠️ Tools & Libraries
pandas, numpy – data loading and manipulation

matplotlib, seaborn – data visualization

sklearn – model training, evaluation, and preprocessing

Streamlit – web app deployment

📂 Files
Logistic_reg.ipynb – Complete notebook for logistic regression modeling

Titanic_train.csv – Dataset used

logistic_streamlit_app.py – Streamlit app for live predictions (optional if implemented)
