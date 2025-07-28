import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv("C:\Python\openai\logistic_regression\Titanic_train.csv")

titanic.head(5)

titanic.info()

titanic.describe()

titanic.isnull().sum()

titanic.nunique()

titanic["Age"].median()

titanic["Age"].mean()

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())

titanic = titanic.drop(["PassengerId","Cabin","Ticket","Name"],axis = 1)

titanic.head(5)

titanic["Embarked"].unique()

titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])


dummies = pd.get_dummies(titanic[['Sex', 'Embarked']], drop_first=True)

titanic = pd.concat([titanic.drop(['Sex', 'Embarked'], axis=1), dummies], axis=1)

titanic.head(5)

titanic.info()

plt.figure(figsize=(14, 8))
sns.boxplot(data=titanic.select_dtypes(include='number'))
plt.xticks(rotation=30, fontsize=12)
plt.title("Boxplot of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

def count_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers)

count_outliers_iqr(titanic,"Age")

count_outliers_iqr(titanic,"Fare")

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower, upper)

# Apply to Age and Fare
cap_outliers_iqr(titanic, 'Age')
cap_outliers_iqr(titanic, 'Fare')

plt.figure(figsize=(14, 8))
sns.boxplot(data=titanic.select_dtypes(include='number'))
plt.xticks(rotation=30, fontsize=12)
plt.title("Boxplot of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.show()


sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic.drop('Survived',axis=1),titanic['Survived'], test_size=0.33,random_state=101)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

from sklearn.metrics import classification_report
import sklearn.metrics as metrics

print(classification_report(y_test,predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Get predicted probabilities for class 1
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Step 2: Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Step 3: Compute AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {auc_score:.4f}")

# Step 4: Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

import pickle
with open("logistic_model.pkl", "wb") as file:
    pickle.dump(model, file)

import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("logistic_model.pkl", "rb"))
# App title
st.title("🚢 Titanic Survival Prediction App")

# User inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 30)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=8, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=6, step=1)
embarked_q = st.checkbox("Embarked at Queenstown (Q)")
embarked_s = st.checkbox("Embarked at Southampton (S)")

# Convert inputs
sex_male = 1 if sex == "male" else 0
input_features = np.array([[pclass, age, fare, sibsp,parch,sex_male, int(embarked_q), int(embarked_s)]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
    st.markdown(f"### Prediction: **{result}**")
    st.write(f"Survival Probability: **{probability:.2f}**")