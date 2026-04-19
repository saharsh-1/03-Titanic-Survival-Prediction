import pandas as pd

# ======================
# 1. Load Dataset
# ======================

df = pd.read_csv("train.csv")

print(df.head())
print(df.info())

print(df.shape)
print(df.columns)
print(df.isna().sum())

# ======================
# 2. Data Cleaning
# ======================

# Droping Cabin as it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Filling Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Filling Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print(df.isna().sum())

# ======================
# 3. Encoding
# ======================

# Encoding using Label Encoding
df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})
print(df['Sex'].head())

df['Embarked'] = df['Embarked'].map({'S' : 0, 'C' : 1, 'Q' : 2})
print(df['Embarked'].head())

# ======================
# 4. Feature Selection
# ======================

# Droping features which are not affecting prediction
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
print(df.columns)

# ======================
# 5. Train Test Split
# ======================

# Importing train test split
from sklearn.model_selection import train_test_split

# Preparing Data: X - Input and y - Output
X = df.drop(columns=['Survived'], axis=1)
y = df['Survived']

# Spliting data into train and test. (test_size=0.2 means 20% data into test and 80% data into train)(random_state makes results consistent, reproducible)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

# ======================
# 6. Model Training
# ======================

# Importing Model: We are using Logistic Regression Model from scikit learn
from sklearn.linear_model import LogisticRegression

# Creating Model (max_iter is number of iterations)
model = LogisticRegression(max_iter=1000)

# Training Model (Logistic Regression)
model.fit(X_train, y_train)

# ======================
# 7. Model Evaluation
# ======================

# Checking Model Score (.score() gives accuracy where Accuracy = Correct Predictions / Total Predictions)
print(model.score(X_test, y_test))

# ======================
# 8. Confusion Matrix
# ======================

# Importing Confusion matrix from scikit learn
from sklearn.metrics import confusion_matrix

# Predicting values
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

# ======================
# 9. Model Training (Decision Tree Model)
# ======================

# Importing Decision Tree Model from Scikit Learn
from sklearn.tree import DecisionTreeClassifier

# Creating and training model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Evaluating Model
print(model_dt.score(X_test, y_test))

# Confusion matrix (Decision Tree Model)
y_pred_dt = model_dt.predict(X_test)
print(confusion_matrix(y_test, y_pred_dt))