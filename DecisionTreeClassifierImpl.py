import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('obesity_data.csv')
df.head()

#Label Encoding
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

X, y = df.drop(['ObesityCategory'], axis =1), df['ObesityCategory']

decision_tree_classifier = DecisionTreeClassifier()

# Split test data for later testing before training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_folds = 10
stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

for train_index, test_index in tqdm(stratified_kfold.split(X, y), total=k_folds, desc="Cross-validation"):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    decision_tree_classifier.fit(X_train, y_train)

    y_pred = decision_tree_classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
