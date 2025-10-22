import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix

st.set_page_config(page_title="Vision générale", layout="wide")
st.title('Projet : Prédiction de diabète')

# Chargement des données
df = pd.read_csv('diabetes_prediction_dataset.csv')
st.header("Aperçu des données")
st.dataframe(df.head(10))

# Nettoyage
df_1 = df[df["smoking_history"] != "No Info"]
df_2 = df_1[df_1["gender"] != "Other"]
df_2['gender'] = df_2["gender"].map({'Female': 0, 'Male': 1})
df_2['smoking_history'] = df_2["smoking_history"].map({'never': 0, 'not current': 1, 'former': 2, 'ever': 2, 'current': 3})
df_3 = df_2.copy()

# Sélection des variables
col_name = df_3.drop('diabetes', axis=1).columns[:]
x = df_3.loc[:, col_name]
y = df_3['diabetes']

# Split des données
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modèles
log_reg = LogisticRegression()
rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75, min_samples_leaf=2, min_samples_split=3, n_estimators=130)

models = [
    {'label': 'LR', 'model': log_reg},
    {'label': 'RF', 'model': rand_clf}
]

means_roc = []
means_accuracy = []

for m in models:
    model = m['model']
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    means_accuracy.append(100 * round(acc, 4))
    auc = roc_auc_score(y_test, y_pred)
    means_roc.append(100 * round(auc, 4))

st.header("Scores des modèles")
st.write("Accuracy (%) LR et RF :", means_accuracy)
st.write("ROC-AUC (%) LR et RF :", means_roc)

# Barplot streamlit
fig, ax = plt.subplots(figsize=(5,5))
index = np.arange(len(models))
bar_width = 0.35

rects1 = ax.bar(index, means_accuracy, bar_width,
                alpha=0.8, color='mediumpurple', label='Accuracy (%)')
rects2 = ax.bar(index + bar_width, means_roc, bar_width,
                alpha=0.8, color='rebeccapurple', label='ROC-AUC (%)')
ax.set_xlim([-1, 3])
ax.set_ylim([0, 100])
ax.set_title('Performance Evaluation - Diabetes Prediction', fontsize=12)
ax.set_xticks(index + bar_width/2)
ax.set_xticklabels(['LR', 'RF'], rotation=0, fontsize=12)
ax.legend(loc="upper right", fontsize=10)
st.pyplot(fig)

# Exemple pour afficher le classification report
st.subheader("Classification report LR")
st.text(classification_report(y_test, log_reg.predict(X_test)))
st.subheader("Classification report RF")
st.text(classification_report(y_test, rand_clf.predict(X_test)))

# (Tu peux ajouter d'autres visualisations avec matplotlib/seaborn comme ci-dessus)

# Fin du script
