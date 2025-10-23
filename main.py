import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Vision générale", layout="wide")
st.title('Projet : Prédiction de diabète')

df = pd.read_csv('diabetes_prediction_dataset.csv')
st.header("Aperçu des données")
st.dataframe(df.head(10))
st.write("Valeurs manquantes par colonne :")
st.write(df.isnull().sum())

df_1 = df[df["smoking_history"] != "No Info"]
st.write("Info DataFrame filtré:")
st.dataframe(df_1.info())
st.dataframe(df_1.describe())
st.write("Valeurs manquantes après filtre :")
st.write(df_1.isnull().sum())

df_2 = df_1[df_1["gender"] != "Other"]
df_2['gender'] = df_2["gender"].map({'Female': 0, 'Male': 1})
df_3 = df_2.copy()

valeurs_uniques_smoking = df_3['smoking_history'].unique()
st.write("Valeurs uniques de smoking_history :", valeurs_uniques_smoking)

df_3['smoking_history'] = df_3["smoking_history"].map({'never': 0, 'not current': 1, 'former': 2, 'ever': 2, 'current': 3})

st.header("Analyse exploratoire : visualisations")
num_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
colors = {0: "steelblue", 1: "salmon"}
labels = {0: "No Diabetes", 1: "Diabetes"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, col in zip(axes, num_features):
    data_to_plot = [df_3[df_3["diabetes"] == d][col].dropna() for d in [0, 1]]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)
    for patch, d in zip(bp['boxes'], [0, 1]):
        patch.set_facecolor(colors[d])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([labels[0], labels[1]])
    ax.set_title(f"{col} par Statut de diabète")
    ax.set_ylabel(col)
fig.legend([Patch(facecolor=colors[0]), Patch(facecolor=colors[1])],
           labels.values(), loc="upper right", title="Statut de diabète")
plt.tight_layout()
st.pyplot(fig)

# Histogrammes
fig2, axarr = plt.subplots(2, 2, figsize=(10, 10))
sns.histplot(df_3.age, bins=20, ax=axarr[0,0], color="red")
sns.histplot(df_3.bmi, bins=20, ax=axarr[0,1], color="red")
sns.histplot(df_3.blood_glucose_level, bins=20, ax=axarr[1,0], color="red")
sns.histplot(df_3.HbA1c_level, bins=20, ax=axarr[1,1], color="red")
plt.tight_layout()
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=[10,5])
sns.heatmap(df_3.corr(), annot=True, fmt='.2f', ax=ax3, cmap='coolwarm')
ax3.set_title("Correlation Matrix", fontsize=20)
st.pyplot(fig3)

# Pie et count plot
fig4, ax = plt.subplots(1,2, figsize=(18,8))
df_3['diabetes'].value_counts().plot.pie(explode=[0,0.1], autopct = "%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot(x='diabetes', data=df_3, ax=ax[1])
ax[1].set_title('diabetes')
st.pyplot(fig4)

# Régression linéaire plots
fig5, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 7))
sns.regplot(y=df_3['diabetes'], x=df_3['gender'], ax=axs[0, 0])
sns.regplot(y=df_3['diabetes'], x=df_3['age'], ax=axs[0, 1])
sns.regplot(y=df_3['diabetes'], x=df_3['hypertension'], ax=axs[0, 2])
sns.regplot(y=df_3['diabetes'], x=df_3['heart_disease'], ax=axs[0, 3])
sns.regplot(y=df_3['diabetes'], x=df_3['smoking_history'], ax=axs[1, 0])
sns.regplot(y=df_3['diabetes'], x=df_3['bmi'], ax=axs[1, 1])
sns.regplot(y=df_3['diabetes'], x=df_3['HbA1c_level'], ax=axs[1, 2])
sns.regplot(y=df_3['diabetes'], x=df_3['blood_glucose_level'], ax=axs[1, 3])
plt.tight_layout()
st.pyplot(fig5)

# Split data
col_name = df_3.drop('diabetes', axis=1).columns[:]
x = df_3.loc[:, col_name]
y = df_3['diabetes']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

# Barplot Streamlit
fig6, ax6 = plt.subplots(figsize=(5,5))
index = np.arange(len(models))
bar_width = 0.35
rects1 = ax6.bar(index, means_accuracy, bar_width,
                alpha=0.8, color='mediumpurple', label='Accuracy (%)')
rects2 = ax6.bar(index + bar_width, means_roc, bar_width,
                alpha=0.8, color='rebeccapurple', label='ROC-AUC (%)')
ax6.set_xlim([-1, 3])
ax6.set_ylim([0, 100])
ax6.set_title('Performance Evaluation - Diabetes Prediction', fontsize=12)
ax6.set_xticks(index + bar_width/2)
ax6.set_xticklabels(['LR', 'RF'], rotation=0, fontsize=12)
ax6.legend(loc="upper right", fontsize=10)
st.pyplot(fig6)

st.subheader("Classification report LR")
st.text(classification_report(y_test, log_reg.predict(X_test)))
st.subheader("Classification report RF")
st.text(classification_report(y_test, rand_clf.predict(X_test)))
st.write("Matrice de confusion LR :", confusion_matrix(y_test, log_reg.predict(X_test)))
st.write("Matrice de confusion RF :", confusion_matrix(y_test, rand_clf.predict(X_test)))

# Tu peux compléter avec tous tes tableaux à afficher ainsi :
st.write("Description du dataset filtré :")
st.dataframe(df_3.describe())
st.write("Valeurs manquantes (filtrées) :")
st.write(df_3.isnull().sum())
