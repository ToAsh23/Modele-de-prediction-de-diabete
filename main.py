import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.patches import Patch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Vision générale", layout="wide")
st.title('Projet : Prédiction de diabète')
st.header('Introduction')
texte = """
le dièate est une maladie

c'est pourquoi
"""
st.markdown(texte)
st.divider()

df = pd.read_csv('diabetes_prediction_dataset.csv')
st.header("Vous pouvez voir ici la base de donnée complète pour l'entrainement de nos modèles")
st.dataframe(df)

st.write("Vous avez ici la description du DataFrame :")
st.write(df.describe())
st.write(df.info())

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

df_3['smoking_history'] = df_3["smoking_history"].map({'never': 0, 'not current': 1, 'former': 2, 'ever': 2, 'current': 3})
df_3
st.divider()
st.header("Analyse exploratoire : visualisations")




st.subheader('Analyse interactive des variables')
variables = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level','diabetes']
variable_select = st.selectbox('Choisissez une variable', variables)

st.write("Sélectionnez la variable quantitative que vous voulez explorer :")


if st.checkbox('Afficher les analyses et graphiques'):
    st.write(f'Variable sélectionnée : **{variable_select}**')
    numeric_data = df_3.dropna(subset=[variable_select])
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(numeric_data, x=variable_select, nbins=20, color_discrete_sequence=["mediumturquoise"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.box(numeric_data, y=variable_select, color_discrete_sequence=["orange"])
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.metric("Min", np.round(numeric_data[variable_select].min(), 3))
        st.metric("Max", np.round(numeric_data[variable_select].max(), 3))
        st.metric("Moyenne", np.round(numeric_data[variable_select].mean(), 3))


st.divider()

st.header ('Distribution en fonction du diabète')

st.subheader("Boîte à moustache en fonction du Diabète")

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
st.subheader("Histogramme")
# Histogrammes
fig2, axarr = plt.subplots(2, 2, figsize=(10, 10))
sns.histplot(df_3.age, bins=20, ax=axarr[0,0], color="red")
sns.histplot(df_3.bmi, bins=20, ax=axarr[0,1], color="red")
sns.histplot(df_3.blood_glucose_level, bins=20, ax=axarr[1,0], color="red")
sns.histplot(df_3.HbA1c_level, bins=20, ax=axarr[1,1], color="red")
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Matrice de corrélation entre les Inputs et le Output")
fig3, ax3 = plt.subplots(figsize=[10,5])
sns.heatmap(df_3.corr(), annot=True, fmt='.2f', ax=ax3, cmap='coolwarm')
ax3.set_title("Correlation Matrix", fontsize=20)
st.pyplot(fig3)

st.subheader("Distribution des personnes atteintes de diabète après les prétraitements")
fig4, ax = plt.subplots(1,2, figsize=(18,8))
df_3['diabetes'].value_counts().plot.pie(explode=[0,0.1], autopct = "%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot(x='diabetes', data=df_3, ax=ax[1])
ax[1].set_title('diabetes')
st.pyplot(fig4)

# Régression linéaire plots
st.subheader("Dans les figures ci-dessous on peut voir la corrélation direct entre les Inputs et le Output")
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


st.divider()
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

st.header("Scores des deux modèles")
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

st.subheader("Evaluation du model LR")
st.text(classification_report(y_test, log_reg.predict(X_test)))
st.subheader("Evalutaion du RF")
st.text(classification_report(y_test, rand_clf.predict(X_test)))
st.write("Matrice de confusion LR :", confusion_matrix(y_test, log_reg.predict(X_test)))
st.write("Matrice de confusion RF :", confusion_matrix(y_test, rand_clf.predict(X_test)))

# Tu peux compléter avec tous tes tableaux à afficher ainsi :
'''st.write("Description du dataset filtré :")
st.dataframe(df_3.describe())
st.write("Valeurs manquantes (filtrées) :")
st.write(df_3.isnull().sum())'''


st.divider()
st.set_page_config(page_title="Prédiction du diabète", layout="wide")
st.title("Application Prédiction Diabète")

# ==== Charger les données et entrainer les modèles ====
@st.cache_data
def load_and_train():
    # Remplace par ton vrai csv si nécessaire
    df = pd.read_csv('diabetes_prediction_dataset.csv')

    df = df[df["smoking_history"] != "No Info"]
    df = df[df["gender"] != "Other"]
    df['gender'] = df["gender"].map({'Female': 0, 'Male': 1})
    df['smoking_history'] = df["smoking_history"].map({'never': 0, 'not current': 1, 'former': 2, 'ever': 2, 'current': 3})

    X = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
    y = df['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression()
    rf = RandomForestClassifier(criterion = 'entropy', max_depth = 15, max_features = 0.75, min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf

logreg, randforest = load_and_train()

# ==== Choix du mode ====
mode = st.radio("Mode d'entrée", ["Uploader un fichier CSV", "Remplir le formulaire manuellement"])

input_df = None

if mode == "Uploader un fichier CSV":
    uploaded_file = st.file_uploader("Choisis un fichier CSV contenant les colonnes requises :", type=["csv"])
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Aperçu des entrées :")
        st.dataframe(df_input)
        input_df = df_input

elif mode == "Remplir le formulaire manuellement":
    st.write("Remplis les informations pour un seul patient :")
    gender = st.selectbox("Genre", ["Homme", "Femme"])
    gender_num = 1 if gender == "Homme" else 0
    age = st.number_input("Âge", min_value=1, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension ?", ["Non", "Oui"])
    heart_disease = st.selectbox("Problème cardiaque ?", ["Non", "Oui"])
    smoking = st.selectbox("Historique tabagique", ["never", "not current", "former", "ever", "current"])
    bmi = st.number_input("IMC", min_value=5.0, max_value=90.0, value=25.0)
    a1c = st.number_input("HbA1c", min_value=2.0, max_value=18.0, value=5.2)
    glucose = st.number_input("Glycémie", min_value=30.0, max_value=600.0, value=100.0)

    input_df = pd.DataFrame({
        'gender': [gender_num],
        'age': [age],
        'hypertension': [1 if hypertension=="Oui" else 0],
        'heart_disease': [1 if heart_disease=="Oui" else 0],
        'smoking_history': [ {'never': 0, 'not current': 1, 'former': 2, 'ever': 2, 'current': 3}[smoking] ],
        'bmi': [bmi],
        'HbA1c_level': [a1c],
        'blood_glucose_level': [glucose]
    })

    st.write("Aperçu de votre saisie :")
    st.dataframe(input_df)

# ==== Prédiction ====
if input_df is not None and st.button("Prédire le diabète"):
    pred_lr = logreg.predict(input_df)
    pred_rf = randforest.predict(input_df)
    st.success(f"**Régression logistique prédit :** {'Diabétique' if pred_lr[0]==1 else 'Non diabétique'}")
    st.success(f"**Random Forest prédit :** {'Diabétique' if pred_rf[0]==1 else 'Non diabétique'}")

