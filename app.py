import numpy as np
import gdown
import tensorflow as tf
import streamlit as st

# --- Configuration page ---
st.set_page_config(
    page_title="🧠 Détection des Tumeurs Cérébrales - ResNet50",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Charger le modèle ---
MODEL_PATH = "projetResnet50.keras"
DRIVE_ID = "1n1ztaaFx4pFUXOVLwa8Q9b0fLBygt_9F"  # juste l'ID

@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Téléchargement du modèle depuis Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
        st.success("✅ Modèle téléchargé !")
    return tf.keras.models.load_model(MODEL_PATH)

model = charger_modele()
# Classes du modèle
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# --- CSS Design ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
    }
    .detected {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
    }
    .normal {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
    }
    .center-img {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .center-img img {
        width: 300px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🧠 Détection des Tumeurs Cérébrales (IRM)</h1>
    <p>Analyse d’images IRM avec un modèle <strong>ResNet50</strong> entraîné sur 4 types de tumeurs.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🧬 À propos")
    st.success("""
    Modèle **ResNet50** fine-tuné pour classifier :
    - Gliome  
    - Méningiome  
    - Pituitaire  
    - Cerveau sain
    """)
    st.markdown("""
    **⚙️ Fonctionnement :**
    - 📥 Entrée : Image IRM cérébrale (JPG/PNG)  
    - 📊 Sortie : Type de tumeur probable
    """)
    st.info("📈 **Précision estimée : ~96%**")
    st.warning("""
    ⚠️ **Note :**  
    Cet outil est à but **éducatif** et ne remplace pas un diagnostic médical professionnel.
    """)

# --- Upload image ---
uploaded_file = st.file_uploader("📤 Choisissez une image IRM cérébrale...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))

    # Affichage image centrée et petite
    st.markdown('<div class="center-img">', unsafe_allow_html=True)
    st.image(img, caption="IRM Cérébrale Chargée", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # Prétraitement
    img_array = np.array(img)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    preds = model.predict(img_array)
    index_pred = np.argmax(preds)
    proba_pred = np.max(preds)
    label_pred = CLASS_NAMES[index_pred]

    # Style selon résultat
    is_tumor = label_pred != "no_tumor"
    style_box = "detected" if is_tumor else "normal"
    couleur_barre = "#dc3545" if is_tumor else "#28a745"

    conseil = (
        "⚠️ Tumeur détectée : veuillez consulter un neurologue pour des examens approfondis."
        if is_tumor
        else "✅ Aucun signe de tumeur détecté."
    )

    # --- Résultat ---
    st.markdown(f"""
    <div class="result-box {style_box}">
        <h2>{'🧠 ' + label_pred.upper()}</h2>
        <p><strong>Probabilité :</strong> {proba_pred:.2%}</p>
        <div style="background-color: #e9ecef; border-radius: 25px; height: 20px; overflow: hidden; margin-top: 1rem;">
            <div style="width: {proba_pred*100}%; height: 100%; background-color: {couleur_barre}; border-radius: 25px;"></div>
        </div>
        <p style="margin-top: 1rem; font-weight: bold;">{conseil}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🧠 Application de Détection des Tumeurs Cérébrales</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créée par <strong>Youssouf</strong> avec <strong>TensorFlow + Streamlit</strong>.
    </p>
    <p style="font-size: 0.9em; color: #6c757d;">
        Version 2025 — Modèle ResNet50 fine-tuné
    </p>
</div>
""", unsafe_allow_html=True)
