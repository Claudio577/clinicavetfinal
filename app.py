import pandas as pd
import numpy as np
import unicodedata
import re
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extrair_variavel(padrao, texto, tipo=float, valor_padrao=None):
    match = re.search(padrao, texto)
    if match:
        try:
            return tipo(match.group(1).replace(',', '.'))
        except:
            return valor_padrao
    return valor_padrao

@st.cache_data
def carregar_dados():
    df = pd.read_csv("Casos_Cl_nicos_Simulados.csv")
    df_doencas = pd.read_csv("doencas_caninas_eutanasia_expandidas.csv")
    df_curaveis = pd.read_csv("top100_doencas_caninas.csv")
    return df, df_doencas, df_curaveis

def treinar_modelos(df, features, features_eutanasia, df_doencas):
    le_mob = LabelEncoder()
    le_app = LabelEncoder()
    df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
    df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

    palavras_chave = [normalizar_texto(d) for d in df_doencas['Doença'].dropna().unique()]
    df['tem_doenca_letal'] = df['Doença'].fillna("").apply(lambda d: int(any(p in normalizar_texto(d) for p in palavras_chave)))

    X_eutanasia = df[features_eutanasia]
    y_eutanasia = df['Eutanasia']
    X_train, _, y_train, _ = train_test_split(X_eutanasia, y_eutanasia, test_size=0.2, stratify=y_eutanasia, random_state=42)

    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train, y_train)
    modelo_internar = RandomForestClassifier(class_weight='balanced', random_state=42).fit(df[features], df['Internar'])
    modelo_dias = RandomForestRegressor().fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_internar, modelo_dias, le_mob, le_app, palavras_chave

def prever(anamnese, modelos, le_mob, le_app, palavras_chave, features, features_eutanasia, palavras_curaveis):
    modelo_eutanasia, modelo_internar, modelo_dias = modelos
    texto_norm = normalizar_texto(anamnese)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:[.,]\d+)?)\s*(?:graus|c|celsius|ºc)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else 5

    dor = 4
    if any(p in texto_norm for p in ["dor intensa", "dor severa", "dor forte"]):
        dor = 10
    elif "dor moderada" in texto_norm:
        dor = 5
    elif any(p in texto_norm for p in ["sem dor", "ausência de dor"]):
        dor = 0

    if any(p in texto_norm for p in ["sem apetite", "não come", "perda de apetite"]):
        apetite = le_app.transform(["nenhum"])[0] if "nenhum" in le_app.classes_ else 0
    elif "baixo apetite" in texto_norm:
        apetite = le_app.transform(["baixo"])[0] if "baixo" in le_app.classes_ else 0
    else:
        apetite = le_app.transform(["normal"])[0] if "normal" in le_app.classes_ else 0

    if any(p in texto_norm for p in ["sem andar", "não anda", "não caminha"]):
        mobilidade = le_mob.transform(["sem andar"])[0] if "sem andar" in le_mob.classes_ else 0
    elif "limitada" in texto_norm:
        mobilidade = le_mob.transform(["limitada"])[0] if "limitada" in le_mob.classes_ else 0
    else:
        mobilidade = le_mob.transform(["normal"])[0] if "normal" in le_mob.classes_ else 0

    doencas_detectadas = [p for p in palavras_chave if p in texto_norm]
    tem_doenca_letal = int(len(doencas_detectadas) > 0)

    doencas_curaveis_detectadas = [d for d in palavras_curaveis if d in texto_norm]
    tem_doenca_curavel = int(len(doencas_curaveis_detectadas) > 0)

    entrada = pd.DataFrame([[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]],
                           columns=features_eutanasia)

    prob_eutanasia = modelo_eutanasia.predict_proba(entrada)[0][1]
    prob_internar = modelo_internar.predict_proba(entrada[features])[0][1]
    internar = 1 if prob_internar > 0.4 else 0

    sintomas_criticos = [
        "vômito", "febre", "diarreia", "apatia", "desidratação", "letargia",
        "sangramento", "prostração", "não levanta", "sem comer", "ofegante"
    ]
    sintomas_criticos_norm = [normalizar_texto(s) for s in sintomas_criticos]
    if any(s in texto_norm for s in sintomas_criticos_norm):
        internar = 1

    dias = int(round(modelo_dias.predict(entrada[features])[0]))
    if internar == 1 and dias < 2:
        dias = 2
    elif internar == 0:
        dias = 0

    alta = 1
    if (
        internar == 1
        or prob_eutanasia >= 0.05
        or tem_doenca_letal
        or temperatura > 39.0
        or any(s in texto_norm for s in sintomas_criticos_norm)
    ):
        alta = 0

    return {
        "Alta": "Sim" if alta == 1 else "Não",
        "Internar": "Sim" if internar == 1 else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": round(prob_eutanasia * 100, 1),
        "Doenças Detectadas": doencas_detectadas if doencas_detectadas else "Nenhuma grave",
        "Doença Potencialmente Curável": "Sim" if tem_doenca_curavel else "Não",
        "Doenças Curáveis Detectadas": doencas_curaveis_detectadas if doencas_curaveis_detectadas else "Nenhuma"
    }

# ========= INTERFACE STREAMLIT =========

st.title("🐶 Sistema de Análise Clínica Veterinária")

try:
    df, df_doencas, df_curaveis = carregar_dados()
    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']
    modelos = treinar_modelos(df, features, features_eutanasia, df_doencas)
    palavras_curaveis = [normalizar_texto(d) for d in df_curaveis['Doença'].dropna().unique()]
except Exception as e:
    st.error(f"Erro ao preparar o sistema: {str(e)}")
    st.stop()

texto = st.text_area("✍️ Digite a anamnese do paciente:")

if st.button("🔍 Analisar"):
    if texto.strip() == "":
        st.warning("Digite uma anamnese para analisar.")
    else:
        modelo_eutanasia, modelo_internar, modelo_dias = modelos[:3]
        le_mob, le_app, palavras_chave = modelos[3:]
        resultado = prever(texto, (modelo_eutanasia, modelo_internar, modelo_dias),
                           le_mob, le_app, palavras_chave,
                           features, features_eutanasia, palavras_curaveis)
        st.subheader("📋 Resultado da Análise")
        for k, v in resultado.items():
            st.write(f"**{k}**: {v}")

