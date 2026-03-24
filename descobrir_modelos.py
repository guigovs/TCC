import streamlit as st
from google import genai

try:
    client = genai.Client(api_key=st.secrets["GEMINI_CHAVE"])
    print("Buscando modelos liberados para a sua chave...\n")
    
    # Busca a lista de modelos usando a sintaxe da biblioteca nova
    for m in client.models.list():
        print(f"- {m.name}")
            
except Exception as e:
    print(f"Ocorreu um erro: {e}")