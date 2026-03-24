from google import genai

CHAVE_API = "AIzaSyAFV2OtDepbsXBMihO1aeuT7uIkRIhTn_8"

try:
    client = genai.Client(api_key=CHAVE_API)
    print("Buscando modelos liberados para a sua chave...\n")
    
    # Busca a lista de modelos usando a sintaxe da biblioteca nova
    for m in client.models.list():
        print(f"- {m.name}")
            
except Exception as e:
    print(f"Ocorreu um erro: {e}")