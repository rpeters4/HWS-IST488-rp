
import google.generativeai as genai
import toml
import os

try:
    with open('.streamlit/secrets.toml', 'r') as f:
        secrets = toml.load(f)
    api_key = secrets.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in secrets.")
        exit(1)
        
    genai.configure(api_key=api_key)
    
    print("Listing available models:")
    for m in genai.list_models():
            with open("models.txt", "a") as out:
                out.write(f"- {m.name}\n")
            
except Exception as e:
    with open("models.txt", "a") as out:
         out.write(f"Error: {e}\n")
