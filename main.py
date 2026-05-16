from click import prompt
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import time


# Carrega o Modelo e o Tokenizer do Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")


# Cria um Pipeline de Geração de Texto
generator = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", tokenizer=tokenizer)

# Função para Gerar Texto com Base em um Prompt
def gerar_texto(prompt, max_length=2048):
    system_prompt = "You are AI Agent called 'CogniTrace' specialized into psychology. Start your answer with 'CogniTrace'."
    resultado = generator(system_prompt + "\n" + prompt, max_length=max_length, num_return_sequences=1)
    return resultado[0]['generated_text']

def typewriter_effect(text: str, speed: float = 0.03):
    """Efeito máquina de escrever."""
    container = st.empty()
    displayed = ""
    for ch in text:
        displayed += ch
        container.text(f"\n{displayed}█\n")
        time.sleep(speed)
    container.text(f"\n{displayed}\n")

# Implementar a função 'gerar_texto' para ser usada no Streamlit
def main():
    st.title("CogniTrace AI")
    prompt = st.chat_input("Digite:")
    
    if prompt:
     with st.spinner("Gerando..."):
        with st.chat_message('ai'):
         output = gerar_texto(prompt)
         typewriter_effect(output)
    else:
        st.warning("Por favor, insira um input.")

if __name__ == "__main__":
    main()









