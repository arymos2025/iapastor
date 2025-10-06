import streamlit as st
import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings # <--- IMPORTACIÓN GRATUITA
from gtts import gTTS
import tempfile

# --- CONFIGURACIÓN DE RUTAS Y MODELOS ---
CHROMA_DIR = "./chroma_db"
BIBLIA_PATH = "biblia_procesada.csv"

# --- FUNCIÓN DE INICIALIZACIÓN (CACHING) ---
@st.cache_resource
def inicializar_modelo():
    """Inicializa los modelos de lenguaje, embeddings y el almacén vectorial."""
    
    # 1. CARGA DE DATOS
    try:
        df = pd.read_csv(BIBLIA_PATH)
        documents = []
        for _, row in df.iterrows():
            documents.append(Document(page_content=row['texto'], metadata={"fuente": row['referencia']}))
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo '{BIBLIA_PATH}'. Asegúrate de que está en tu repositorio.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar la Biblia: {e}")
        st.stop()
        
    # 2. MODELO DE EMBEDDINGS (GRATUITO Y LOCAL)
    with st.spinner("Inicializando modelos y vectorizando la Biblia (esto puede tardar unos minutos)..."):
        try:
            # Usamos HuggingFaceEmbeddings con model_name para evitar el error de Pydantic
            # Esto NO requiere la clave de Gemini para la vectorización
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2" # <--- CAMBIO CLAVE PARA SOLUCIONAR EL ERROR
            )
            
            # 3. CREACIÓN DE LA BASE DE DATOS VECTORIAL
            # Cargar o crear la base vectorial con el modelo gratuito
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR
            )
        except Exception as e:
            st.error(f"Error en la vectorización (Línea 54): {e}. Asegúrate de que 'sentence-transformers' esté en requirements.txt.")
            st.stop()

    # 4. MODELO DE CHAT (AÚN NECESITA LA CLAVE GEMINI_API_KEY)
    try:
        # La clave todavía es necesaria para que el modelo Gemini genere la respuesta.
        # Streamlit lee automáticamente la clave de los Secrets.
        google_api_key = st.secrets["GEMINI_API_KEY"] 
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.0, 
            google_api_key=google_api_key
        )
    except Exception as e:
        st.error("Error de configuración: La clave GEMINI_API_KEY no es válida o no está configurada correctamente en los secretos de Streamlit. El chat no funcionará.")
        st.stop()


    # 5. CONFIGURACIÓN DEL PROMPT
    prompt_template = """Eres un experto consejero bíblico. Tu única fuente de conocimiento es el 'Contexto' que se te proporciona, 
    que contiene versículos de la Santa Biblia.

    Instrucciones:
    1. Responde a la pregunta del usuario basándote estrictamente en el 'Contexto' proporcionado.
    2. Si el 'Contexto' no tiene información relevante, responde con: "Lo siento, mi respuesta debe basarse únicamente en la Biblia proporcionada, y no encontré versículos directamente relevantes para esta pregunta."
    3. Siempre incluye las referencias bíblicas completas (Libro, Capítulo:Versículo) de donde tomaste la respuesta.
    4. No inventes versículos ni referencias.

    CONTEXTO:
    {context}

    PREGUNTA DEL USUARIO: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 6. CONFIGURACIÓN DE LA CADENA DE RECUPERACIÓN
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return retrieval_chain

# --- FUNCIÓN PARA GENERAR AUDIO ---
def generar_audio(texto):
    """Genera y reproduce audio a partir del texto."""
    try:
        tts = gTTS(text=texto, lang='es')
        # Usar un archivo temporal para guardar y reproducir
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_path = fp.name
        st.audio(audio_path)
        os.remove(audio_path) # Limpiar el archivo temporal
    except Exception as e:
        st.error(f"Error al generar audio: {e}")

# --- INTERFAZ DE STREAMLIT ---
def main():
    st.set_page_config(page_title="IA Pastor: Consejero Bíblico", layout="centered")
    
    st.image("ia_pastor_logo.png", width=100) # Reemplaza "ia_pastor_logo.png" con tu logo real si lo tienes.
    st.title("IA Pastor: Consejero Bíblico")
    st.markdown("Haz una pregunta y el Pastor IA responderá utilizando únicamente la Santa Biblia.")

    # Inicializa el modelo y la cadena de recuperación (usa el resultado de la función inicializar_modelo)
    retrieval_chain = inicializar_modelo()
    
    st.success("¡Pastor IA listo para la consulta!")

    # Historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes del historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Escribe tu pregunta bíblica aquí..."):
        # Mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Mensaje de la IA
        with st.chat_message("assistant"):
            with st.spinner("Buscando y generando respuesta..."):
                full_response = retrieval_chain.invoke(prompt)
                st.markdown(full_response)
            
            # Botón de audio
            if st.button("Escuchar respuesta"):
                generar_audio(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if _name_ == "_main_":
    main()