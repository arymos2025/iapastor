import os
import pandas as pd
import streamlit as st # Importación de Streamlit
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuraciones de la APP ---
st.set_page_config(page_title="IA Pastor", layout="wide")
st.title("🕊️ IA Pastor: Consejero Bíblico")
st.markdown("Haz una pregunta y el Pastor IA responderá utilizando únicamente la Santa Biblia.")

# =================================================================
# CONFIGURACIÓN DE LA IA
# =================================================================

# 1. AUTENTICACIÓN: La clave se lee de los secretos de Streamlit Cloud.
# Esto es más seguro que ponerla en el código.
try:
    # Intenta leer la clave de los secretos de Streamlit
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
except KeyError:
    st.error("Error de configuración: La clave GEMINI_API_KEY no está configurada en los secretos de Streamlit.")
    st.stop() # Detiene la aplicación si la clave no se encuentra.

NOMBRE_ARCHIVO_PROCESADO = 'biblia_procesada.csv'
COLUMNA_TEXTO = 'texto' 
COLUMNA_META_DATA = 'cita_biblica' 
CHROMA_DIR = "./chroma_db"

# Usamos st.cache_resource para evitar que la app recargue y vectorice
# todo de nuevo con cada interacción.
@st.cache_resource
def inicializar_modelo():
    # 1. CARGA Y PREPARACIÓN DE DATOS
    try:
        df = pd.read_csv(NOMBRE_ARCHIVO_PROCESADO)
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo {NOMBRE_ARCHIVO_PROCESADO}. Asegúrate de subirlo a GitHub.")
        return None, None
        
    df[COLUMNA_META_DATA] = df['libro'] + ' ' + df['capitulo'].astype(str) + ':' + df['verso'].astype(str)

    loader = DataFrameLoader(df, page_content_column=COLUMNA_TEXTO) 
    documentos = loader.load()

    # 2. VECTORIZACIÓN (usando la clave del entorno)
    with st.spinner("Inicializando modelos y vectorizando la Biblia..."):
        embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
        
        # En Streamlit Cloud, forzaremos la creación porque la base de datos no es persistente.
        vector_store = Chroma.from_documents(
            documents=documentos, 
            embedding=embedding_model, 
            persist_directory=CHROMA_DIR
        )
        
        # 3. CONFIGURACIÓN DEL LLM Y RAG
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 

        prompt_template = """
        Eres un pastor, consejero espiritual y teólogo erudito. Tu única fuente de conocimiento son los siguientes versículos de la Santa Biblia, etiquetados como 'CONTEXTO'.
        ... (el mismo prompt que ya tienes) ...
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
    st.success("¡Pastor IA listo para la consulta!")
    return retrieval_chain

retrieval_chain = inicializar_modelo()

# =================================================================
# 4. CICLO DE CONVERSACIÓN (CON INTERFAZ WEB)
# =================================================================

if retrieval_chain:
    pregunta = st.text_input("Usuario (Busca ayuda):", key="user_input")
    
    if pregunta:
        with st.spinner("Pastor (Respondiendo...):"):
            try:
                response = retrieval_chain.invoke({"input": pregunta})
                respuesta_texto = response["answer"]
                
                # Muestra el texto en un cuadro de texto de Streamlit
                st.info(respuesta_texto)
                
                # Nota: La funcionalidad gTTS/Voz es más compleja de implementar en Streamlit Cloud
                # debido a que la reproducción de audio en un servidor es complicada. 
                # Sugerencia: Enfócate primero en la respuesta de texto.
                
            except Exception as e:
                st.error(f"Error al ejecutar la consulta: {e}")