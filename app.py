import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import Chroma  # <--- Importación estable desde la librería base
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer  # <--- Modelo puro y gratuito
from gtts import gTTS
import tempfile

# --- CONFIGURACIÓN DE RUTAS ---
BIBLIA_PATH = "biblia_procesada.csv"

# --- CLASE ADAPTADORA PARA CHROMA (WRAPPER) ---
# Usamos SentenceTransformer directamente para evitar conflictos de Pydantic y versiones de LangChain.
class CustomEmbeddings:
    """Clase adaptadora que usa SentenceTransformer para LangChain/Chroma."""
    def _init_(self, model_name='all-MiniLM-L12-v2'): # <--- MODELO ULTRALIGERO (60MB)
        # Carga el modelo ultraligero directamente
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        # Implementa la función que Chroma usa para vectorizar documentos
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        # Implementa la función que Chroma usa para vectorizar consultas
        return self.model.encode(text).tolist()

# --- FUNCIÓN DE INICIALIZACIÓN (CACHING) ---
@st.cache_resource
def inicializar_modelo():
    """Inicializa los modelos de lenguaje, embeddings y el almacén vectorial."""
    
    # 1. CARGA DE DATOS
    try:
        # Verifica que el archivo de la Biblia esté en el mismo directorio
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
            # Usamos la clase adaptadora con el modelo ultraligero
            embedding_model = CustomEmbeddings()
            
            # 3. CREACIÓN DE LA BASE DE DATOS VECTORIAL
            # Creada solo en memoria para evitar el fallo de persistencia y memoria en Streamlit Cloud
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model
            )
        except Exception as e:
            # Si Streamlit se queda sin memoria, mostrará este error explícito.
            st.error(f"Error CRÍTICO en la vectorización: {e}. La memoria de Streamlit falló al procesar el modelo y la Biblia.")
            st.stop()

    # 4. MODELO DE CHAT (USA LA CLAVE GEMINI_API_KEY)
    try:
        # Se requiere la clave de secretos de Streamlit
        google_api_key = st.secrets["GEMINI_API_KEY"] 
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.0, 
            google_api_key=google_api_key
        )
    except Exception:
        # Muestra el error si la clave no está configurada
        st.error("Error de configuración: La clave GEMINI_API_KEY no es válida o no está configurada en los secretos de Streamlit.")
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
    except Exception:
        st.warning("No se pudo generar el audio. El servicio de gTTS falló.")

# --- INTERFAZ DE STREAMLIT ---
def main():
    st.set_page_config(page_title="IA Pastor: Consejero Bíblico", layout="centered")
    
    # st.image("ia_pastor_logo.png", width=100) # Imagen desactivada para evitar errores de carga

    st.title("IA Pastor: Consejero Bíblico")
    st.markdown("Haz una pregunta y el Pastor IA responderá utilizando únicamente la Santa Biblia.")

    # Inicializa el modelo y la cadena de recuperación (retrieval_chain)
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

if __name__ == "_main_":
    main()