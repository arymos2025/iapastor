import streamlit as st
import pandas as pd
# 🚨 CAMBIO DE CONTINGENCIA: Usamos la librería antigua
import pinecone 
from sentence_transformers import SentenceTransformer
import random

# --- 0. Configuración de la Página ---
st.set_page_config(
    page_title="Buscador Bíblico Vectorial",
    page_icon="📖",
    layout="wide"
)

# --- 1. CONEXIÓN A PINECONE Y MODELO (USANDO st.secrets) ---

@st.cache_resource
def get_embedding_model():
    """Carga el modelo de Sentence Transformer para vectorizar las consultas."""
    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_pinecone_index():
    """Inicializa la conexión a Pinecone y retorna el índice."""
    try:
        # Lee las claves desde los Secretos 
        # ATENCIÓN: Esta forma de leer los secretos es de Streamlit Cloud.
        # Si da error en Hugging Face, tendremos que usar os.environ (ver nota final).
        PINECONE_API_KEY = st.secrets['pinecone']['api_key']
        INDEX_NAME = st.secrets['pinecone']['index_name']
        REGION = "us-east-1" # Asumimos la región por la captura de tu índice
        
        # 🚨 Inicialización de CONTINGENCIA con la librería antigua (pinecone)
        # Esto nos permite obtener el índice, si la librería antigua arranca.
        pinecone.init(api_key=PINECONE_API_KEY, environment=REGION)
        
        return pinecone.Index(INDEX_NAME)
        
    except KeyError:
        st.error("Error de configuración: Asegúrate de que las claves 'api_key' e 'index_name' estén configuradas en los Secretos de Hugging Face bajo la sección [pinecone].")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error al conectar/inicializar Pinecone. Detalle: {e}")
        st.stop()

# Cargar el modelo y la conexión
model = get_embedding_model()
index = get_pinecone_index()

# --- 2. Interfaz Principal y Búsqueda ---
# ... (El resto del código de la interfaz es el mismo) ...
st.title("📖 Buscador Bíblico Vectorial (Semantic Search)")
st.markdown("""
Escribe una *frase, **emoción* o *concepto* (ej: *'cómo encontrar paz') y la aplicación buscará los **versos* más similares semánticamente.
""")

query = st.text_input(
    "Escribe tu consulta o sentimiento:",
    placeholder="Ej: Necesito fortaleza para afrontar un desafío difícil.",
    key="user_query"
)

top_k = st.sidebar.slider(
    "Número de versos a mostrar:",
    min_value=1, max_value=15, value=5
)

# --- 3. Lógica de Búsqueda Vectorial ---

if query:
    with st.spinner(f"Buscando los {top_k} versos más relevantes..."):
        try:
            query_vector = model.encode(query).tolist()
            
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True 
            )

            if response and response.matches:
                st.subheader(f"Resultados de búsqueda por similitud para: *'{query}'*")
                
                results_list = []
                for match in response.matches:
                    metadata = match.metadata
                    
                    # 🚨 CORRECCIÓN DE LÓGICA: Lógica de contingencia para obtener el texto completo
                    texto_del_verso = metadata.get('texto', metadata.get('verso', metadata.get('texto_completo', 'N/A')))
                    
                    results_list.append({
                        "Similitud": f"{match.score:.4f}",
                        "Libro": metadata.get('libro', 'N/A'),
                        "Capítulo": metadata.get('capitulo', 'N/A'),
                        "Verso": metadata.get('verso', 'N/A'),
                        "Texto": texto_del_verso 
                    })
                
                df_results = pd.DataFrame(results_list)
                
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True,
                    column_order=('Similitud', 'Libro', 'Capítulo', 'Verso', 'Texto') 
                )
                
                # Destacar el mejor match
                st.markdown("---")
                st.subheader("🥇 Verso Más Relevante")
                best_match = df_results.iloc[0]
                
                st.info(f"*{best_match['Texto']}\n\nReferencia:* {best_match['Libro']} {best_match['Capítulo']}:{best_match['Verso']} | Similitud: {best_match['Similitud']}")
                
            else:
                st.warning("No se encontraron versos con alta similitud para esta consulta.")
                
        except Exception as e:
            # Si aquí da un error, es el error original del texto incompleto ('verso')
            st.error(f"Ocurrió un error durante la consulta a Pinecone. Detalle: {e}")

else:
    st.info("Escribe tu consulta para empezar la búsqueda semántica en la Biblia.")

# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"Índice de Pinecone: *{st.secrets['pinecone']['index_name']}*") 
st.sidebar.markdown("Proyecto de Búsqueda Semántica Bíblica.")