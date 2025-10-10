import streamlit as st
import pandas as pd
# 🚨 CORRECCIÓN 1: Usamos pinecone_client para evitar el ModuleNotFoundError fatal en Streamlit
from pinecone_client import Pinecone 
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
        # Lee la clave de API desde los secretos de Streamlit Cloud
        PINECONE_API_KEY = st.secrets['pinecone']['api_key']
        INDEX_NAME = st.secrets['pinecone']['index_name']
        
        pc = Pinecone(api_key=PINECONE_API_KEY) 
        
        return pc.Index(INDEX_NAME)
        
    except KeyError:
        st.error("Error de configuración: Asegúrate de que las claves 'api_key' e 'index_name' estén configuradas en la sección 'Secretos' de Streamlit Cloud, bajo la sección [pinecone].")
        st.stop()
    except Exception as e:
        # Aseguramos que la aplicación se detenga si la conexión a Pinecone falla
        st.error(f"Error al conectar con Pinecone. Revisa tus claves y el nombre del índice. Detalle: {e}")
        st.stop()

# Cargar el modelo y la conexión
model = get_embedding_model()
index = get_pinecone_index()

# --- 2. Interfaz Principal y Búsqueda ---
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
                    
                    # 🚨 CORRECCIÓN 2 (Contingencia): Intentamos todas las posibles claves para el texto
                    # Si no se pudo sobrescribir, el texto está en 'texto', 'verso' o en la clave corregida 'texto_completo'
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
                
                # Muestra el texto completo y la referencia. Esto evita el error 'verso' en la tarjeta.
                st.info(f"*{best_match['Texto']}\n\nReferencia:* {best_match['Libro']} {best_match['Capítulo']}:{best_match['Verso']} | Similitud: {best_match['Similitud']}")
                
            else:
                st.warning("No se encontraron versos con alta similitud para esta consulta.")
                
        except Exception as e:
            # Ahora este error solo debería saltar si falla la consulta Query, no por las claves internas
            st.error(f"Ocurrió un error durante la consulta a Pinecone. Detalle: {e}")

else:
    st.info("Escribe tu consulta para empezar la búsqueda semántica en la Biblia.")

# --- Pie de página ---
st.sidebar.markdown("---")
# Usamos el nombre del índice de los secretos
st.sidebar.markdown(f"Índice de Pinecone: *{st.secrets['pinecone']['index_name']}*") 
st.sidebar.markdown("Proyecto de Búsqueda Semántica Bíblica.")