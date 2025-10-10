import streamlit as st
import pandas as pd
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import random

# --- 0. Configuración de la Página ---
st.set_page_config(
    page_title="Buscador Bíblico Vectorial",
    page_icon="📖",
    layout="wide"
)

# --- 1. CONEXIÓN A PINECONE Y MODELO (USANDO st.secrets) ---

@st.cache_resource # Asegura que el modelo solo se cargue una vez (MUY IMPORTANTE)
def get_embedding_model():
    """Carga el modelo de Sentence Transformer para vectorizar las consultas."""
    # Debe ser el mismo modelo usado en 'vectorizar_cargar.py'
    MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource # Asegura que la conexión a Pinecone solo se haga una vez
def get_pinecone_index():
    """Inicializa la conexión a Pinecone."""
    try:
        # Lee las claves desde los secretos de Streamlit Cloud (o secrets.toml)
        PINECONE_API_KEY = st.secrets['pinecone']['api_key']
        PINECONE_ENVIRONMENT = st.secrets['pinecone']['environment']
        INDEX_NAME = st.secrets['pinecone']['index_name']
        
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        return pc.Index(INDEX_NAME)
        
    except KeyError:
        st.error("Error: Las claves de Pinecone no están configuradas en secrets.toml o en la sección 'Secretos' de Streamlit Cloud.")
        st.stop()
    except Exception as e:
        st.error(f"Error al conectar con Pinecone. Revisa tus claves y el nombre del índice. Detalle: {e}")
        st.stop()

# Cargar el modelo y la conexión al inicio
model = get_embedding_model()
index = get_pinecone_index()

# --- 2. Interfaz Principal y Búsqueda ---
st.title("📖 Buscador Bíblico Vectorial (Semantic Search)")
st.markdown("""
Escribe una frase o una emoción (ej: *'un versículo de consuelo en la desesperación') y la aplicación buscará los versículos de la Biblia que sean **semánticamente* más similares, utilizando la base de datos vectorial de *Pinecone*.
""")

# Campo de entrada de usuario
query = st.text_input(
    "Escribe tu consulta o sentimiento:",
    placeholder="Ej: Necesito fortaleza para afrontar un desafío difícil.",
    key="user_query"
)

# Número de resultados a mostrar
top_k = st.sidebar.slider(
    "Número de resultados a mostrar:",
    min_value=1, max_value=20, value=5
)

# --- 3. Lógica de Búsqueda Vectorial ---

if query:
    with st.spinner(f"Buscando los {top_k} versículos más relevantes..."):
        try:
            # 1. Vectorizar la consulta del usuario
            query_vector = model.encode(query).tolist()
            
            # 2. Consultar el índice de Pinecone
            response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True # Pedimos que nos devuelva el texto y la metadata
            )

            # 3. Procesar los resultados y presentarlos
            if response and response.matches:
                st.subheader(f"Resultados de búsqueda por similitud para: *'{query}'*")
                
                # Crear un DataFrame para la presentación
                results_list = []
                for match in response.matches:
                    metadata = match.metadata
                    
                    results_list.append({
                        "Score de Similitud": f"{match.score:.4f}",
                        "Libro": metadata.get('libro', 'N/A'),
                        "Sentimiento Clasificado": metadata.get('sentimiento', 'N/A'),
                        "Versículo": metadata.get('texto', 'N/A')
                    })
                
                df_results = pd.DataFrame(results_list)
                
                # Mostrar los resultados
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Opcional: Mostrar el versículo con el mayor score
                st.markdown("---")
                st.subheader("🥇 Versículo Más Relevante")
                best_match = df_results.iloc[0]
                st.code(f"Similitud: {best_match['Score de Similitud']} | Libro: {best_match['Libro']} | Sentimiento: {best_match['Sentimiento Clasificado']}")
                st.success(f"*{best_match['Versículo']}*")
                
            else:
                st.warning("No se encontraron resultados para esta consulta en el índice.")
                
        except Exception as e:
            st.error(f"Ocurrió un error durante la consulta a Pinecone. Detalle: {e}")

else:
    st.info("Escribe tu consulta arriba para empezar a buscar.")

# --- Pie de página ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"Índice de Pinecone: *{index.name}*")
st.sidebar.markdown("Proyecto de Búsqueda Semántica Bíblica.")