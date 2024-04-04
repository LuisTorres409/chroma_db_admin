import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import json
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter
import os
import chromadb.utils.embedding_functions as embedding_functions
import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from llama_index.core import Document, Settings
import json
import shutil
import plotly.express as px
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extrair_filename(metadata):
    node_content = json.loads(metadata['_node_content'])
    metadata = node_content['metadata']
    return metadata['file_name']


#Função que percorre o JSON da metadata em busca da chave 'file_name', pois a metadata está formatada de forma diferente para alguns arquivos
def extrair_filename_v2(metadata):
    json_data = metadata
    # Verifica se o JSON é um dicionário
    if isinstance(json_data, dict):
        # Verifica se a chave 'file_name' está presente no dicionário
        if 'file_name' in json_data:
            return json_data['file_name']
        else:
            # Itera sobre os valores do dicionário em busca da chave 'file_name'
            for value in json_data.values():
                result = extrair_filename_v2(value)
                if result is not None:
                    return result
    # Se não encontrar a chave 'file_name', retorna None
    return None


def criar_nova_collection(_db, collection_name, embedding_name, distance_form,documents):

    if documents == []:
        return False, 'Nenhum documento foi enviado, insira pelo menos um documento para criar a collection'

    distance_conversion_dict = {'Cosine Similarity':'cosine','Inner product':'ip','Squared L2':'l2'}

    try:
        
        #Implementar outros embeddings além do embedding default, não consegui fazer funcionar de primeira.

        #if embedding_name.startswith('text-embedding'):
        #    embed_model = embedding_functions.OpenAIEmbeddingFunction(
        #        api_key = os.getenv('OPENAI_API_KEY'),
        #        model_name=embedding_name
        #    )
        #else:
        #    embed_model = embedding_functions.HuggingFaceEmbeddingFunction(
        #    api_key=os.getenv('HUGGINGFACE_API_KEY'),
        #    model_name=f"sentence-transformers/{embedding_name}"
        #    ) 

        collection = _db.create_collection(name=collection_name,metadata = {'hnsw:space':distance_conversion_dict[distance_form]})
        st.session_state.collections[collection_name] = collection

        status,error_message = process_documents(documents,collection_name)
        
        if status:
            return True, None
        else:
            _db.delete_collection(collection_name)
            st.session_state.collections.pop(collection_name)
            return False, error_message
        
    except Exception as e:
        try:
            _db.delete_collection(collection_name)
            st.session_state.collections.pop(collection_name)
        except:
            print('Erro foi antes ou durante a criação da collection')
        return False,f'{str(e)}'

def get_collection_data(_collection):
    collection_data = _collection.get(
        include=["documents", "embeddings", "metadatas"]
    )

    return pd.DataFrame(data=collection_data)

def calcular_media(vetores):
    # Convertendo para um array numpy para facilitar os cálculos
    vetores_array = np.array(vetores)
    # Calculando a média ao longo do eixo 0 (média de todos os vetores)
    media = np.mean(vetores_array, axis=0)
    return media

def calcular_distancia(vetor_1,vetor_2):
    return np.linalg.norm(vetor_1-vetor_2)

def calcular_distancia_media(vetores,centroide):

    return sum([calcular_distancia(vetor,centroide) for vetor in vetores])/len(vetores)

def generate_sphere(center, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z
    

@st.cache_resource
def get_database_connection():
    db = chromadb.PersistentClient(path="./chroma_db")
    embed_model_name = 'all-MiniLM-L6-v2'


    embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))  
    
    return db,embed_model

def get_collection_names():
    
    collections = db.list_collections()

    names= [collection.name for collection in collections]

    return names

def reduce_dimensions(vectors):
    pca = PCA(n_components=3)
    vetores_reduzidos = pca.fit_transform(vectors)
    return vetores_reduzidos

def process_documents(docs,collection_name):
    text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
    )
    for pdf in docs:
        try:
            caminho_arquivo = f'chroma_db/{collection_name}_files'
            pdf_read = PdfReader(pdf)
            for page in pdf_read.pages:
                text = page.extract_text()
                text_chunks = text_parser.split_text(text)
                for chunk in text_chunks:
                    chunk_document = Document(
                        text=chunk,
                        metadata={'file_name':pdf.name,'page':pdf_read.get_page_number(page)}
                    )
                    st.session_state.collections[collection_name].add(documents=[chunk],ids=[chunk_document.doc_id],metadatas=[chunk_document.metadata])
            if not os.path.exists(caminho_arquivo):
                os.makedirs(caminho_arquivo)
            with open(os.path.join(caminho_arquivo,pdf.name), "wb") as f:
                f.write(pdf.getbuffer())   

            return True, None

        except Exception as e:
            try:
                shutil.rmtree(f'chroma_db/{collection_name}_files')
            except:
                print('Pasta não tinha sido criada ainda')
            return False, str(e)
        

st.set_page_config(page_title="ChromaDB Admin", layout="wide", initial_sidebar_state="expanded", menu_items=None)
db ,embed_model= get_database_connection()



if 'collections' not in st.session_state.keys():
    collections = db.list_collections()
    st.session_state.collections = {collection.name : collection for collection in collections}

if 'collections_docs' not in st.session_state.keys():
    st.session_state.collections_docs = {}#if 'selected_docs' not in st.session_state.keys():    #st.session_state.selected_docs = []

with st.sidebar:

    st.title('ChromaDB Admin')

    new_collection_column , delete_colletion_column = st.columns(2)

    with new_collection_column:
        with st.popover('Criar nova collection'):
            new_collection_name = st.text_input('Nome da collection')
            new_collection_embedding_name = st.radio('Tipo de embedding', ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'text-embedding-3-large'])
            new_collection_distance_form = st.radio('Calculo de distancia', ['Cosine Similarity', 'Inner product','Squared L2'])
            new_collection_documents = st.file_uploader(label='Documentos para colocar incialmente no bano',accept_multiple_files=True,help="Insira arquivos PDF")
            if st.button('Criar collection'):
                with st.spinner('Criando collection'):
                    create_status,create_error = criar_nova_collection(db,new_collection_name,new_collection_embedding_name,new_collection_distance_form,new_collection_documents)
                    if create_status:
                        st.success('Collection criada com sucesso!')
                    else:
                        st.error(f'Erro ao criar collection {create_error}')
                    #st.rerun()
    
    with delete_colletion_column:
        with st.popover('Deletar collection'):
            delete_collection_name = st.selectbox('Collections',st.session_state.collections.keys(),key='delete_collection')
            if st.button('Deletar collection'):
                with st.spinner('Deletando collection'):
                    try:
                        db.delete_collection(delete_collection_name)
                        shutil.rmtree(f'chroma_db/{delete_collection_name}_files')
                        st.session_state.collections.pop(delete_collection_name)
                        st.success('Collection deletada com sucesso!')
                        st.rerun()
                    except Exception as e:
                        st.error(f'Erro ao deletar collection {str(e)}')
        
    st.header('Selecione a collection')
    collection_name = st.selectbox('Collections',st.session_state.collections.keys(),key='select_collection')
    st.header('Enviar Documentos')
    docs = st.file_uploader(label = "Faça upload dos documentos para o banco de dados", accept_multiple_files=True,help="Insira aquivos PDF")
    if st.button("Processar documentos"):
        if docs == []:
            st.error("Nenhum documento foi enviado")
        else:
            with st.spinner("Processando"):
                status,error_message = process_documents(docs,collection_name)
                if status:
                    st.success("Documentos adicionados ao banco de dados com sucesso!")
                else:
                    st.error(error_message)


documents_collection_df =  get_collection_data(st.session_state.collections[collection_name])

col1, col2 = st.columns(2)

with col1:
    st.title('**Collection Data**')


    documents_collection_df['filename'] = documents_collection_df['metadatas'].apply(lambda x: extrair_filename_v2(x))
    documents_collection_df = documents_collection_df.rename(columns={'documents':'Conteúdo','embeddings':'Embeddings','filename':'Nome do Arquivo'})
    #st.write(st.session_state.selected_docs)
    #if st.session_state.selected_docs != []:
        #documents_collection_df = documents_collection_df[documents_collection_df['Nome do Arquivo'].isin(st.session_state.selected_docs)]
    #collection_df = st.dataframe(documents_collection_df[['Nome do Arquivo','Embeddings','Conteúdo']])
    edited_df = st.data_editor(documents_collection_df[['Nome do Arquivo','Embeddings','Conteúdo']], num_rows="dynamic")
    vetores_reduzidos = reduce_dimensions(documents_collection_df['Embeddings'].tolist())
    df_vectors = pd.DataFrame(vetores_reduzidos, columns=[0,1,2])
    df_vectors['Nome do Arquivo'] = documents_collection_df['Nome do Arquivo']


with col2:
    _ , col_metrics , _ = st.columns(3)
    with col_metrics:
        st.title('**Métricas**')
        st.metric('Embedding Size', len(edited_df['Embeddings'].iloc[0]))
        st.metric('Número de Vetores', len(edited_df))
        lista_arquivos = edited_df['Nome do Arquivo'].unique()
        st.metric('Número de Documentos',len(lista_arquivos))
        st.session_state.collections_docs[collection_name] = lista_arquivos

#with st.sidebar:
    #st.header('Selecione documentos do banco de dados')
    #selected_docs = st.multiselect('Selecione os documentos', st.session_state.collections_docs[collection_name])
    #st.session_state.selected_docs = selected_docs
    #st.write(selected_docs)

col_graph_left, col_graph_right = st.columns(2)

with col_graph_left:

    st.title('**Centroide dos vetores**')

    df_vectors_mean = df_vectors.groupby('Nome do Arquivo').agg({
    0: lambda x: calcular_media(x),
    1: lambda x: calcular_media(x),
    2: lambda x: calcular_media(x)
    }).reset_index()
    df_vectors_mean['Centroides'] = df_vectors_mean[[0,1,2]].apply(lambda x: x.values.tolist(),axis=1)
    df_vectors_radius = df_vectors_mean.apply(lambda x: calcular_distancia_media(df_vectors[df_vectors['Nome do Arquivo'] == x['Nome do Arquivo']][[0,1,2]].values,x['Centroides']),axis=1)
    df_vectors_mean['Raio médio'] = df_vectors_radius
    df_vectors_mean['Raio normalizado'] = df_vectors_mean['Raio médio'] / df_vectors_mean['Raio médio'].max()
    df_raios = pd.DataFrame({
    'Nome do Arquivo': df_vectors_mean['Nome do Arquivo'],
    'Raio': df_vectors_mean['Raio médio'],
    'Raio normalizado': df_vectors_mean['Raio normalizado']
    })

    graph = px.scatter_3d(df_vectors_mean, x=0, y=1, z=2, color='Nome do Arquivo')
    st.plotly_chart(graph)

with col_graph_right:

    st.title('**Vetores**')

    graph = px.scatter_3d(df_vectors, x=0, y=1, z=2, color='Nome do Arquivo')
    st.plotly_chart(graph)

_,col_esfera_title,_ = st.columns(3)

with col_esfera_title:
    st.title('**Esferas de raio médio**')


fig = px.scatter_3d(df_vectors_mean, x=0, y=1, z=2, opacity=0.5)

for i in range(len(df_vectors_mean)):
    center = df_vectors_mean[['Centroides']].iloc[i][0]
    radius = df_vectors_mean['Raio médio'].iloc[i]
    x, y, z = generate_sphere(center, radius)
    fig.add_trace(go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), mode='lines', 
                                name=f'Esfera {df_vectors_mean["Nome do Arquivo"].iloc[i]}'))
    
fig.update_traces(marker=dict(size=2))
fig.update_layout(scene=dict(aspectmode="cube"))

col_sphere_graph , col_radius_dataframe = st.columns(2)

with col_sphere_graph:
    st.plotly_chart(fig)

with col_radius_dataframe:
    st.dataframe(df_raios)

