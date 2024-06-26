# ChromaDB Admin

## Estrutura dos arquivos

O repositório possui dois arquivos principais, um notebook chamado `populate_db.ipynb`, e outro chamado `chroma_admin.py` o notebook contém instruções de como carregar
documentos de uma pasta para iniciar seu banco de dados, enquanto o outro é um aplicativo streamlit que contém toda a interface do admin.

## Populando o banco

No arquivo `populate_db.ipynb` , é mostrada a possibilidade de dois métodos (existem muitos outros) de pré-processar os dados que irão ser salvos no banco.

- Sentence Splitter

Essa é a forma mais básica de particionar o texto dos seus documentos para posteriormente fazer o embedding, ela particionando o documento
baseado no número de caracteres. Existe também o atributo chunk_overlap que permite que blocos de texto tenham partes em comum para aumentar
a contextualização e evitar perda semântica e fragmentos de frase. Esse método é a forma mais rápida de pré-processar os dados.

- Semantic Chunking

Esse método consiste em agrupar fragmentos de texto que são semanticamente semelhantes, dessa forma, durante o pré-processamento dos dados
é feito o embedding dos fragmentos de texto , o que gera um aumento significativo de tempo para processar. Em geral, como foi visto nos
documentos de exemplo presentes do banco, a utilização de semantic chunking aumentou o agrupamento e diferenciação de arquivos distintos,
o que caracteriza uma melhora no desempenho da organização dos dados.

Para popular o banco foi utilizado o [Llama-Index](https://www.llamaindex.ai/) como agente integrador, que disponibiliza varias ferramentas que facilitam e impulsionam
esse tipo de tarefa.

Além disso, após enviar os documentos para o banco vetorial, é feito uma cópia dos documentos para uma pasta dentro da pasta `chroma_db`, com o nome `{collection_name}_files`.


## Chroma Admin

Já o arquivo `chroma_admin.py` , é um aplicativo [Streamlit](https://streamlit.io/), que vai fazer toda a conexão e manejo do banco. Nele utilizamos a biblioteca padrão do
ChromaDB para estabelecar as conexões e fazer as modificações. Também é possivel enviar documentos para um banco através da aba lateral do App, criar novas collections e 
excluir collections existentes. Para mostrar os dados de maneira gráfica, foram utilizadas técnicas a fim de melhorar o entendimento sobre a base de dados e a performance dos
modelos de embedding e pré-processamento.

- Decomposição PCA

Essa estratégia permite reduzir a dimensionalidade de um vetor porém preservando informação suficiente para podermos analisar suas componentes principais e extrair as informações mais importantes. Dessa forma , um vetor de 384 dimensões como é o caso do embedding `all-MiniLM-L6-v2` , pode ser reduzido para três dimensões de forma
que suas componentes principais, ou seja, as que carregam maior valor semântico , sejam preservadas em 3 dimensões, o que permite vizualizar os clusters de cada documento
de forma espacial.

- Densidade dos clusters

Após a decomposição PCA de cada vetor do banco, é possível plotar os vetores num espaço 3D de forma que formem uma nuvem de pontos para cada arquivo processado no embedding.
Dessa forma, é possivel calcular a densidade de distribuição dos pontos , isto é, a distancia média dos pontos ao centroide, que chamaremos de raio médio.

A fórmula para o raio médio é dada pela média da distância euclidiana de cada vetor ao centroide da coleção,
dessa forma, seja $C$ o centroide da coleção , e $V_i$ o vetor $i$ da coleção,o centroide é dado por:

$$
C = \frac{\sum\limits_{i=1}^{n} V_i}{n}
$$

Dessa forma, podemos calcular o raio médio da coleção como:

$$
R_{medio}=\frac{\sum\limits_{i=1}^{n} ||C-V_i||}{n}
$$

Dessa forma, quanto menor o raio médio, mais denso é o cluster semântico dos vetores, mostrando que os fragmentos dos documentos são mais similares entre si, ou
que a separação de embedding está bem feita.

