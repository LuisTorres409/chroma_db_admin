# ChromaDB Admin

## Usabilidade

O repositório possui dois arquivos principais, um notebook chamado `populate_db.ipynb`, esse notebook contém instruções de como carregar
documentos de uma pasta para iniciar seu banco de dados, nele são apresentados dois métodos de separação dos dados.

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