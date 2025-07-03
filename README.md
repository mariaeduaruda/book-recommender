## Sistema de Recomendação e Busca Semântica de Livros
# Este projeto demonstra a construção de um sistema de recomendação de livros multifacetado, utilizando técnicas de Processamento de Linguagem Natural (PLN) para analisar, classificar e recomendar livros com base em seu conteúdo e tom emocional. O sistema é capaz de entender consultas em linguagem natural para fornecer sugestões contextualmente relevantes.

# 📜 Descrição do Projeto
O objetivo deste projeto foi desenvolver uma pipeline completa de análise de dados e machine learning para um conjunto de dados de mais de 5.000 livros. O processo abrange desde a limpeza e exploração inicial dos dados até a implementação de modelos avançados para classificação de gênero, análise de sentimentos e, finalmente, a criação de um motor de busca semântica.

A pipeline é dividida nos seguintes notebooks:

data-exploration.ipynb: Focado na limpeza, tratamento de dados faltantes e engenharia de features.

text-classification.ipynb: Utiliza um modelo de IA para classificar os livros em categorias gerais (Ficção, Não Ficção, etc.).

sentiment-analysis.ipynb: Analisa a descrição dos livros para extrair um perfil emocional de cada obra.

vector-search.ipynb: Implementa um sistema de busca vetorial para encontrar livros com base no significado semântico de uma consulta.

# ✨ Funcionalidades
Limpeza de Dados: Tratamento de valores ausentes e filtragem de dados para garantir a qualidade do conjunto de dados para análise.

Engenharia de Features: Criação de novas colunas, como idade_do_livro e descricao_faltante, para enriquecer a análise.

Classificação de Gênero: Uso de um modelo de classificação "zero-shot" (facebook/bart-large-mnli) para categorizar livros sem gênero definido, alcançando uma acurácia de 81% em testes.

Análise de Sentimentos: Aplicação de um modelo de emoções (j-hartmann/emotion-english-distilroberta-base) para atribuir pontuações de raiva, alegria, medo, tristeza, etc., a cada livro com base em sua descrição.

Busca Semântica Vetorial: Implementação de um sistema de busca usando OpenAI Embeddings e um banco de dados vetorial ChromaDB, permitindo recomendações baseadas no significado de uma consulta em linguagem natural.

# ⚙️ Tecnologias e Bibliotecas
Análise de Dados: Pandas, NumPy

Visualização: Matplotlib, Seaborn

Machine Learning e PLN: Scikit-learn, Hugging Face Transformers

Busca Vetorial: LangChain, OpenAI, ChromaDB

Ambiente: Jupyter Notebook, dotenv
