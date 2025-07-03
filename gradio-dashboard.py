# --- 1. Importação de Bibliotecas ---
# Importa bibliotecas essenciais para manipulação de dados (pandas, numpy),
# carregamento de variáveis de ambiente (dotenv), construção da interface (gradio)
# e as funcionalidades da framework LangChain para busca semântica.
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# --- 2. Configuração Inicial ---
# Carrega as variáveis de ambiente do arquivo .env (geralmente a chave da API da OpenAI).
load_dotenv()

# --- 3. Carregamento e Preparação dos Dados dos Livros ---
# Carrega o dataset de livros a partir de um arquivo CSV.
books = pd.read_csv("books_with_categories.csv")
# Cria uma nova coluna com um link para uma imagem de capa de maior resolução.
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# Substitui capas não encontradas (NaN) por uma imagem padrão local.
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# --- 4. Preparação do Banco de Dados Vetorial (LangChain e Chroma) ---
# Carrega o arquivo de texto que contém as descrições dos livros.
raw_documents = TextLoader("tagged_description.txt", encoding='latin1').load()
# Define um divisor de texto para separar cada descrição (cada linha é um documento).
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
# Converte os documentos de texto em vetores numéricos (embeddings) usando a OpenAI
# e os armazena em um banco de dados vetorial do Chroma para busca por similaridade.
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


# --- 5. Função de Busca Semântica e Filtragem ---
# Define a função principal que busca e filtra as recomendações.
def retrieve_semantic_recommendations(
        query: str, # A busca do usuário em linguagem natural.
        category: str = None, # Filtro opcional de categoria.
        tone: str = None, # Filtro opcional de emoção/tom.
        initial_top_k: int = 50, # Número inicial de livros a buscar para garantir variedade.
        final_top_k: int = 16, # Número final de livros a exibir.
) -> pd.DataFrame:

    # 1. Realiza a busca por similaridade no banco vetorial com a consulta do usuário.
    recs = db_books.similarity_search(query, k=initial_top_k)
    # 2. Extrai os ISBNs dos livros retornados na busca.
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    # 3. Filtra o DataFrame original para obter os detalhes dos livros recomendados.
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # 4. Aplica o filtro de categoria, se um for selecionado (diferente de "All").
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # 5. Ordena os resultados com base no tom/emoção selecionado, se aplicável.
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    # 6. Retorna o DataFrame final com os livros recomendados e filtrados.
    return book_recs


# --- 6. Função de Formatação para a Interface Gráfica ---
# Prepara os dados retornados pela função de busca para serem exibidos na galeria do Gradio.
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    # Chama a função anterior para obter as recomendações.
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = [] # Lista para armazenar os resultados formatados.

    # Itera sobre cada livro recomendado para formatar sua exibição.
    for _, row in recommendations.iterrows():
        # Trunca a descrição do livro para 30 palavras.
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Formata a lista de autores para uma leitura mais natural (e.g., "A, B e C").
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Cria a legenda que aparecerá abaixo da capa do livro.
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        # Adiciona a tupla (imagem_da_capa, legenda) à lista de resultados.
        results.append((row["large_thumbnail"], caption))
    return results

# --- 7. Construção da Interface com Gradio ---
# Prepara as listas de opções para os menus dropdown.
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Inicia a construção da interface gráfica (dashboard).
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    # Título do dashboard.
    gr.Markdown("# Recomendador semântico de Livros")

    # Linha com os campos de entrada do usuário.
    with gr.Row():
        user_query = gr.Textbox(label = "Descreva o que procura:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Selecione uma categoria:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Selecione uma emoção:", value = "All")
        submit_button = gr.Button("Achar recomendações")

    # Seção de saída para os resultados.
    gr.Markdown("## Recomendações")
    # Galeria para exibir as capas dos livros recomendados.
    output = gr.Gallery(label = "Livros recomendados", columns = 8, rows = 2)

    # Conecta o botão de "submit" à função que executa a busca e formata os resultados.
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


# --- 8. Execução do Dashboard ---
# Verifica se o script está sendo executado diretamente e, em caso afirmativo, inicia o servidor do Gradio.
if __name__ == "__main__":
    dashboard.launch()