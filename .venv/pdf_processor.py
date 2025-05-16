import pypdf
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Configuração ---
# Nome do arquivo PDF da Constituição Federal
# ** LEMBRE-SE DE MUDAR ESTE NOME PARA O NOME EXATO DO SEU ARQUIVO PDF **
pdf_filename = "LEIS2024.pdf"

# Caminho completo para o arquivo PDF (se não estiver na mesma pasta)
# pdf_path = os.path.join("caminho", "para", "sua", pdf_filename)
# Se estiver na mesma pasta do script:
pdf_path = pdf_filename

# Configuração de Chunking (exemplo básico)
CHUNK_SIZE = 1000  # Tamanho de cada pedaço em caracteres (ajuste se precisar)
CHUNK_OVERLAP = 200 # Quantos caracteres de um pedaço anterior se repetem no próximo (ajusta para manter contexto)

# --- Função para extrair texto do PDF ---
def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extrai todo o texto de um arquivo PDF.
    """
    full_text = ""
    try:
        with open(pdf_file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            print(f"Total de páginas no PDF: {len(reader.pages)}")
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                # print(f"Página {page_num + 1} processada.") # Opcional: descomente para ver o progresso
        print("Extração de texto concluída.")
        return full_text
    except FileNotFoundError:
        print(f"Erro: Arquivo PDF não encontrado no caminho: {pdf_file_path}")
        return ""
    except Exception as e:
        print(f"Ocorreu um erro ao ler o PDF: {e}")
        return ""

# --- Função para quebrar texto em Chunks ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Quebra um texto longo em pedaços menores (chunks).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start < 0: # Garante que o start não seja negativo
            start = 0

    print(f"Texto chunkado em {len(chunks)} pedaços.")
    return chunks

# --- Exemplo de uso (no script principal) ---
if __name__ == "__main__":
    print(f"Tentando extrair texto do PDF: {pdf_filename}")
    constitution_text = extract_text_from_pdf(pdf_path)

    if constitution_text:
        # Opcional: Adicionar uma etapa básica de limpeza aqui antes do chunking
        # Por exemplo: constitution_text = clean_text(constitution_text)

        print("\n--- Quebrando texto em chunks ---")
        text_chunks = chunk_text(constitution_text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Opcional: Mostrar exemplos dos primeiros chunks
        # print("\n--- Exemplo dos primeiros chunks ---")
        # for i, chunk in enumerate(text_chunks[:5]):
        #     print(f"--- Chunk {i+1} ---")
        #     print(chunk)
        #     print("-" * 15)

        # --- CRIAÇÃO DOS EMBEDDINGS E DO ÍNDICE FAISS ---

        print("\n--- Criando Embeddings e Índice FAISS ---")
        try:
            print("Carregando modelo de embedding...")
            # Carrega um modelo pré-treinado da sentence-transformers
            model_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("Modelo de embedding carregado.")

            print("Gerando embeddings para os chunks...")
            # Gera os embeddings para cada chunk de texto
            embeddings = model_embedding.encode(text_chunks, convert_to_numpy=True)
            print(f"Embeddings criados. Formato dos embeddings: {embeddings.shape}")

            # --- Criação do Índice FAISS ---
            # D (Dimension): A dimensão dos seus vetores (embeddings).
            dimension = embeddings.shape[1]
            print(f"Dimensão dos embeddings: {dimension}")
            print("Criando índice FAISS...")
            # IndexFlatIP é um tipo simples de índice FAISS para busca de produto interno
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

            # Adiciona os embeddings ao índice.
            ids = np.array(range(len(text_chunks))) # IDs simples de 0 a N-1
            index.add_with_ids(embeddings, ids)
            print(f"Embeddings adicionados ao índice FAISS. Total de vetores no índice: {index.ntotal}")

            # --- Armazenamento Temporário dos Chunks e Embeddings (para teste) ---
            # text_chunks contém os textos originais. O índice no FAISS
            # (os 'ids' que usamos, 0 a N-1) corresponde à posição do chunk
            # original na lista text_chunks.

            print("\n--- Índice FAISS e Embeddings criados com sucesso! ---")
            print(f"Total de chunks processados e indexados: {len(text_chunks)}")


        except Exception as e:
            print(f"Ocorreu um erro ao criar embeddings ou índice FAISS: {e}")
            print("Não foi possível criar a 'memória' buscável.")

    else:
        print("\nNão foi possível extrair texto do PDF. Verifique o nome e caminho do arquivo.")