import os
import glob
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pypdf # Necessário para carregar PDFs da base de conhecimento

class RAGSystem:
    def __init__(self, kb_directory, chunk_size, chunk_overlap, embedding_model_name='paraphrase-MiniLM-L6-v2'):
        self.kb_directory = kb_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

        self.embedding_model = None
        self.faiss_index = None
        self.text_chunks = []

        print("\n[RAG System] Iniciando configuração do RAG...")
        self._initialize_rag()
        print("[RAG System] Configuração do RAG finalizada.")

    def _extract_text_from_pdf_kb(self, pdf_file_path: str) -> str:
        full_text = ""
        try:
            with open(pdf_file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                # print(f"  Extraindo de KB: {os.path.basename(pdf_file_path)} ({len(reader.pages)} páginas)") # Comentado para logs mais limpos
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
            return full_text
        except Exception as e:
            print(f"  [RAG System] Erro ao extrair texto do arquivo KB {os.path.basename(pdf_file_path)}: {e}")
            return ""

    def _load_and_process_knowledge_base(self) -> str:
        all_text = ""
        # ** CORREÇÃO AQUI: Usa o diretório onde rag.py está como base, que é a raiz do projeto **
        kb_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.kb_directory)
        # A linha abaixo estava incorreta para a sua estrutura, subindo um nível a mais:
        # kb_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), self.kb_directory)


        if not os.path.exists(kb_dir_path):
            print(f"[RAG System] Erro: Diretório da base de conhecimento não encontrado: {kb_dir_path}")
            return ""

        pdf_files = glob.glob(os.path.join(kb_dir_path, "*.pdf"))

        if not pdf_files:
            print(f"[RAG System] Nenhum arquivo PDF encontrado no diretório: {kb_dir_path}")
            return ""

        print(f"[RAG System] Processando {len(pdf_files)} arquivos PDF do diretório: {kb_dir_path}")

        for pdf_file in pdf_files:
            file_text = self._extract_text_from_pdf_kb(pdf_file)
            if file_text:
                 all_text += file_text + "\n\n--- FIM DO DOCUMENTO ---\n\n"

        if not all_text.strip():
             print("[RAG System] Nenhum texto foi extraído de todos os arquivos PDF.")
             return ""

        print("[RAG System] Extração de texto de todos os PDFs concluída.")
        return all_text

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        if not text:
             return chunks
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            if start < 0:
                start = 0
        print(f"[RAG System] Texto chunkado em {len(chunks)} pedaços.")
        return chunks

    def _initialize_rag(self):
        try:
            combined_text = self._load_and_process_knowledge_base()

            if combined_text:
                self.text_chunks = self._chunk_text(combined_text)

                if self.text_chunks:
                    print(f"[RAG System] Carregando modelo de embedding: {self.embedding_model_name}...")
                    self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    print("[RAG System] Modelo de embedding carregado.")

                    print("[RAG System] Gerando embeddings para os chunks...")
                    batch_size = 32
                    embeddings = []
                    for i in range(0, len(self.text_chunks), batch_size):
                        batch_chunks = self.text_chunks[i:i + batch_size]
                        batch_embeddings = self.embedding_model.encode(batch_chunks, convert_to_numpy=True)
                        embeddings.append(batch_embeddings)
                    embeddings = np.vstack(embeddings)

                    print(f"[RAG System] Embeddings criados. Formato: {embeddings.shape}")

                    print("[RAG System] Criando índice FAISS...")
                    dimension = embeddings.shape[1]
                    self.faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
                    ids = np.array(range(len(self.text_chunks)))
                    self.faiss_index.add_with_ids(embeddings, ids)
                    print(f"[RAG System] Embeddings adicionados ao índice FAISS. Total de vetores: {self.faiss_index.ntotal}")

                    print("[RAG System] Configuração do RAG concluída com sucesso!")

                else:
                    print("[RAG System] Nenhum chunk foi criado. Configuração do RAG falhou.")
                    self.embedding_model = None
                    self.faiss_index = None

            else:
                print("[RAG System] Não foi possível extrair texto de nenhum documento da pasta. Configuração do RAG falhou.")
                self.embedding_model = None
                self.faiss_index = None

        except Exception as e:
            print(f"[RAG System] Ocorreu um erro durante a inicialização do RAG: {e}")
            self.embedding_model = None
            self.faiss_index = None
            self.text_chunks = []
            print("[RAG System] Inicialização do RAG falhou.")

    def is_ready(self, check_embeddings=True):
        """Verifica se o RAG está pronto, opcionalmente checando se há embeddings."""
        if check_embeddings:
             return self.embedding_model is not None and self.faiss_index is not None and self.text_chunks is not None and len(self.text_chunks) > 0 and self.faiss_index.ntotal > 0
        else:
             # Verifica apenas se o modelo de embedding e o índice existem (pode ser útil mesmo com base vazia para outras buscas)
             return self.embedding_model is not None and self.faiss_index is not None


    # Retorna os chunks de texto baseados em IDs
    def get_chunks_by_ids(self, ids: list[int]) -> list[str]:
         if not self.text_chunks:
              return []
         return [self.text_chunks[id] for id in ids if id >= 0 and id < len(self.text_chunks)]


    # Busca os k chunks mais relevantes na base de conhecimento usando embedding(s)
    def search_chunks_with_embeddings(self, query_embeddings: np.ndarray, k: int) -> list[int]:
        """
        Busca os k chunks mais relevantes na base de conhecimento para cada embedding na lista.
        Retorna uma lista de IDs únicos encontrados.
        """
        if not self.is_ready(check_embeddings=True):
            print("[RAG System] RAG não inicializado/pronto para busca. Pulando busca FAISS.")
            return []
        if query_embeddings is None or query_embeddings.shape[0] == 0:
             print("[RAG System] Embeddings de busca vazios. Pulando busca FAISS.")
             return []

        try:
            print(f"[RAG System] Buscando no índice FAISS com {query_embeddings.shape[0]} embedding(s) (top {k})...")
            # search retorna distancias e ids. ids[i][j] é o j-ésimo vizinho mais próximo da i-ésima query
            distances, ids = self.faiss_index.search(query_embeddings, k)

            relevant_chunk_ids = set()
            # Percorre os resultados de todas as queries e adiciona IDs válidos ao set
            for i in range(ids.shape[0]):
                 valid_ids = ids[i][ids[i] != -1] # Remove IDs inválidos (-1) para a i-ésima query
                 relevant_chunk_ids.update(valid_ids)


            print(f"[RAG System] Encontrado {len(relevant_chunk_ids)} IDs únicos relevantes na busca FAISS.")
            return list(relevant_chunk_ids) # Retorna lista de IDs únicos

        except Exception as e:
            print(f"[RAG System] Erro durante a busca FAISS com embeddings: {e}")
            return []

    # Método para gerar embedding para um texto (útil para embeddings de query e arquivo)
    def generate_embedding(self, text: str) -> np.ndarray | None:
        if not self.embedding_model:
            print("[RAG System] Modelo de embedding não carregado. Não é possível gerar embedding.")
            return None
        if not text or not text.strip():
             print("[RAG System] Texto vazio para gerar embedding.")
             return None
        try:
            print("[RAG System] Gerando embedding para texto fornecido...")
            # encode espera uma lista de strings
            embedding = self.embedding_model.encode([text], convert_to_numpy=True).reshape(1, -1)
            print("[RAG System] Embedding gerado.")
            return embedding
        except Exception as e:
             print(f"[RAG System] Erro ao gerar embedding: {e}")
             return None