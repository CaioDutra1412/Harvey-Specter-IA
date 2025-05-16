import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
import glob
import io # Importa io para lidar com bytes em memória
from PIL import Image # Importa Pillow para processamento de imagem
import pytesseract # Importa pytesseract para OCR

import pypdf # Já usado para processar PDFs da base de conhecimento
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


# --- Configuração do Tesseract OCR ---
# ** IMPORTANTE **
# Configure o caminho para o executável do Tesseract OCR no seu sistema, se necessário.
# Se você adicionou o Tesseract ao PATH do sistema e o teste funcionou (printou a versão),
# esta configuração explícita pode não ser necessária, mas é mais segura.
# Exemplo Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Exemplo Linux: r'/usr/bin/tesseract'
# Exemplo macOS (via brew): r'/usr/local/bin/tesseract'
try:
    # Tenta configurar o caminho do Tesseract usando variável de ambiente TESSERACT_PATH
    # ou tenta encontrá-lo no PATH do sistema
    tesseract_path_env = os.getenv("TESSERACT_PATH")
    if tesseract_path_env and os.path.exists(tesseract_path_env):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path_env
        print(f"[OCR Setup] Caminho do Tesseract configurado via TESSERACT_PATH: {tesseract_path_env}")
    else:
         # Se TESSERACT_PATH não estiver definida ou não existir, pytesseract tenta achar sozinho no PATH
         # Testa se o Tesseract é encontrado
         pytesseract.get_tesseract_version()
         print("[OCR Setup] Tesseract encontrado no PATH do sistema.")

except pytesseract.TesseractNotFoundError:
    print("[OCR Setup] Erro: Executável Tesseract OCR não encontrado!")
    print("Por favor, instale o Tesseract OCR no seu sistema.")
    print("Se já instalou, verifique se ele está no PATH do sistema ou configure a variável de ambiente TESSERACT_PATH no seu .env com o caminho correto.")
    # Define o caminho como None ou um valor que indique falha, para que a função OCR não seja chamada ou falhe controladamente
    pytesseract.pytesseract.tesseract_cmd = None # Define explicitamente como None se não for encontrado


# --- Configuração para o RAG ---
# Diretório onde estão os documentos da base de conhecimento
KB_DIRECTORY = "knowledge_base" # <--- Nome da pasta que você criou

# Caminho completo para o diretório da base de conhecimento
KB_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), KB_DIRECTORY)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K_CHUNKS = 5
MAX_HISTORY_TURNS = 5

# Variáveis globais inicializadas
gemini_model = None
embedding_model = None
faiss_index = None
text_chunks = []


# --- Funções para Processamento dos Documentos e OCR ---

# Função auxiliar para extrair texto de UM arquivo PDF da base de conhecimento (mantida)
def extract_text_from_pdf_kb(pdf_file_path: str) -> str:
    full_text = ""
    try:
        with open(pdf_file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            print(f"  Extraindo de KB: {os.path.basename(pdf_file_path)} ({len(reader.pages)} páginas)")
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text
    except Exception as e:
        print(f"  Erro ao extrair texto do arquivo KB {os.path.basename(pdf_file_path)}: {e}")
        return ""

# Função para carregar e processar TODOS os documentos de um diretório (adaptada para chamar extract_text_from_pdf_kb)
def load_and_process_knowledge_base(directory_path: str) -> str:
    all_text = ""
    if not os.path.exists(directory_path):
        print(f"[RAG Setup] Erro: Diretório da base de conhecimento não encontrado: {directory_path}")
        return ""

    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    if not pdf_files:
        print(f"[RAG Setup] Nenhum arquivo PDF encontrado no diretório: {directory_path}")
        return ""

    print(f"[RAG Setup] Processando {len(pdf_files)} arquivos PDF do diretório: {directory_path}")

    for pdf_file in pdf_files:
        print(f"[RAG Setup] Processando arquivo: {os.path.basename(pdf_file)}")
        # Chama a função específica para PDFs da base de conhecimento
        file_text = extract_text_from_pdf_kb(pdf_file)
        if file_text:
             all_text += file_text + "\n\n--- FIM DO DOCUMENTO ---\n\n"
        else:
             print(f"[RAG Setup] Arquivo {os.path.basename(pdf_file)} não retornou texto.")

    if not all_text.strip(): # Verifica se todo o texto combinado está vazio ou só com whitespace
         print("[RAG Setup] Nenhum texto foi extraído de todos os arquivos PDF.")
         # Retorna uma string vazia para indicar que a base de conhecimento está vazia/falhou
         return ""


    print("[RAG Setup] Extração de texto de todos os PDFs concluída.")
    return all_text


# Função para quebrar texto em Chunks (mantida)
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks = []
    if not text: # Verifica se o texto de entrada não está vazio
         return chunks
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start < 0:
            start = 0
    print(f"[RAG Setup] Texto chunkado em {len(chunks)} pedaços.")
    return chunks

# Função para processar uma imagem via OCR (Adicionada na Turn 38)
def process_image_with_ocr(image_file) -> str:
    """Extrai texto de uma imagem usando Tesseract OCR."""
    extracted_text = ""
    # Verifica se o caminho do Tesseract foi configurado ou encontrado
    if pytesseract.pytesseract.tesseract_cmd is None:
        print("[OCR Process] Tesseract OCR não encontrado. Pulando processamento de imagem.")
        return "Erro: Tesseract OCR não configurado/encontrado."

    try:
        # Pillow precisa que o arquivo seja "rewindable", então lemos os bytes
        # Rebobina o stream do arquivo para garantir que a leitura comece do início
        image_file.seek(0)
        image = Image.open(io.BytesIO(image_file.read()))
        print("[OCR Process] Imagem aberta com sucesso usando Pillow.")

        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("[OCR Process] Imagem convertida para RGB.")

        # Realiza OCR na imagem
        # Especifica o idioma para português
        extracted_text = pytesseract.image_to_string(image, lang='por')
        print(f"[OCR Process] Texto extraído da imagem (primeiros 100 chars): {extracted_text[:100]}...")
        if not extracted_text.strip():
             print("[OCR Process] OCR não extraiu texto significativo da imagem.")

    except Exception as e:
        print(f"[OCR Process] Ocorreu um erro ao processar a imagem com Pillow ou Tesseract: {e}")
        extracted_text = f"Erro ao processar a imagem: {e}"

    return extracted_text

# NOVA função para processar um arquivo PDF enviado via upload (Adicionada na Turn 51)
def process_uploaded_pdf(pdf_file) -> str:
    """Extrai texto de um arquivo PDF (FileStorage) enviado via upload."""
    extracted_text = ""
    try:
        # pypdf pode ler diretamente de um objeto tipo arquivo (como o FileStorage do Flask)
        # Rebobina o stream do arquivo para garantir que a leitura comece do início
        pdf_file.seek(0)
        reader = pypdf.PdfReader(pdf_file)
        print(f"[PDF Process] Extraindo texto de PDF anexado ({pdf_file.filename}, {len(reader.pages)} páginas)...")
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n" # Adiciona texto da página

        print("[PDF Process] Extração de texto do PDF concluída.")
        if not extracted_text.strip():
            print("[PDF Process] PDF anexado não contém texto legível.")
            return "O PDF anexado não contém texto legível." # Mensagem para o contexto se não extraiu nada

    except Exception as e:
        print(f"[PDF Process] Ocorreu um erro ao processar o PDF anexado: {e}")
        return f"Erro ao processar o PDF anexado: {e}"

    return extracted_text


# --- Configuração do Flask ---
app = Flask(__name__)
if SECRET_KEY:
    app.config['SECRET_KEY'] = SECRET_KEY
    print("Flask SECRET_KEY configurada. Sessões habilitadas.")
else:
    print("Erro: SECRET_KEY não encontrada nas variáveis de ambiente. Sessões NÃO habilitadas!")
    print("Crie uma chave secreta (ex: rode os.urandom(24).hex() no Python) e adicione SECRET_KEY=SUA_CHAVE no seu arquivo .env")


# --- Inicialização da IA e do RAG ---
print("Iniciando configuração da IA e do RAG...")

# Certifique-se que as variáveis globais são declaradas AQUI, antes do bloco try/except, e NÃO dentro
# Isso evita o SyntaxError "assigned before global declaration"
gemini_model = None
embedding_model = None
faiss_index = None
text_chunks = [] # Já inicializadas aqui

if GOOGLE_API_KEY:
    print("Chave API encontrada. Configurando a API Google AI...")
    try:
        print("Inicializando o modelo Gemini Pro...")
        # Não precisa do global aqui porque estamos atribuindo ao nome global já declarado acima
        gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
        print("Modelo Gemini Pro inicializado com sucesso!")

        # --- Configuração do RAG ---
        print("Iniciando configuração do RAG (processamento dos documentos e índice FAISS)...")

        combined_text = load_and_process_knowledge_base(KB_DIR_PATH)

        if combined_text:
            # Não precisa do global text_chunks aqui
            text_chunks = chunk_text(combined_text, CHUNK_SIZE, CHUNK_OVERLAP)

            if text_chunks:
                print("[RAG Setup] Carregando modelo de embedding...")
                # Não precisa do global embedding_model aqui
                embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                print("[RAG Setup] Modelo de embedding carregado.")

                print("[RAG Setup] Gerando embeddings para os chunks...")
                embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
                print(f"[RAG Setup] Embeddings criados. Formato: {embeddings.shape}")

                print("[RAG Setup] Criando índice FAISS...")
                # Não precisa do global faiss_index aqui
                dimension = embeddings.shape[1]
                faiss_index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
                ids = np.array(range(len(text_chunks)))
                faiss_index.add_with_ids(embeddings, ids)
                print(f"[RAG Setup] Embeddings adicionados ao índice FAISS. Total de vetores: {faiss_index.ntotal}")

                print("[RAG Setup] Configuração do RAG concluída com sucesso!")

            else:
                print("[RAG Setup] Nenhum chunk foi criado. Configuração do RAG falhou.")
                embedding_model = None
                faiss_index = None

        else:
            print("[RAG Setup] Não foi possível extrair texto de nenhum documento da pasta. Configuração do RAG falhou.")
            embedding_model = None
            faiss_index = None


    except Exception as e:
        print(f"Ocorreu um erro durante a inicialização da IA ou do RAG: {e}")
        # Garante que as variáveis globais fiquem None em caso de erro
        gemini_model = None
        embedding_model = None
        faiss_index = None
        # text_chunks já está [] do início
        print("Inicialização falhou.")

else:
     print("Erro: Chave GOOGLE_API_KEY ou SECRET_KEY não encontrada. API Google AI e Sessões NÃO serão configuradas.")
     gemini_model = None
     embedding_model = None
     faiss_index = None
     # text_chunks já está [] do início


# --- Rotas da Aplicação Web ---

@app.route('/')
def index():
    # Limpa o histórico da conversa da sessão toda vez que a página principal é acessada
    session.pop('chat_history', None)
    print("[/] Histórico da sessão limpo ao carregar a página.")
    # Verifica se todos os componentes necessários para a IA estão prontos
    ia_status = "ok" if gemini_model and embedding_model and faiss_index else "error"
    chat_history = session.get('chat_history', []) # Deve ser [] após o pop()
    return render_template('index.html', ia_status=ia_status, chat_history=chat_history)


# Rota para receber perguntas e chamar a IA (AGORA RECEBE ARQUIVOS TAMBÉM)
@app.route('/ask', methods=['POST'])
def ask_ia():
    # ** AGORA RECEBE FormData (multipart/form-data) **
    # request.form contém os campos de texto (como 'user_input') - CORRIGIDO AQUI
    # request.files contém os arquivos enviados (como 'file')

    user_question = request.form.get('user_input') # <--- CORRIGIDO AQUI para 'user_input'
    uploaded_file = request.files.get('file') # Pega o arquivo com o nome 'file' que enviamos no JS

    print(f"[/ask] Pergunta recebida: '{user_question}'")
    print(f"[/ask] Arquivo recebido (se houver): {uploaded_file.filename if uploaded_file else 'Nenhum'}")


    # ** Verifica se tem pergunta OU arquivo ANTES de processar **
    # Se não tiver texto NA PERGUNTA NEM ARQUIVO ANEXADO, retorna erro
    if not user_question and not uploaded_file:
        print("[/ask] Requisição sem pergunta ou arquivo.")
        return jsonify({"error": "Nenhuma pergunta de texto ou arquivo anexado fornecido."}), 400


    # --- Processar Arquivo Anexado (se houver) ---
    file_processing_result = "" # Variável para armazenar informações do arquivo processado (texto OCR ou texto PDF)
    if uploaded_file:
        print(f"[/ask] Tipo MIME do arquivo: {uploaded_file.content_type}")

        # Processa Imagens com OCR
        if uploaded_file.content_type.startswith('image/'):
            print("[/ask] Arquivo identificado como imagem. Processando com OCR...")
            file_processing_result = process_image_with_ocr(uploaded_file)
            if file_processing_result and not file_processing_result.startswith("Erro"): # Verifica se houve erro no OCR (mantendo msgs de erro de Tesseract)
                 file_processing_result = f"Texto extraído da imagem:\n{file_processing_result}\n"
                 print("[/ask] Texto da imagem adicionado ao contexto/prompt.")
            else:
                 # Mantém a mensagem de erro do OCR (incluindo Tesseract não encontrado)
                 file_processing_result = file_processing_result if file_processing_result else "Não foi possível extrair texto da imagem.\n"
                 print("[/ask] Falha ao extrair texto da imagem ou Tesseract não encontrado.")

        # Processa PDFs (NOVO elif)
        elif uploaded_file.content_type == 'application/pdf':
            print("[/ask] Arquivo identificado como PDF. Processando...")
            file_processing_result = process_uploaded_pdf(uploaded_file) # Chama a nova função para PDFs
            if file_processing_result and not file_processing_result.startswith("Erro"): # Verifica se não retornou um erro
                 file_processing_result = f"Conteúdo do PDF:\n{file_processing_result}\n"
                 print("[/ask] Conteúdo do PDF adicionado ao contexto/prompt.")
            else:
                 # Mantém a mensagem de erro da função process_uploaded_pdf
                 file_processing_result = file_processing_result if file_processing_result.startswith("Erro") else "Não foi possível processar o PDF.\n"
                 print("[/ask] Falha ao processar o PDF.")


        # Outros tipos de arquivo (não suportado)
        else:
            print(f"[/ask] Tipo de arquivo não suportado para processamento: {uploaded_file.content_type}")
            file_processing_result = f"Tipo de arquivo anexado ({uploaded_file.filename}, Tipo: {uploaded_file.content_type}) não suportado para processamento no momento.\n"
            # A IA receberá essa mensagem no contexto e pode avisar o usuário.


    # --- Continuar com a lógica de RAG e Gemini ---
    chat_history = session.get('chat_history', [])
    print(f"[/ask] Histórico da sessão recuperado ({len(chat_history)} turns).")

    formatted_history = ""
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        # Garante que a chave 'user' exista no dicionário antes de acessar
        user_msg = turn.get('user', '')
        ai_msg = turn.get('ai', '')
        formatted_history += f"Usuário: {user_msg}\nAssistente: {ai_msg}\n---\n"


    if gemini_model and embedding_model and faiss_index and text_chunks:
        try:
            # Gera embedding da pergunta do usuário (ainda só do texto)
            print("[/ask] Gerando embedding da pergunta de texto do usuário...")
            # Garante que user_question não é None ou vazio antes de gerar embedding
            embedding_input_text = user_question if user_question else ""
            # Se a pergunta de texto for vazia mas tiver arquivo, usar uma string padrão para o embedding da busca no RAG
            if not embedding_input_text and file_processing_result and not file_processing_result.startswith("Erro"):
                 embedding_input_text = "analise o documento" # Texto genérico para buscar contexto relevante para análise de documento


            query_embedding = embedding_model.encode([embedding_input_text], convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)

            # Busca no índice FAISS (agora baseada no texto da pergunta OU no texto genérico se só tiver arquivo)
            print(f"[/ask] Buscando no índice FAISS os {TOP_K_CHUNKS} chunks mais relevantes (baseado no texto da pergunta ou no texto do arquivo se aplicável)...")
            distances, ids = faiss_index.search(query_embedding, TOP_K_CHUNKS)

            print(f"[/ask] Recuperando texto dos chunks encontrados (IDs: {ids[0]})...")
            relevant_chunks_text = [text_chunks[id] for id in ids[0] if id != -1 and id < len(text_chunks)]

            context_for_gemini = "\n---\n".join(relevant_chunks_text)

            if not context_for_gemini.strip(): # Verifica se o contexto RAG está vazio ou só com whitespace
                 print("[/ask] Nenhum contexto relevante encontrado na busca FAISS.")
                 context_for_gemini = "Nenhum texto relevante encontrado na base de dados legal."
            else:
                 print("[/ask] Contexto relevante encontrado na busca FAISS.")


            # ** Adicionar o resultado do processamento do arquivo ao contexto para o Gemini **
            if file_processing_result: # Adiciona se houver algum resultado (texto do arquivo ou mensagem de erro/status)
                 context_for_gemini += "\n\n--- INFORMAÇÃO DO ARQUIVO ANEXADO ---\n\n" + file_processing_result
                 print("[/ask] Resultado do processamento de arquivo adicionado ao contexto do prompt.")


            # Prompt final enviado para o Gemini, incluindo histórico, RAG e informação do arquivo (se houver)
            prompt_with_rag_context = f"""
Você é um assistente especializado em simplificar processos burocráticos e legais para leigos no Brasil.
Você está participando de uma conversa com um usuário.

Histórico da Conversa (últimas {MAX_HISTORY_TURNS} turns):
---
{formatted_history if formatted_history else "Nenhum histórico anterior."}
---

Considere o seguinte CONTEXTO RELEVANTE para fundamentar sua resposta SE ele for aplicável à pergunta. Este contexto pode incluir informações da base de dados legal e/ou informações extraídas de um arquivo anexado.
---
{context_for_gemini}
---
Utilize também seu conhecimento geral para complementar a resposta ou para responder a perguntas que não são totalmente cobertas pelo contexto fornecido.
SEMPRE forneça uma resposta direta e útil. Não mencione se o contexto foi obtido de documentos legais ou de um arquivo anexado, a menos que seja estritamente necessário (ex: se nenhum contexto relevante foi encontrado E a pergunta dependia exclusivamente dele ou do arquivo, ou se houver um erro específico no processamento do arquivo que precise ser reportado ao usuário). Apenas integre a informação relevante na sua resposta de forma natural.

Pergunta do usuário: {user_question if user_question else "Analise o arquivo anexado conforme o contexto e me diga sobre o que se trata ou o que você encontrou de relevante."}
"""
            print("[/ask] Prompt aumentado com histórico, contexto RAG e informação de arquivo (se houver) montado.")
            # print(f"Prompt completo enviado para Gemini: {prompt_with_rag_context}") # Log opcional do prompt completo

            ai_response = gemini_model.generate_content(prompt_with_rag_context)
            print("[/ask] Resposta do Gemini recebida.")

            if ai_response and ai_response.text:
                resposta_texto = ai_response.text
                print("[/ask] Resposta da IA pronta para envio.")
            else:
                print("[/ask] Resposta da IA vazia ou sem texto.")
                resposta_texto = "Desculpe, não consegui gerar uma resposta para isso no momento."
                if hasattr(ai_response, 'prompt_feedback'):
                     print(f"[/ask] Feedback do Prompt da IA: {ai_response.prompt_feedback}")
                if hasattr(ai_response, 'candidates'):
                     print(f"[/ask] Candidatos da IA: {ai_response.candidates}")

            # Atualiza o histórico apenas com a pergunta original do usuário e a resposta da IA
            # Inclui a pergunta mesmo que seja vazia (no caso de envio apenas de arquivo)
            chat_history.append({'user': user_question if user_question else "Arquivo anexado para análise", 'ai': resposta_texto})
            session['chat_history'] = chat_history[-MAX_HISTORY_TURNS:]
            print(f"[/ask] Histórico da sessão atualizado. Tamanho atual: {len(session['chat_history'])} turns.")

            return jsonify({"response": resposta_texto})

        except Exception as e:
            print(f"[/ask] Ocorreu um erro durante a busca RAG, chamada da API Gemini ou atualização do histórico: {e}")
            if hasattr(e, 'response'):
                 print(f"[/ask] Detalhes da Resposta da API: {e.response}")
            # Se o erro for de algum tipo de bloqueio da API por conteúdo, a mensagem de erro pode ser diferente
            error_message = "Ocorreu um erro ao processar sua pergunta com a IA e RAG."
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 try:
                      # Tenta extrair uma mensagem de erro mais específica da resposta da API se disponível
                      api_error_details = e.response.json()
                      if "message" in api_error_details:
                           error_message = f"Erro da API: {api_error_details['message']}"
                      elif "error" in api_error_details and "message" in api_error_details["error"]:
                            error_message = f"Erro da API: {api_error_details['error']['message']}"
                 except:
                       pass # Ignora se não conseguir extrair o JSON de erro

            return jsonify({"error": error_message}), 500

    else:
        print("[/ask] Serviço de IA/RAG não disponível pois a inicialização falhou.")
        return jsonify({"error": "Serviço de assistência legal com IA não está disponível no momento. Verifique a configuração e a inicialização."}), 503

# Rota para limpar o histórico da conversa (mantida)
@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('chat_history', None)
    print("[/clear_history] Histórico da sessão limpo.")
    return jsonify({"status": "success", "message": "Histórico da conversa limpo."})

if __name__ == '__main__':
    print("\nIniciando o servidor Flask...")
    app.run(debug=True)