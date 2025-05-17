import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv

# Importa a classe e as funções dos novos módulos
from rag import RAGSystem
from processing import process_image_with_ocr, process_uploaded_pdf, search_text_for_query


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


# --- Configuração do RAG e IA ---
KB_DIRECTORY = "knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_CHUNKS = 5 # Quantidade de chunks da BASE DE CONHECIMENTO a buscar por fonte (texto OU arquivo atual)
COMBINED_TOP_K_CHUNKS = 7 # Quantidade total de chunks da BASE DE CONHECIMENTO a considerar após combinar fontes
MAX_LAST_FILE_CONTENT_SIZE = 15000 # Limite para texto de arquivo na sessão

MAX_HISTORY_TURNS = 5 # Número de turnos de chat de texto a considerar no histórico


# Inicializa o sistema RAG e o modelo Gemini
# A inicialização do Tesseract agora está dentro do módulo processing.py
rag_system = None
gemini_model = None

if GOOGLE_API_KEY:
    print("Chave API encontrada. Configurando a API Google AI...")
    try:
        print("Inicializando o modelo Gemini Pro...")
        gemini_model = genai.GenerativeModel('models/gemini-2.0-flash')
        print("Modelo Gemini Pro inicializado com sucesso!")

        # Inicializa o sistema RAG
        rag_system = RAGSystem(
            kb_directory=KB_DIRECTORY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Verifica se o RAG inicializou corretamente
        if not rag_system.is_ready():
             print("Erro: Sistema RAG não inicializado corretamente.")
             rag_system = None # Define como None para indicar falha

    except Exception as e:
        print(f"Ocorreu um erro durante a inicialização da IA ou do RAG: {e}")
        gemini_model = None
        rag_system = None
        print("Inicialização falhou.")

else:
     print("Erro: Chave GOOGLE_API_KEY ou SECRET_KEY não encontrada. API Google AI e RAG NÃO serão configurados.")
     gemini_model = None
     rag_system = None


# --- Configuração do Flask ---
app = Flask(__name__)
if SECRET_KEY:
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['PERMANENT_SESSION_LIFETIME'] = 3600
    app.permanent_session_lifetime = app.config['PERMANENT_SESSION_LIFETIME']
    print(f"Flask SECRET_KEY configurada. Sessões habilitadas com tempo de vida de {app.config['PERMANENT_SESSION_LIFETIME']}s.")
else:
    print("Erro: SECRET_KEY não encontrada nas variáveis de ambiente. Sessões NÃO habilitadas!")


# --- Rotas da Aplicação Web ---

@app.route('/')
def index():
    # Limpa o histórico de chat E o histórico de arquivo na sessão ao carregar a página principal
    session.pop('chat_history', None)
    session.pop('last_file_content', None) # Limpa o conteúdo do último arquivo
    session.pop('last_file_name', None) # Limpa o nome do último arquivo
    print("[/] Histórico da sessão (chat e arquivo) limpo ao carregar a página.")

    # O status da IA agora depende se o modelo Gemini e o RAG inicializaram
    ia_status = "ok" if gemini_model and (rag_system is None or rag_system.is_ready()) else "error"
    # O RAG pode não estar pronto se não houver PDFs, mas a IA ainda pode responder perguntas gerais.
    # Talvez o status deva ser "rag_error" ou "ai_error" para ser mais específico.
    # Por enquanto, 'ok' se a IA estiver pronta, mesmo que o RAG falhe.
    # Ajustando: Status é 'ok' se Gemini estiver pronto. Se RAG falhar, informamos no log.
    ia_status = "ok" if gemini_model else "error"


    chat_history = session.get('chat_history', [])
    return render_template('index.html', ia_status=ia_status, chat_history=chat_history)


@app.route('/ask', methods=['POST'])
def ask_ia():
    user_question = request.form.get('user_input')
    uploaded_file = request.files.get('file')

    print(f"[/ask] Pergunta recebida: '{user_question}'")
    print(f"[/ask] Arquivo recebido (se houver): {uploaded_file.filename if uploaded_file else 'Nenhum'}")

    # Recupera o conteúdo e nome do último arquivo processado da sessão, se existirem
    last_file_content_from_session = session.get('last_file_content', '')
    last_file_name_from_session = session.get('last_file_name', 'arquivo anterior')
    print(f"[/ask] Conteúdo do último arquivo da sessão recuperado ('{last_file_name_from_session}', tamanho: {len(last_file_content_from_session)} chars).")


    # --- Processar Arquivo Anexado NESTA TURNO (se houver) ---
    file_processing_result_current_turn = "" # Conteúdo do arquivo enviado NESTA turno
    is_file_processed_ok_current_turn = False # Flag para saber se o processamento do arquivo NESTA turno foi bem sucedido

    if uploaded_file:
        print(f"[/ask] Tipo MIME do arquivo NESTA TURNO: {uploaded_file.content_type}")

        if uploaded_file.content_type.startswith('image/'):
            print("[/ask] Arquivo (imagem) NESTA TURNO. Processando com OCR...")
            file_processing_result_current_turn = process_image_with_ocr(uploaded_file)
            if file_processing_result_current_turn and not file_processing_result_current_turn.startswith("Erro") and not file_processing_result_current_turn.startswith("A imagem não contém texto legível"):
                 is_file_processed_ok_current_turn = True
                 print("[/ask] Texto da imagem NESTA TURNO extraído com sucesso.")
            else:
                 print("[/ask] Falha ou sem texto detectado no OCR NESTA TURNO.")


        elif uploaded_file.content_type == 'application/pdf':
            print("[/ask] Arquivo (PDF) NESTA TURNO. Processando...")
            file_processing_result_current_turn = process_uploaded_pdf(uploaded_file)
            if file_processing_result_current_turn and not file_processing_result_current_turn.startswith("Erro") and not file_processing_result_current_turn.startswith("O PDF anexado não contém texto legível"):
                 is_file_processed_ok_current_turn = True
                 print("[/ask] Conteúdo do PDF NESTA TURNO extraído com sucesso.")
            else:
                 print("[/ask] Falha ou sem texto detectado no processamento do PDF NESTA TURNO.")

        else:
            print(f"[/ask] Tipo de arquivo NESTA TURNO não suportado: {uploaded_file.content_type}")
            file_processing_result_current_turn = f"Tipo de arquivo anexado NESTA TURNO ({uploaded_file.filename}, Tipo: {uploaded_file.content_type}) não suportado para processamento no momento.\n"

        # ** ATUALIZA o conteúdo e nome do último arquivo na sessão APÓS processar o arquivo NESTA TURNO **
        # Se o processamento NESTA TURNO foi OK, salva o conteúdo e nome NA SESSAO
        if is_file_processed_ok_current_turn and file_processing_result_current_turn:
             # Limita o tamanho do texto salvo na sessão
             session['last_file_content'] = file_processing_result_current_turn[:MAX_LAST_FILE_CONTENT_SIZE]
             session['last_file_name'] = uploaded_file.filename
             print(f"[/ask] Conteúdo e nome do arquivo NESTA TURNO ({uploaded_file.filename}) salvos na sessão.")
        # Se o processamento NESTA TURNO falhou ou não teve arquivo, o conteúdo e nome do último arquivo na sessão NÃO mudam.


    # --- Iniciar montagem do Contexto e Processamento ---
    chat_history = session.get('chat_history', [])
    print(f"[/ask] Histórico de CHAT da sessão recuperado ({len(chat_history)} turns).")

    formatted_history = ""
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        user_msg = turn.get('user', '')
        ai_msg = turn.get('ai', '')
        formatted_history += f"Usuário: {user_msg}\nAssistente: {ai_msg}\n---\n"
    print("[/ask] Histórico de chat formatado.")


    # ** LÓGICA: Busca Direta no Conteúdo do Último Arquivo da Sessão (se aplicável) **
    file_search_summary = ""
    # Só tenta buscar no arquivo anterior se houver conteúdo salvo E se houver uma pergunta de texto NESTA TURNO
    if last_file_content_from_session and user_question:
        print(f"[/ask] Tentando buscar '{user_question}' ou termos relevantes no conteúdo do último arquivo da sessão ('{last_file_name_from_session}')...")
        # Usa a função de busca do módulo processing
        occurrences = search_text_for_query(last_file_content_from_session, user_question, snippet_size=150)

        if occurrences:
            print(f"[/ask] Encontrado {len(occurrences)} ocorrência(s) no último arquivo da sessão.")
            file_search_summary = f"Resultado da busca por '{user_question}' no último arquivo da sessão ('{last_file_name_from_session}'): Encontrado {len(occurrences)} ocorrência(s).\nTrechos relevantes: " + "\n---\n".join(occurrences)
        else:
            print(f"[/ask] '{user_question}' (ou termos relevantes) NÃO encontrado(s) no último arquivo da sessão.")
            file_search_summary = f"Resultado da busca por '{user_question}' no último arquivo da sessão ('{last_file_name_from_session}'): Nenhuma ocorrência encontrada."
    # else:
        # print("[/ask] Não há conteúdo de arquivo anterior ou pergunta de texto para realizar busca direta.")


    context_parts = []

    # Adiciona o resultado da busca direta no arquivo anterior (se feita)
    if file_search_summary:
         context_parts.append("--- RESULTADO DA BUSCA NO ÚLTIMO ARQUIVO ANEXADO ---\n\n" + file_search_summary)
         print("[/ask] Resultado da busca direta no arquivo anterior adicionado ao contexto.")


    # Prepara os textos que serão usados para a busca RAG (para chunks da BASE DE CONHECIMENTO)
    rag_search_texts = []
    if user_question:
        rag_search_texts.append(user_question)
    # Usa o texto do arquivo ANEXADO NESTA TURNO para a busca RAG na base de conhecimento
    if is_file_processed_ok_current_turn and file_processing_result_current_turn:
        rag_search_texts.append(file_processing_result_current_turn)
    # Não usamos o last_file_content_from_session para a busca RAG na base KB, focamos na busca direta nele.


    relevant_chunks_rag = []
    if rag_system and rag_system.is_ready():
        try:
            if rag_search_texts:
                 print(f"[/ask] Realizando busca RAG na BASE DE CONHECIMENTO a partir de {len(rag_search_texts)} fontes (Pergunta/ArquivoAtual)...")
                 # Gera embeddings para as fontes de busca RAG
                 search_embeddings_rag = rag_system.embedding_model.encode(rag_search_texts, convert_to_numpy=True)

                 all_found_ids = set() # Usa set para ids únicos
                 for embedding in search_embeddings_rag:
                     query_embedding_rag = embedding.reshape(1, -1)
                     # Busca TOP_K_CHUNKS para cada embedding de busca RAG
                     distances, ids = rag_system.faiss_index.search(query_embedding_rag, TOP_K_CHUNKS)
                     valid_ids = ids[0][ids[0] != -1]
                     all_found_ids.update(valid_ids)

                 print(f"[/ask] IDs relevantes combinados (únicos) da busca RAG: {list(all_found_ids)}")

                 # Recupera o texto dos chunks relevantes da BASE DE CONHECIMENTO (limitando)
                 sorted_unique_ids = sorted(list(all_found_ids))[:COMBINED_TOP_K_CHUNKS]
                 relevant_chunks_rag = [rag_system.text_chunks[id] for id in sorted_unique_ids if id < len(rag_system.text_chunks)]


            if relevant_chunks_rag:
                 context_parts.append("--- CONTEXTO RELEVANTE DA BASE DE CONHECIMENTO (RAG) ---\n\n" + "\n---\n".join(relevant_chunks_rag))
                 print("[/ask] Contexto RAG da BASE DE CONHECIMENTO adicionado.")
            else:
                 print("[/ask] Nenhum contexto relevante encontrado na busca RAG na BASE DE CONHECIMENTO.")


        except Exception as e:
             print(f"[/ask] Ocorreu um erro durante a busca RAG na BASE DE CONHECIMENTO: {e}")
             context_parts.append("--- ERRO NA BUSCA RAG --- Ocorreu um erro ao buscar informações na base de conhecimento.")
    else:
         print("[/ask] Sistema RAG não inicializado ou pronto. Pulando busca RAG.")
         context_parts.append("--- RAG INDISPONÍVEL --- O sistema de busca na base de conhecimento não está disponível.")


    # Adiciona o conteúdo do arquivo ANEXADO NESTA TURNO (se processado OK)
    if is_file_processed_ok_current_turn and file_processing_result_current_turn:
        context_parts.append("--- CONTEÚDO DO ARQUIVO ANEXADO NESTA TURNO ---\n\n" + file_processing_result_current_turn)
        print("[/ask] Conteúdo do arquivo NESTA TURNO adicionado ao contexto.")
    # else: # Se não teve arquivo NESTA TURNO ou falhou, não adiciona esta seção.


    # Adiciona o conteúdo COMPLETO (potencialmente truncado) do ÚLTIMO ARQUIVO da sessão
    # Isso serve como uma referência para a IA, além do resultado da busca direta.
    # Só adicionamos se houver conteúdo e se ele não for o mesmo que acabou de ser processado NESTA TURNO (para evitar duplicação)
    if last_file_content_from_session:
         # Uma checagem simples para não duplicar se o mesmo arquivo foi re-anexado
         is_same_file_reuploaded = (uploaded_file and is_file_processed_ok_current_turn and session.get('last_file_name') == uploaded_file.filename)
         if not is_same_file_reuploaded:
              context_parts.append(f"--- CONTEÚDO COMPLETO (POTENCIALMENTE TRUNCADO) DO ÚLTIMO ARQUIVO DA SESSÃO ('{last_file_name_from_session}') ---\n\n" + last_file_content_from_session)
              print("[/ask] Conteúdo completo (potencialmente truncado) do último arquivo da sessão adicionado ao contexto.")
         # else:
              # print("[/ask] Conteúdo do último arquivo da sessão não adicionado ao contexto, pois arquivo foi re-anexado e processado nesta turno.")


    context_for_gemini = "\n\n".join(context_parts)

    if not context_for_gemini.strip():
         print("[/ask] Contexto final para Gemini está vazio.")
         context_for_gemini = "Nenhum contexto relevante ou informação de arquivo disponível."
    else:
         print("[/ask] Contexto final para Gemini montado.")


    # Prompt final enviado para o Gemini
    # Ajuste a "Pergunta do usuário" no prompt caso seja apenas envio de arquivo ou se a pergunta é sobre o arquivo anterior
    final_user_query_in_prompt = user_question if user_question else (f"Analise o conteúdo do último arquivo da sessão ('{last_file_name_from_session}') e o contexto fornecido para me dizer sobre o que se trata ou o que você encontrou de relevante." if last_file_content_from_session else "Por favor, forneça mais informações ou anexe um arquivo.")


    prompt_with_rag_context = f"""
Você é um assistente especializado em simplificar processos burocráticos e legais para leigos no Brasil.
Você está participando de uma conversa com um usuário.

Histórico da Conversa (últimas {MAX_HISTORY_TURNS} turns):
---
{formatted_history if formatted_history else "Nenhum histórico anterior."}
---

Considere o seguinte CONTEXTO RELEVANTE para fundamentar sua resposta:
{context_for_gemini}

Utilize também seu conhecimento geral para complementar a resposta ou para responder a perguntas que não são totalmente cobertas pelo contexto fornecido.
SEMPRE forneça uma resposta direta e útil. Baseie-se no contexto fornecido e no seu conhecimento para responder à pergunta do usuário. **Se o contexto incluir um '--- RESULTADO DA BUSCA NO ÚLTIMO ARQUIVO ANEXADO ---', utilize esta informação diretamente para responder a perguntas sobre o conteúdo específico do arquivo anterior (como buscar nomes ou termos).** Se o resultado da busca for 'Nenhuma ocorrência encontrada', diga isso ao usuário de forma clara.

Pergunta do usuário: {final_user_query_in_prompt}
"""
    print("[/ask] Prompt final montado.")
    # print(f"Prompt completo enviado para Gemini: {prompt_with_rag_context}") # Log opcional do prompt completo


    # Verifica se há contexto relevante ou pergunta antes de chamar a IA
    # Agora pode chamar a IA mesmo que só tenha contexto de arquivo anterior ou resultado de busca direta
    # Só não chama se não tiver PERGUNTA E não tiver NADA de contexto (nem RAG, nem arquivo atual, nem arquivo anterior/busca)
    if not user_question and not context_for_gemini.strip() and not (uploaded_file and file_processing_result_current_turn):
         print("[/ask] Sem entrada (pergunta/arquivo) e sem contexto relevante/arquivo anterior. Não chamando a IA.")
         resposta_texto = "Desculpe, não recebi uma pergunta de texto, arquivo anexado ou contexto relevante para processar."
    elif not gemini_model:
        print("[/ask] Modelo Gemini não inicializado.")
        resposta_texto = "Desculpe, o serviço de IA não está disponível no momento. Verifique a inicialização."
    else:
        try:
            ai_response = gemini_model.generate_content(prompt_with_rag_context)
            print("[/ask] Resposta do Gemini recebida.")

            if ai_response and ai_response.text:
                resposta_texto = ai_response.text
                print("[/ask] Resposta da IA pronta para envio.")
            else:
                print("[/ask] Resposta da IA vazia ou sem texto.")
                # Tenta pegar algum feedback do prompt para incluir na resposta de erro, se disponível
                feedback = ""
                if hasattr(ai_response, 'prompt_feedback') and ai_response.prompt_feedback:
                     feedback_reason = []
                     if hasattr(ai_response.prompt_feedback, 'block_reason'):
                          feedback_reason.append(f"Block Reason: {ai_response.prompt_feedback.block_reason.name}")
                     if hasattr(ai_response.prompt_feedback, 'safety_ratings'):
                          ratings = [f"{r.category.name}: {r.probability.name}" for r in ai_response.prompt_feedback.safety_ratings]
                          feedback_reason.append(f"Safety Ratings: {', '.join(ratings)}")
                     if feedback_reason:
                          feedback = " (Feedback: " + "; ".join(feedback_reason) + ")"

                resposta_texto = f"Desculpe, não consegui gerar uma resposta para isso no momento.{feedback}"
                print(f"[/ask] Resposta da IA vazia. Feedback: {feedback}")


        except Exception as e:
            print(f"[/ask] Ocorreu um erro ao chamar a API Gemini: {e}")
            if hasattr(e, 'response'):
                 print(f"[/ask] Detalhes da Resposta da API: {e.response}")
            error_message = "Ocorreu um erro ao gerar a resposta da IA."
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 try:
                      # Tenta extrair mensagem de erro da API se disponível
                      api_error_details = e.response.json()
                      if "message" in api_error_details:
                           error_message = f"Erro da API: {api_error_details['message']}"
                      elif "error" in api_error_details and "message" in api_error_details["error"]:
                            error_message = f"Erro da API: {api_error_details['error']['message']}"
                 except:
                       pass
            resposta_texto = error_message # Define a resposta como a mensagem de erro


    # Atualiza o histórico de chat
    user_hist_display = user_question if user_question else (f"Anexado: {uploaded_file.filename}" if uploaded_file else "Sem entrada") # Simplifica a exibição do histórico
    chat_history.append({'user': user_hist_display, 'ai': resposta_texto})
    session['chat_history'] = chat_history[-MAX_HISTORY_TURNS:]
    print(f"[/ask] Histórico de CHAT da sessão atualizado. Tamanho atual: {len(session['chat_history'])} turns.")


    return jsonify({"response": resposta_texto})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    # Limpa o histórico de chat E o histórico de arquivo na sessão
    session.pop('chat_history', None)
    session.pop('last_file_content', None) # Limpa o conteúdo do último arquivo
    session.pop('last_file_name', None) # Limpa o nome do último arquivo
    print("[/clear_history] Histórico da sessão (chat e arquivo) limpo.")
    return jsonify({"status": "success", "message": "Histórico da conversa limpo."})

if __name__ == '__main__':
    print("\nIniciando o servidor Flask...")
    app.run(debug=True)