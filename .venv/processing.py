import os
import io
from PIL import Image
import pytesseract
import pypdf
import re # Importa regex para busca de texto

# --- Configuração do Tesseract OCR (Movida para cá) ---
# A configuração do caminho do Tesseract agora é feita DENTRO deste módulo
TESSERACT_CMD = None
try:
    tesseract_path_env = os.getenv("TESSERACT_PATH")
    if tesseract_path_env and os.path.exists(tesseract_path_env):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path_env
        TESSERACT_CMD = pytesseract.pytesseract.tesseract_cmd # Salva o caminho configurado
        print(f"[Processing] Caminho do Tesseract configurado via TESSERACT_PATH: {tesseract_path_env}")
    else:
         try:
             # Tenta achar no PATH do sistema
             pytesseract.get_tesseract_version()
             TESSERACT_CMD = pytesseract.pytesseract.tesseract_cmd # Salva o caminho encontrado
             print("[Processing] Tesseract encontrado no PATH do sistema.")
         except pytesseract.TesseractNotFoundError:
             print("[Processing] Erro: Executável Tesseract OCR não encontrado no PATH!")
             print("Por favor, instale o Tesseract OCR ou configure TESSERACT_PATH no .env")
             TESSERACT_CMD = None # Define como None se não encontrar

except Exception as e:
    print(f"[Processing] Erro inesperado na configuração do Tesseract: {e}")
    TESSERACT_CMD = None


# --- Funções de Processamento e Busca ---

def search_text_for_query(text: str, query: str, snippet_size: int = 150) -> list[str]:
    """
    Busca a query (case-insensitive) dentro de um texto e retorna snippets ao redor das ocorrências.
    Se a query for composta por múltiplas palavras, tenta buscar a frase completa e também
    palavras individuais que pareçam importantes (excluindo stop words simples).
    Retorna uma lista de strings, onde cada string é um snippet.
    """
    if not text or not query:
        return []

    # Simple stop words list for basic phrase extraction
    # Mantive a lista de stop words, pode ajustar conforme necessário
    stop_words = set(["a", "o", "de", "da", "do", "e", "é", "um", "uma", "o", "os", "as", "em", "no", "na", "para", "com", "por", "que", "tem", "algum", "alguma", "alguns", "algumas", "esse", "essa", "nesse", "nessa", "lista", "arquivo", "documento", "meu", "minha", "meus", "minhas", "seus", "suas", "seu", "sua"])

    search_terms = set()
    # Adiciona a query completa como um termo de busca inicial
    search_terms.add(query.strip())

    # Tenta adicionar palavras individuais que não são stop words
    words = re.findall(r'\b\w+\b', query.lower())
    for word in words:
        if word not in stop_words and len(word) > 2: # Palavras com mais de 2 letras e não stop words
            search_terms.add(word)

    print(f"[Processing/Search] Termos de busca identificados: {list(search_terms)}")

    all_occurrences = set() # Usa um set para armazenar snippets únicos
    text_lower = text.lower()


    # Tenta buscar cada termo identificado
    for term in search_terms:
        term_lower = term.lower()
        if not term_lower: continue # Pula se o termo for vazio após strip()

        # Usa re.escape para garantir que caracteres especiais no termo sejam tratados
        for match in re.finditer(re.escape(term_lower), text_lower, re.IGNORECASE):
             start, end = match.span() # Posição no texto original (lower case)

             # Extrai o snippet do texto ORIGINAL (para manter capitalização original)
             snippet_start = max(0, start - snippet_size)
             snippet_end = min(len(text), end + snippet_size)

             snippet = text[snippet_start:snippet_end]
             if snippet_start > 0: snippet = "..." + snippet
             if snippet_end < len(text): snippet = snippet + "..."

             all_occurrences.add(snippet)

    print(f"[Processing/Search] Busca encontrou {len(all_occurrences)} snippets únicos.")

    return list(all_occurrences) # Converte o set de volta para lista


def process_image_with_ocr(image_file) -> str:
    """Extrai texto de uma imagem usando Tesseract OCR."""
    extracted_text = ""
    if TESSERACT_CMD is None:
        print("[Processing/OCR] Tesseract OCR não configurado/encontrado. Pulando processamento de imagem.")
        return "Erro: Tesseract OCR não configurado/encontrado."

    try:
        image_file.seek(0)
        image = Image.open(io.BytesIO(image_file.read()))
        # print("[Processing/OCR] Imagem aberta com sucesso usando Pillow.") # Comentado

        if image.mode != 'RGB':
            image = image.convert('RGB')
            # print("[Processing/OCR] Imagem convertida para RGB.") # Comentado

        extracted_text = pytesseract.image_to_string(image, lang='por')
        # print(f"[Processing/OCR] Texto extraído da imagem (first 100 chars): {extracted_text[:100]}...") # Comentado
        if not extracted_text.strip():
             print("[Processing/OCR] OCR não extraiu texto significativo da imagem.")
             return "A imagem não contém texto legível via OCR." # Mensagem específica se OCR falhar em extrair texto

    except Exception as e:
        print(f"[Processing/OCR] Ocorreu um erro ao processar a imagem com Pillow ou Tesseract: {e}")
        return f"Erro ao processar a imagem: {e}"

    return extracted_text


def process_uploaded_pdf(pdf_file) -> str:
    """Extrai texto de um arquivo PDF (FileStorage) enviado via upload."""
    extracted_text = ""
    try:
        pdf_file.seek(0)
        reader = pypdf.PdfReader(pdf_file)
        print(f"[Processing/PDF] Extraindo texto de PDF anexado ({pdf_file.filename}, {len(reader.pages)} páginas)...")
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

        print("[Processing/PDF] Extração de texto do PDF concluída.")
        if not extracted_text.strip():
            print("[Processing/PDF] PDF anexado não contém texto legível.")
            return "O PDF anexado não contém texto legível."

    except Exception as e:
        print(f"[Processing/PDF] Ocorreu um erro ao processar o PDF anexado: {e}")
        return f"Erro ao processar o PDF anexado: {e}"

    return extracted_text