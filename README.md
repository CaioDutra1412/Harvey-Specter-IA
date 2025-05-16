# ✨ Harvey Specter IA: Seu Assistente Jurídico e Burocrático Potencializado por IA ✨

![Harvey Specter IA Demo](https://s7494.pcdn.co/college-blog/files/2016/10/Harvey-Specter.gif)

## 📄 Sobre o Projeto

Este projeto é uma aplicação web desenvolvida em Python com Flask que simula um assistente jurídico e burocrático, inspirado na eficiência e no conhecimento profundo de **Harvey Specter** da série Suits, quem ai conhece? kkk. Utilizando o poder da **Inteligência Artificial Generativa (IA)** através da API **Google Gemini Flash 2.0** e a técnica de **Retrieval Augmented Generation (RAG)**, a aplicação permite interagir com uma base de conhecimento legal e processar documentos (PDFs e Imagens) para fornecer respostas e resumos relevantes.


## 🧠 Funcionalidades Implementadas

Desenvolvemos diversas funcionalidades para tornar este assistente inteligente e útil:

* **Integração com Google Gemini Flash 2.0:** Utiliza o modelo de IA mais recente e eficiente para gerar respostas coerentes e informadas.
* **Sistema RAG (Retrieval Augmented Generation):**
    * 📚 Carregamento e processamento de uma **Base de Conhecimento (Knowledge Base)** a partir de arquivos PDF em um diretório local (`knowledge_base/`).
    * ✂️ **Chunking** inteligente do texto para dividir documentos longos em partes gerenciáveis.
    * 📈 Geração de **Embeddings** (representações numéricas do texto) usando o modelo `paraphrase-MiniLM-L6-v2`.
    * 🔍 **Indexação e Busca por Similaridade** no índice **FAISS** para encontrar os trechos mais relevantes da base de conhecimento com base na pergunta do usuário.
* **Conteúdo da Base de Conhecimento:**
        A pasta `knowledge_base/` contém os seguintes documentos que serviram de referência para a aprendizado do Harvey:
        * `CC.pdf`: Código Civil Brasileiro (aprox. 616 páginas)
        * `CDDC.pdf`: Código de Defesa do Consumidor (aprox. 97 páginas)
        * `CLT.pdf`: Consolidação das Leis do Trabalho (CLT) (aprox. 193 páginas)
        * `CPC.pdf`: Código de Processo Civil (aprox. 322 páginas)
        * `CPP.pdf`: Código de Processo Penal (aprox. 191 páginas)
        * `CPpdf.pdf`: Código Penal (aprox. 143 páginas)
        * `CTB.pdf`: Código de Trânsito Brasileiro (aprox. 232 páginas)
        * `ECA.pdf`: Estatuto da Criança e do Adolescente (aprox. 119 páginas)
        * `EDI.pdf`: Estatuto do Idoso (aprox. 41 páginas)
        * `LEIS2024.pdf`: Legislação Atualizada (Compilação de Leis - aprox. 286 páginas)
        *(O número exato de páginas pode variar dependendo da versão do arquivo).*
* **Processamento de Documentos Anexados:**
    * 📎 **Upload de Arquivos:** Permite ao usuário anexar arquivos diretamente na interface do chat.
    * 📄 **Extração de Texto de PDFs:** Processa arquivos PDF anexados para extrair seu conteúdo textual.
    * 🖼️ **Reconhecimento de Texto em Imagens (OCR):** Utiliza a biblioteca **Pillow** e **pytesseract** para extrair texto de imagens (requer instalação do Tesseract OCR).
    * ✨ O texto extraído de documentos anexados é **adicionado ao contexto** enviado para a IA, permitindo que ela responda com base no conteúdo dos seus arquivos.
* **Interface de Chat Interativa (Frontend):**
    * 💬 Interface moderna e responsiva construída com HTML, CSS e JavaScript puro.
    * 🔄 Exibição em tempo real das mensagens do usuário e das respostas da IA.
    * ⌨️ Envio de mensagens via botão ou tecla Enter.
    * ⏳ Indicador visual de "digitando" enquanto a IA processa a resposta.
    * 🗑️ Botão **"Nova Conversa"** para limpar o histórico do chat no frontend e a sessão no backend, iniciando uma nova interação do zero.
    * ✨ Renderização de **Markdown** nas respostas da IA (usando `marked.js`) para uma formatação mais agradável (negrito, listas, etc.).
    * ⬆️ Pré-visualização do nome do arquivo anexado antes do envio.
* **Gerenciamento de Sessão:**
    * 🔒 Utiliza sessões Flask para manter o histórico da conversa entre as requisições do usuário (limitado às últimas X turnos).
    * 🧹 Limpeza automática do histórico da sessão ao carregar a página principal ou clicar em "Nova Conversa".

## 🚀 Como Rodar o Projeto Localmente

Siga os passos abaixo para configurar e executar o projeto na sua máquina:

### Pré-requisitos

* **Python 3.8+**
* **Git** (opcional, para clonar o repositório)
* **Tesseract OCR:** Necessário para o processamento de imagens. Faça o download e instale de acordo com seu sistema operacional (O EXECUTÁVEL ESTA DISPONÍVEL TAMBÉM AQUI NO REPOSITÓRIO). [Guia de Instalação do Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    * No Windows, considere adicionar o diretório de instalação do Tesseract ao PATH do sistema ou configurar a variável de ambiente `TESSERACT_PATH`.
    * No Linux (Debian/Ubuntu), `sudo apt-get install tesseract-ocr tesseract-ocr-por`.
    * No macOS (via Homebrew), `brew install tesseract tesseract-lang`.

### Configuração

1.  **Clone o Repositório (ou baixe os arquivos):**
    ```bash
    git clone <link_do_seu_repositorio_github>
    cd <nome_da_pasta_do_seu_projeto>
    ```

2.  **Crie um Ambiente Virtual (Recomendado):**
    ```bash
    python -m venv .venv
    ```

3.  **Ative o Ambiente Virtual:**
    * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Instale as Dependências:**
    Crie um arquivo chamado `requirements.txt` na raiz do projeto com o seguinte conteúdo:
    ```
    flask
    python-dotenv
    google-generativeai
    pypdf
    Pillow
    pytesseract
    numpy
    faiss-cpu # Use faiss-gpu se tiver GPU compatível e quiser performance
    sentence-transformers
    ```
    Em seguida, instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Obtenha suas Chaves de API:**
    * **Google AI API Key:** Obtenha uma chave API para o Google Gemini no [Google AI Studio](https://aistudio.google.com/).
    * **Flask SECRET_KEY:** Gere uma chave secreta para as sessões Flask (você pode gerar uma rodando `import os; print(os.urandom(24).hex())` em uma sessão Python).

6.  **Crie o Arquivo `.env`:**
    Na raiz do projeto, crie um arquivo chamado `.env` e adicione suas chaves:
    ```env
    GOOGLE_API_KEY=SUA_CHAVE_DO_GOOGLE_AQUI
    SECRET_KEY=SUA_CHAVE_SECRETA_DO_FLASK_AQUI
    # Opcional: Se o Tesseract OCR não estiver no PATH, defina o caminho completo:
    # TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    ```
    Substitua `SUA_CHAVE_DO_GOOGLE_AQUI` e `SUA_CHAVE_SECRETA_DO_FLASK_AQUI` pelos seus valores reais. Configure `TESSERACT_PATH` se necessário.

7.  **Adicione Documentos à Base de Conhecimento:**
    Crie uma pasta chamada `knowledge_base` na raiz do projeto (se ela ainda não existir). Coloque dentro dela os arquivos PDF que você quer que a IA use como base de conhecimento.

### Executando a Aplicação

1.  Certifique-se que seu ambiente virtual está ativado.
2.  Execute o arquivo `app.py`:
    ```bash
    python app.py
    ```
3.  Abra seu navegador e acesse `http://127.0.0.1:5000/`.

A aplicação deverá carregar, processar os documentos da sua `knowledge_base` na inicialização e estar pronta para interagir!

## 📂 Estrutura do Projeto

A estrutura básica do projeto é a seguinte:

├── app.py              # Código principal da aplicação Flask, IA e RAG
├── .env                # Arquivo para variáveis de ambiente (chaves API, etc.)
├── requirements.txt    # Lista de dependências do Python
├── knowledge_base/     # Diretório para seus arquivos PDF da base de conhecimento
│   └── documento1.pdf
│   └── documento2.pdf
│   └── ...
└── templates/
└── index.html      # Estrutura HTML da interface do chat
└── static/
├── style.css       # Estilos visuais da aplicação
└── script.js       # Lógica JavaScript para a interface e comunicação com o backend

## 🌱 Implementações Futuras

Este projeto serve como uma base sólida e há diversas possibilidades de aprimoramento para torná-lo ainda mais poderoso e robusto:

* **Aprimoramento da Busca RAG:** Implementar técnicas de busca híbrida que combinem a busca por similaridade de embeddings com a busca por palavras-chave ou que considerem o conteúdo do arquivo anexado diretamente na consulta de similaridade.
* **Tratamento de Erros:** Melhorar a comunicação de erros do backend para o frontend, fornecendo feedback mais útil ao usuário em caso de falhas (ex: API Key inválida, Tesseract não encontrado, arquivo corrompido, etc.).
* **Refatoração do Código:** Modularizar o código do `app.py` em classes ou arquivos separados (ex: `rag.py`, `ai_model.py`) para melhorar a organização e manutenção.
* **Novos Tipos de Arquivo:** Adicionar suporte para processar outros formatos de documento (DOCX, TXT, etc.).
* **Interface do Usuário:** Melhorias na UX, como feedback visual mais claro sobre o processamento de arquivos longos, ou opções de formatação para o usuário.
* **Persistência do Histórico:** Implementar um banco de dados simples para armazenar o histórico de conversas de forma persistente (além da sessão temporária).
* **Configurações:** Permitir a configuração de parâmetros como `CHUNK_SIZE`, `TOP_K_CHUNKS` através da interface ou arquivo de configuração.
* **Deploy:** Publicar a aplicação em um serviço de hospedagem para que seja acessível publicamente.
