document.addEventListener('DOMContentLoaded', function() {
    const chatbox = document.getElementById('chatbox');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const fileNameSpan = document.getElementById('file-name');
    const clearFileButton = document.getElementById('clear-file');
    const newChatButton = document.getElementById('new-chat-button');

    let attachedFile = null;

    // Função para adicionar mensagens ao chat (MODIFICADA para usar marked.js)
    function addMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender + '-message');

        if (sender === 'assistant') {
            // Converte Markdown para HTML APENAS para mensagens da IA
            messageElement.innerHTML = marked.parse(message);
        } else {
            // Para mensagens do usuário, apenas define o texto
            messageElement.textContent = message; // Usar textContent é mais seguro para inputs de usuário
        }

        // Adiciona animação de entrada
        messageElement.style.opacity = '0';
        messageElement.style.transform = 'translateY(10px)';
        
        chatbox.appendChild(messageElement);
        
        // Força reflow para garantir que a animação funcione
        void messageElement.offsetWidth;
        
        // Aplica a animação
        messageElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        messageElement.style.opacity = '1';
        messageElement.style.transform = 'translateY(0)';
        
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        const fileToSend = attachedFile;

        console.log("--- Tentando enviar mensagem ---");
        console.log("Valor do input de texto (message):", message);
        console.log("Arquivo anexado (fileToSend):", fileToSend);
        console.log("---------------------------------");

        if (!message && !fileToSend) {
            console.log("Nenhuma mensagem ou arquivo para enviar. Abortando.");
            return;
        }

        // Adiciona a mensagem do usuário ou indicação de anexo
        if (message) {
            addMessage(message, 'user');
        } else if (fileToSend) {
            addMessage(`Arquivo anexado: ${fileToSend.name}`, 'user'); // Indica que um arquivo foi enviado
        }

        const formData = new FormData();
        formData.append('user_input', message);
        if (fileToSend) {
            formData.append('file', fileToSend);
        }

        console.log("Conteúdo do FormData antes de enviar:");
        for (let pair of formData.entries()) {
            console.log(pair[0] + ': ' + pair[1]);
        }
        console.log("---------------------------------");

        userInput.value = '';
        sendButton.disabled = true;
        fileInput.disabled = true;
        newChatButton.disabled = true;

        // Novo indicador de digitação com pontos animados
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'assistant-message', 'typing-indicator');
        typingIndicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
        chatbox.appendChild(typingIndicator);
        chatbox.scrollTop = chatbox.scrollHeight;

        try {
            console.log("Enviando requisição para /ask...");
            const response = await fetch('/ask', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`HTTP error! status: ${response.status}`, errorText);
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log("Resposta da IA recebida:", data);

            if (chatbox.contains(typingIndicator)) {
                // Animação de saída para o indicador de digitação
                typingIndicator.style.opacity = '0';
                setTimeout(() => {
                    if (chatbox.contains(typingIndicator)) {
                        chatbox.removeChild(typingIndicator);
                    }
                    // Usa a função addMessage, que agora converte Markdown
                    addMessage(data.response, 'assistant');
                }, 300);
            } else {
                addMessage(data.response, 'assistant');
            }

            if (fileToSend) {
                filePreview.style.display = 'none';
                fileNameSpan.textContent = '';
                attachedFile = null;
                fileInput.value = '';
            }

        } catch (error) {
            console.error('Erro durante a requisição /ask:', error);
            if (chatbox.contains(typingIndicator)) {
                chatbox.removeChild(typingIndicator);
            }
            addMessage('Desculpe, ocorreu um erro ao processar sua solicitação.', 'assistant');
            if (fileToSend) {
                filePreview.style.display = 'none';
                fileNameSpan.textContent = '';
                attachedFile = null;
                fileInput.value = '';
            }

        } finally {
            console.log("Requisição /ask finalizada.");
            sendButton.disabled = false;
            fileInput.disabled = false;
            newChatButton.disabled = false;
        }
    }

    async function startNewChat() {
        newChatButton.disabled = true;
        sendButton.disabled = true;
        fileInput.disabled = true;

        // Animação de saída para todas as mensagens existentes
        const existingMessages = chatbox.querySelectorAll('.message');
        existingMessages.forEach(msg => {
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-10px)';
        });

        // Aguarda a animação terminar antes de limpar
        setTimeout(() => {
            chatbox.innerHTML = '';
            const initialMessage = document.createElement('div');
            initialMessage.classList.add('message', 'assistant-message');
            // Converte a mensagem inicial também, caso ela contenha Markdown
            initialMessage.innerHTML = marked.parse('Olá! Sou Harvey Specter IA. Como posso te ajudar com questões legais ou processos administrativos hoje?');
            
            // Inicia com opacidade 0 para animar entrada
            initialMessage.style.opacity = '0';
            initialMessage.style.transform = 'translateY(10px)';
            
            chatbox.appendChild(initialMessage);
            
            // Força reflow
            void initialMessage.offsetWidth;
            
            // Aplica animação de entrada
            initialMessage.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            initialMessage.style.opacity = '1';
            initialMessage.style.transform = 'translateY(0)';
            
            chatbox.scrollTop = chatbox.scrollHeight;
        }, 300);

        userInput.value = '';
        filePreview.style.display = 'none';
        fileNameSpan.textContent = '';
        attachedFile = null;
        fileInput.value = '';

        try {
            console.log("Enviando requisição para /clear_history...");
            const response = await fetch('/clear_history', {
                method: 'POST'
            });

            if (!response.ok) {
                console.error('Erro ao limpar histórico no backend:', response.statusText);
                // A mensagem de erro pode ser adicionada diretamente, sem conversão Markdown
                setTimeout(() => {
                    addMessage('Houve um erro ao reiniciar a conversa no servidor. A interface foi limpa, mas o histórico anterior pode ter sido mantido.', 'assistant');
                }, 500);
            } else {
                console.log('Histórico de conversa limpo no backend.');
            }
        } catch (error) {
            console.error('Erro ao conectar com o backend para limpar histórico:', error);
            // A mensagem de erro pode ser adicionada diretamente, sem conversão Markdown
            setTimeout(() => {
                addMessage('Houve um erro ao comunicar com o servidor para reiniciar a conversa. A interface foi limpa, mas o histórico anterior pode ter sido mantido.', 'assistant');
            }, 500);
        } finally {
            console.log("Requisição /clear_history finalizada.");
            setTimeout(() => {
                newChatButton.disabled = false;
                sendButton.disabled = false;
                fileInput.disabled = false;
            }, 300);
        }
    }

    // Efeito de foco no input
    userInput.addEventListener('focus', function() {
        this.parentElement.classList.add('input-focused');
    });

    userInput.addEventListener('blur', function() {
        this.parentElement.classList.remove('input-focused');
    });

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            attachedFile = this.files[0];
            fileNameSpan.textContent = attachedFile.name;
            
            // Animação de entrada para o preview do arquivo
            filePreview.style.opacity = '0';
            filePreview.style.display = 'flex';
            
            // Força reflow
            void filePreview.offsetWidth;
            
            // Aplica animação
            filePreview.style.transition = 'opacity 0.3s ease';
            filePreview.style.opacity = '1';
        } else {
            attachedFile = null;
            fileNameSpan.textContent = '';
            filePreview.style.display = 'none';
        }
    });

    clearFileButton.addEventListener('click', function() {
        // Animação de saída para o preview do arquivo
        filePreview.style.opacity = '0';
        
        setTimeout(() => {
            attachedFile = null;
            fileInput.value = '';
            fileNameSpan.textContent = '';
            filePreview.style.display = 'none';
        }, 300);
    });

    newChatButton.addEventListener('click', startNewChat);
});
