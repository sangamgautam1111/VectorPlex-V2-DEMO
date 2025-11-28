// VectorPlex V2 - Chat Script
document.addEventListener('DOMContentLoaded', () => {
    // Get session ID from window object (set by Flask template)
    const SESSION_ID = window.SESSION_ID || getSessionIdFromUrl();
    
    // Elements
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceInputBtn = document.getElementById('voiceInputBtn');
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');
    const themeToggle = document.getElementById('themeToggle');
    const voiceModeBtn = document.getElementById('voiceModeBtn');
    const leaveChatBtn = document.getElementById('leaveChatBtn');
    const exportBtn = document.getElementById('exportBtn');
    
    // Modals
    const voiceModal = document.getElementById('voiceModal');
    const leaveModal = document.getElementById('leaveModal');
    const exportModal = document.getElementById('exportModal');
    
    // Ray effects
    const blueRayEffect = document.getElementById('blueRayEffect');
    const yellowRayEffect = document.getElementById('yellowRayEffect');
    
    // Audio
    const swooshSound = document.getElementById('swooshSound');
    
    // State
    const messages = [];
    let isProcessing = false;
    let recognition = null;

    // Initialize
    init();

    function init() {
        setupEventListeners();
        setupTextarea();
        setupSpeechRecognition();
        loadTheme();
        updateSendButton();
        
        // Load existing messages if any
        loadChatHistory();
    }

    function getSessionIdFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/chat\/([^/]+)/);
        return match ? match[1] : null;
    }

    function setupEventListeners() {
        // Send message
        if (sendBtn) {
            sendBtn.addEventListener('click', sendMessage);
        }

        // Enter to send
        if (messageInput) {
            messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            messageInput.addEventListener('input', updateSendButton);
        }

        // Quick prompts
        document.querySelectorAll('.prompt-btn, .suggestion-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.dataset.prompt;
                if (prompt && messageInput) {
                    messageInput.value = prompt;
                    updateSendButton();
                    sendMessage();
                }
            });
        });

        // Menu toggle (mobile)
        if (menuToggle) {
            menuToggle.addEventListener('click', toggleSidebar);
        }

        // Theme toggle
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
        }

        // Voice mode button
        if (voiceModeBtn) {
            voiceModeBtn.addEventListener('click', () => openModal(voiceModal));
        }

        // Voice input button
        if (voiceInputBtn) {
            voiceInputBtn.addEventListener('click', () => openModal(voiceModal));
        }

        // Leave chat button
        if (leaveChatBtn) {
            leaveChatBtn.addEventListener('click', () => openModal(leaveModal));
        }

        // Export button
        if (exportBtn) {
            exportBtn.addEventListener('click', () => openModal(exportModal));
        }

        // Modal close buttons
        document.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
            el.addEventListener('click', (e) => {
                if (e.target === el) {
                    const modal = e.target.closest('.modal');
                    if (modal) closeModal(modal);
                }
            });
        });

        // Leave chat confirmation
        const confirmLeaveBtn = document.getElementById('confirmLeaveBtn');
        const cancelLeaveBtn = document.getElementById('cancelLeaveBtn');
        
        if (confirmLeaveBtn) {
            confirmLeaveBtn.addEventListener('click', leaveChat);
        }
        if (cancelLeaveBtn) {
            cancelLeaveBtn.addEventListener('click', () => closeModal(leaveModal));
        }

        // Voice modal buttons
        const startVoiceBtn = document.getElementById('startVoiceBtn');
        const sendVoiceBtn = document.getElementById('sendVoiceBtn');
        const closeVoiceModal = document.getElementById('closeVoiceModal');

        if (startVoiceBtn) {
            startVoiceBtn.addEventListener('click', toggleVoiceRecognition);
        }
        if (sendVoiceBtn) {
            sendVoiceBtn.addEventListener('click', sendVoiceMessage);
        }
        if (closeVoiceModal) {
            closeVoiceModal.addEventListener('click', () => closeModal(voiceModal));
        }

        // Export options
        document.querySelectorAll('.export-option').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.dataset.format;
                exportChat(format);
                closeModal(exportModal);
            });
        });

        const closeExportModal = document.getElementById('closeExportModal');
        if (closeExportModal) {
            closeExportModal.addEventListener('click', () => closeModal(exportModal));
        }

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (sidebar && sidebar.classList.contains('active')) {
                    if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                        sidebar.classList.remove('active');
                    }
                }
            }
        });
    }

    function setupTextarea() {
        if (!messageInput) return;
        
        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        });
    }

    function setupSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                const voiceTranscript = document.getElementById('voiceTranscript');
                if (voiceTranscript) {
                    voiceTranscript.textContent = transcript;
                }
                const sendVoiceBtn = document.getElementById('sendVoiceBtn');
                if (sendVoiceBtn) {
                    sendVoiceBtn.disabled = !transcript.trim();
                }
            };

            recognition.onend = () => {
                const voiceVisualizer = document.getElementById('voiceVisualizer');
                const voiceStatus = document.getElementById('voiceStatus');
                const startVoiceBtn = document.getElementById('startVoiceBtn');
                
                if (voiceVisualizer) voiceVisualizer.classList.remove('listening');
                if (voiceStatus) voiceStatus.textContent = 'Click to start speaking';
                if (startVoiceBtn) {
                    startVoiceBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Listening';
                }
            };
        }
    }

    function updateSendButton() {
        if (sendBtn && messageInput) {
            sendBtn.disabled = !messageInput.value.trim() || isProcessing;
        }
    }

    // Sidebar toggle
    function toggleSidebar() {
        if (sidebar) {
            sidebar.classList.toggle('active');
        }
    }

    // Theme functions
    function loadTheme() {
        const savedTheme = localStorage.getItem('vectorplex-theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('vectorplex-theme', newTheme);
        updateThemeIcon(newTheme);
    }

    function updateThemeIcon(theme) {
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }

    // Modal functions
    function openModal(modal) {
        if (modal) modal.classList.add('active');
    }

    function closeModal(modal) {
        if (modal) modal.classList.remove('active');
    }

    // Voice functions
    function toggleVoiceRecognition() {
        if (!recognition) {
            showToast('Speech recognition not supported in this browser', 'error');
            return;
        }

        const voiceVisualizer = document.getElementById('voiceVisualizer');
        const voiceStatus = document.getElementById('voiceStatus');
        const startVoiceBtn = document.getElementById('startVoiceBtn');

        if (voiceVisualizer && voiceVisualizer.classList.contains('listening')) {
            recognition.stop();
        } else {
            recognition.start();
            if (voiceVisualizer) voiceVisualizer.classList.add('listening');
            if (voiceStatus) voiceStatus.textContent = 'Listening...';
            if (startVoiceBtn) startVoiceBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Listening';
        }
    }

    function sendVoiceMessage() {
        const voiceTranscript = document.getElementById('voiceTranscript');
        if (voiceTranscript && voiceTranscript.textContent.trim()) {
            messageInput.value = voiceTranscript.textContent;
            closeModal(voiceModal);
            sendMessage();
            voiceTranscript.textContent = '';
        }
    }

    // Load chat history
    async function loadChatHistory() {
        if (!SESSION_ID) return;
        
        try {
            const response = await fetch(`/api/history/${SESSION_ID}`);
            const data = await response.json();
            
            if (data.success && data.messages && data.messages.length > 0) {
                // Hide welcome screen
                if (welcomeScreen) {
                    welcomeScreen.style.display = 'none';
                }
                
                // Add existing messages
                data.messages.forEach(msg => {
                    const type = msg.role === 'user' ? 'user' : 'ai';
                    addMessage(type, msg.content, false);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    // Send message
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isProcessing || !SESSION_ID) return;

        isProcessing = true;
        updateSendButton();

        // Hide welcome screen
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }

        // Add user message
        addMessage('user', text);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Add AI typing indicator
        const typingId = addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    session_id: SESSION_ID,
                    message: text 
                })
            });

            const data = await response.json();

            // Remove typing indicator
            removeTypingIndicator(typingId);

            if (data.success) {
                const aiResponse = data.response;
                addMessage('ai', aiResponse);

                // Check for code and trigger blue ray
                if (containsCode(aiResponse)) {
                    triggerBlueRay();
                }

                // Check for math/formulas and trigger yellow ray
                if (containsMath(aiResponse)) {
                    triggerYellowRay();
                }
            } else {
                addMessage('ai', 'Sorry, I encountered an error. Please try again.');
                showToast(data.error || 'Failed to get response', 'error');
            }
        } catch (error) {
            console.error('Chat error:', error);
            removeTypingIndicator(typingId);
            addMessage('ai', 'Sorry, I encountered an error. Please try again.');
            showToast('Network error. Please check your connection.', 'error');
        }

        isProcessing = false;
        updateSendButton();
    }

    function addMessage(type, content, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (!animate) {
            messageDiv.style.animation = 'none';
        }

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = formatMessage(content);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        messages.push({ type, content, timestamp: new Date() });
    }

    function formatMessage(content) {
        // Escape HTML first
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Format code blocks
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'code';
            return `
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-lang">${language}</span>
                        <button class="copy-btn" onclick="copyCode(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <div class="code-content">
                        <pre>${code.trim()}</pre>
                    </div>
                </div>
            `;
        });

        // Format inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Format math formulas ($$...$$)
        formatted = formatted.replace(/\$\$([^$]+)\$\$/g, '<span class="math-highlight">$1</span>');

        // Format headers
        formatted = formatted.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        formatted = formatted.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        formatted = formatted.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Format bold text
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Format italic
        formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Format line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai';
        messageDiv.id = id;

        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return id;
    }

    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    // Content detection
    function containsCode(text) {
        return text.includes('```') || /`[^`]+`/.test(text);
    }

    function containsMath(text) {
        return /\$\$[^$]+\$\$/.test(text) || 
               /\b(equation|formula|calculate|math|=|×|÷|∑|∫|√)\b/i.test(text);
    }

    // Ray effects
    function triggerBlueRay() {
        if (blueRayEffect) {
            blueRayEffect.classList.add('active');
            playSwoosh();
            setTimeout(() => {
                blueRayEffect.classList.remove('active');
            }, 800);
        }
    }

    function triggerYellowRay() {
        if (yellowRayEffect) {
            yellowRayEffect.classList.add('active');
            setTimeout(() => {
                yellowRayEffect.classList.remove('active');
            }, 800);
        }
    }

    function playSwoosh() {
        if (swooshSound) {
            swooshSound.currentTime = 0;
            swooshSound.play().catch(() => {});
        }
    }

    // Leave chat
    async function leaveChat() {
        try {
            await fetch(`/api/session/${SESSION_ID}/cleanup`, {
                method: 'POST'
            });
            
            showToast('Session data deleted', 'success');
            
            setTimeout(() => {
                window.location.href = '/';
            }, 500);
        } catch (error) {
            console.error('Leave chat error:', error);
            showToast('Error leaving chat', 'error');
        }
    }

    // Export chat
    function exportChat(format) {
        if (messages.length === 0) {
            showToast('No messages to export', 'error');
            return;
        }

        let content = '';
        const filename = `vectorplex-chat-${SESSION_ID ? SESSION_ID.slice(0, 8) : 'export'}`;

        if (format === 'json') {
            content = JSON.stringify(messages, null, 2);
        } else if (format === 'md') {
            content = `# VectorPlex Chat Export\n\n`;
            content += messages.map(m => {
                const prefix = m.type === 'user' ? '**You:**' : '**AI:**';
                return `${prefix}\n${m.content}\n`;
            }).join('\n---\n\n');
        } else {
            content = `VectorPlex Chat Export\n${'='.repeat(40)}\n\n`;
            content += messages.map(m => {
                const prefix = m.type === 'user' ? 'You:' : 'AI:';
                return `${prefix}\n${m.content}\n`;
            }).join('\n---\n\n');
        }

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${filename}.${format}`;
        a.click();
        URL.revokeObjectURL(url);

        showToast('Chat exported successfully', 'success');
    }

    // Toast
    function showToast(message, type = 'success') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
        `;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // Global copy function
    window.copyCode = (btn) => {
        const codeBlock = btn.closest('.code-block');
        const code = codeBlock.querySelector('pre').textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => {
                btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        }).catch(() => {
            showToast('Failed to copy', 'error');
        });
    };
});