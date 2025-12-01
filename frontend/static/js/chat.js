// VectorPlex Demo Model - Chat Script (Complete)
document.addEventListener('DOMContentLoaded', () => {
    // ============================================
    // Configuration & State
    // ============================================

    const SESSION_ID = window.SESSION_ID || getSessionIdFromUrl();

    const STATE = {
        messages: [],
        isProcessing: false,
        isSpeaking: false,
        currentUtterance: null,
        selectedVoice: null
    };

    // ============================================
    // DOM Elements
    // ============================================

    const elements = {
        chatMessages: document.getElementById('chatMessages'),
        welcomeScreen: document.getElementById('welcomeScreen'),
        sidebar: document.getElementById('sidebar'),
        messageInput: document.getElementById('messageInput'),
        sendBtn: document.getElementById('sendBtn'),
        menuToggle: document.getElementById('menuToggle'),
        themeToggle: document.getElementById('themeToggle'),
        clearChatBtn: document.getElementById('clearChatBtn'),
        sidebarCollapseBtn: document.getElementById('sidebarCollapseBtn'),
        exportModal: document.getElementById('exportModal'),
        leaveModal: document.getElementById('leaveModal'),
        exportBtn: document.getElementById('exportBtn'),
        leaveChatBtn: document.getElementById('leaveChatBtn'),
        confirmLeaveBtn: document.getElementById('confirmLeaveBtn'),
        cancelLeaveBtn: document.getElementById('cancelLeaveBtn'),
        closeExportModal: document.getElementById('closeExportModal'),
        toastContainer: document.getElementById('toastContainer')
    };

    // ============================================
    // Initialization
    // ============================================

    function init() {
        setupEventListeners();
        setupTextarea();
        setupSpeechSynthesis();
        loadTheme();
        updateSendButton();
        loadChatHistory();
        setupCodeCopyButtons();
        
        console.log('VectorPlex Chat initialized');
    }

    function getSessionIdFromUrl() {
        const path = window.location.pathname;
        const match = path.match(/\/chat\/([^/]+)/);
        return match ? match[1] : null;
    }

    // ============================================
    // Event Listeners
    // ============================================

    function setupEventListeners() {
        // Send message
        elements.sendBtn?.addEventListener('click', sendMessage);
        
        // Enter to send
        elements.messageInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        elements.messageInput?.addEventListener('input', () => {
            updateSendButton();
            autoResizeTextarea();
        });

        // Quick prompts & suggestions
        document.querySelectorAll('.action-btn, .suggestion-chip').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.dataset.prompt;
                if (prompt) {
                    elements.messageInput.value = prompt;
                    updateSendButton();
                    sendMessage();
                }
            });
        });

        // Sidebar toggle (mobile)
        elements.menuToggle?.addEventListener('click', toggleSidebar);
        
        // Sidebar collapse
        elements.sidebarCollapseBtn?.addEventListener('click', () => {
            elements.sidebar?.classList.toggle('collapsed');
        });

        // Theme toggle
        elements.themeToggle?.addEventListener('click', toggleTheme);

        // Clear chat
        elements.clearChatBtn?.addEventListener('click', clearChat);

        // Modal triggers
        elements.exportBtn?.addEventListener('click', () => openModal(elements.exportModal));
        elements.leaveChatBtn?.addEventListener('click', () => openModal(elements.leaveModal));

        // Modal close
        elements.closeExportModal?.addEventListener('click', () => closeModal(elements.exportModal));
        elements.cancelLeaveBtn?.addEventListener('click', () => closeModal(elements.leaveModal));
        elements.confirmLeaveBtn?.addEventListener('click', leaveChat);

        // Modal backdrop clicks
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.addEventListener('click', (e) => {
                const modal = e.target.closest('.modal');
                if (modal) closeModal(modal);
            });
        });

        // Export options
        document.querySelectorAll('.export-option').forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.dataset.format;
                exportChat(format);
                closeModal(elements.exportModal);
            });
        });

        // Close sidebar on mobile when clicking outside
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (elements.sidebar?.classList.contains('active')) {
                    if (!elements.sidebar.contains(e.target) && 
                        !elements.menuToggle?.contains(e.target)) {
                        elements.sidebar.classList.remove('active');
                    }
                }
            }
        });

        // Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (STATE.isSpeaking) {
                    stopSpeaking();
                }
                document.querySelectorAll('.modal.active').forEach(modal => {
                    closeModal(modal);
                });
            }
        });
    }

    // Setup code copy buttons (delegated event)
    function setupCodeCopyButtons() {
        document.addEventListener('click', (e) => {
            const copyBtn = e.target.closest('.copy-btn');
            if (copyBtn) {
                const codeId = copyBtn.dataset.codeId;
                const codeElement = document.getElementById(codeId);
                if (codeElement) {
                    navigator.clipboard.writeText(codeElement.textContent).then(() => {
                        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                        setTimeout(() => {
                            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                        }, 2000);
                    }).catch(() => {
                        showToast('Failed to copy code', 'error');
                    });
                }
            }
        });
    }

    // ============================================
    // Textarea Functions
    // ============================================

    function setupTextarea() {
        if (!elements.messageInput) return;
        autoResizeTextarea();
    }

    function autoResizeTextarea() {
        if (!elements.messageInput) return;
        elements.messageInput.style.height = 'auto';
        elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 150) + 'px';
    }

    function updateSendButton() {
        if (elements.sendBtn && elements.messageInput) {
            const hasText = elements.messageInput.value.trim().length > 0;
            elements.sendBtn.disabled = !hasText || STATE.isProcessing;
        }
    }

    // ============================================
    // Speech Synthesis - Indian Female Voice
    // ============================================

    function setupSpeechSynthesis() {
        if (!('speechSynthesis' in window)) {
            console.warn('Speech synthesis not supported');
            return;
        }

        function loadVoices() {
            const voices = speechSynthesis.getVoices();
            
            if (voices.length === 0) {
                return; // Wait for voices to load
            }

            console.log('Available voices:', voices.map(v => `${v.name} (${v.lang})`));

            // Priority list: Indian female voices first, then other quality female voices
            // Ordered by preference for clear, melodious, confident sound
            const preferredVoices = [
                // Indian English Female Voices (Top Priority)
                'Microsoft Heera',           // Windows - Indian English Female (excellent)
                'Microsoft Heera Online',    // Windows Online - Indian English Female
                'Microsoft Heera Desktop',   // Windows Desktop - Indian English Female
                'Google हिन्दी',              // Google Hindi Female
                'Lekha',                     // Apple - Indian English Female
                'Veena',                     // Apple - Indian English Female (clear & confident)
                'Priya',                     // Indian Female
                'Neerja',                    // Indian Female
                'Raveena',                   // Indian Female
                'Aditi',                     // Amazon Polly - Indian English Female
                
                // High-Quality Female Voices (Fallback - still melodious)
                'Google UK English Female',  // Clear British female
                'Google US English Female',  // Clear American female
                'Microsoft Zira',            // Windows - US Female (very clear)
                'Microsoft Zira Desktop',    // Windows Desktop
                'Microsoft Zira Online',     // Windows Online
                'Samantha',                  // Apple - US Female (natural)
                'Karen',                     // Apple - Australian Female
                'Moira',                     // Apple - Irish Female (melodious)
                'Tessa',                     // Apple - South African Female
                'Fiona',                     // Apple - Scottish Female
                'Victoria',                  // Apple - US Female
                'Allison',                   // US Female
                'Ava',                       // US Female (premium)
                'Susan',                     // UK Female
                'Kate',                      // UK Female
                
                // Generic Female (Last Resort)
                'Female',
                'Woman'
            ];

            // Find the best available voice
            STATE.selectedVoice = null;

            // First pass: Look for exact matches
            for (const preferred of preferredVoices) {
                const voice = voices.find(v => 
                    v.name.toLowerCase().includes(preferred.toLowerCase())
                );
                if (voice) {
                    STATE.selectedVoice = voice;
                    console.log('✅ Selected voice:', voice.name, `(${voice.lang})`);
                    break;
                }
            }

            // Second pass: Look for Indian English voices
            if (!STATE.selectedVoice) {
                const indianVoice = voices.find(v => 
                    (v.lang === 'en-IN' || v.lang.startsWith('en-IN') || v.lang === 'hi-IN') &&
                    (v.name.toLowerCase().includes('female') || 
                     !v.name.toLowerCase().includes('male'))
                );
                if (indianVoice) {
                    STATE.selectedVoice = indianVoice;
                    console.log('✅ Selected Indian voice:', indianVoice.name);
                }
            }

            // Third pass: Any female English voice
            if (!STATE.selectedVoice) {
                const femaleVoice = voices.find(v => 
                    v.lang.startsWith('en') &&
                    (v.name.toLowerCase().includes('female') ||
                     v.name.toLowerCase().includes('woman') ||
                     v.name.toLowerCase().includes('zira') ||
                     v.name.toLowerCase().includes('samantha') ||
                     v.name.toLowerCase().includes('karen') ||
                     v.name.toLowerCase().includes('victoria') ||
                     v.name.toLowerCase().includes('heera') ||
                     v.name.toLowerCase().includes('veena'))
                );
                if (femaleVoice) {
                    STATE.selectedVoice = femaleVoice;
                    console.log('✅ Selected female voice:', femaleVoice.name);
                }
            }

            // Fourth pass: Any English voice that's not explicitly male
            if (!STATE.selectedVoice) {
                const englishVoice = voices.find(v => 
                    v.lang.startsWith('en') &&
                    !v.name.toLowerCase().includes('male') &&
                    !v.name.toLowerCase().includes('david') &&
                    !v.name.toLowerCase().includes('james') &&
                    !v.name.toLowerCase().includes('daniel') &&
                    !v.name.toLowerCase().includes('george') &&
                    !v.name.toLowerCase().includes('rishi')
                );
                if (englishVoice) {
                    STATE.selectedVoice = englishVoice;
                    console.log('✅ Fallback voice:', englishVoice.name);
                }
            }

            // Last resort: First available voice
            if (!STATE.selectedVoice && voices.length > 0) {
                STATE.selectedVoice = voices[0];
                console.log('⚠️ Using default voice:', STATE.selectedVoice.name);
            }
        }

        // Load voices immediately if available
        if (speechSynthesis.getVoices().length) {
            loadVoices();
        }
        
        // Also listen for voices changed event (required for some browsers)
        speechSynthesis.onvoiceschanged = loadVoices;

        // Retry loading voices after a short delay (some browsers need this)
        setTimeout(() => {
            if (!STATE.selectedVoice) {
                loadVoices();
            }
        }, 100);

        setTimeout(() => {
            if (!STATE.selectedVoice) {
                loadVoices();
            }
        }, 500);
    }

    function speakText(text, button) {
        if (!('speechSynthesis' in window)) {
            showToast('Text-to-speech not supported in this browser', 'error');
            return;
        }

        // If already speaking, stop
        if (STATE.isSpeaking) {
            stopSpeaking();
            return;
        }

        // Clean text for speaking - remove code, markdown, special characters
        const cleanText = text
            // Remove code blocks
            .replace(/```[\s\S]*?```/g, '. Code block omitted. ')
            // Remove inline code
            .replace(/`([^`]+)`/g, '$1')
            // Remove bold/italic markers but keep text
            .replace(/\*\*([^*]+)\*\*/g, '$1')
            .replace(/\*([^*]+)\*/g, '$1')
            // Remove headers markers
            .replace(/#{1,6}\s/g, '')
            // Remove markdown links but keep text
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            // Remove HTML tags
            .replace(/<[^>]*>/g, '')
            // Fix HTML entities
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&nbsp;/g, ' ')
            .replace(/&quot;/g, '"')
            // Remove bullet points and list markers
            .replace(/^[\s]*[-•*]\s/gm, '')
            .replace(/^[\s]*\d+\.\s/gm, '')
            // Clean up emojis (keep some, remove others that sound weird)
            .replace(/[🎯🧠💡🔍📝✅❌⚠️🎪🎨🎭🚀💻📚🔢🔬🌟✨💎🎉]/g, '')
            // Fix multiple spaces and newlines
            .replace(/\n+/g, '. ')
            .replace(/\s+/g, ' ')
            // Remove URLs
            .replace(/https?:\/\/[^\s]+/g, 'link')
            // Clean up multiple periods
            .replace(/\.+/g, '.')
            .replace(/\.\s*\./g, '.')
            .trim();

        if (!cleanText || cleanText.length < 2) {
            showToast('No text to speak', 'warning');
            return;
        }

        // Cancel any ongoing speech
        speechSynthesis.cancel();

        // Create utterance
        STATE.currentUtterance = new SpeechSynthesisUtterance(cleanText);
        
        // Set voice if available
        if (STATE.selectedVoice) {
            STATE.currentUtterance.voice = STATE.selectedVoice;
        }
        
        // Voice settings for melodious, clear, confident sound
        // These settings are optimized for a professional female voice
        STATE.currentUtterance.rate = 0.95;      // Slightly slower for clarity
        STATE.currentUtterance.pitch = 1.1;      // Slightly higher for feminine tone
        STATE.currentUtterance.volume = 1.0;     // Full volume

        // Adjust settings based on voice type for best quality
        if (STATE.selectedVoice) {
            const voiceName = STATE.selectedVoice.name.toLowerCase();
            
            // Indian voices - adjust for natural flow
            if (voiceName.includes('heera') || 
                voiceName.includes('veena') || 
                voiceName.includes('lekha') ||
                voiceName.includes('priya') ||
                STATE.selectedVoice.lang === 'en-IN') {
                STATE.currentUtterance.rate = 0.92;   // Slightly slower for Indian accent clarity
                STATE.currentUtterance.pitch = 1.05;  // Natural pitch
                console.log('🎤 Using Indian voice settings');
            }
            // Google voices - tend to be faster
            else if (voiceName.includes('google')) {
                STATE.currentUtterance.rate = 0.9;
                STATE.currentUtterance.pitch = 1.0;
            }
            // Microsoft voices - good quality
            else if (voiceName.includes('zira') || voiceName.includes('microsoft')) {
                STATE.currentUtterance.rate = 0.95;
                STATE.currentUtterance.pitch = 1.05;
            }
            // Apple voices - natural sounding
            else if (voiceName.includes('samantha') || 
                     voiceName.includes('karen') ||
                     voiceName.includes('moira') ||
                     voiceName.includes('victoria')) {
                STATE.currentUtterance.rate = 1.0;
                STATE.currentUtterance.pitch = 1.0;
            }
        }

        STATE.currentUtterance.onstart = () => {
            STATE.isSpeaking = true;
            if (button) {
                button.innerHTML = '<i class="fas fa-stop"></i> Stop';
                button.classList.add('speaking');
            }
            console.log('🔊 Speaking started with voice:', STATE.selectedVoice?.name || 'default');
        };

        STATE.currentUtterance.onend = () => {
            STATE.isSpeaking = false;
            STATE.currentUtterance = null;
            if (button) {
                button.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
                button.classList.remove('speaking');
            }
            console.log('🔇 Speaking ended');
        };

        STATE.currentUtterance.onerror = (e) => {
            console.error('Speech synthesis error:', e);
            STATE.isSpeaking = false;
            STATE.currentUtterance = null;
            if (button) {
                button.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
                button.classList.remove('speaking');
            }
            if (e.error !== 'canceled' && e.error !== 'interrupted') {
                showToast('Error playing audio', 'error');
            }
        };

        // Chrome bug workaround: pause and resume for long texts
        STATE.currentUtterance.onpause = () => {
            console.log('Speech paused');
        };

        // Start speaking
        speechSynthesis.speak(STATE.currentUtterance);

        // Chrome bug: speech stops after ~15 seconds
        // Workaround: periodically check and resume
        const chromeBugFix = setInterval(() => {
            if (!STATE.isSpeaking) {
                clearInterval(chromeBugFix);
                return;
            }
            if (speechSynthesis.paused) {
                speechSynthesis.resume();
            }
        }, 10000);

        // Store interval reference for cleanup
        STATE.currentUtterance.chromeBugFix = chromeBugFix;
    }

    function stopSpeaking() {
        if (speechSynthesis) {
            speechSynthesis.cancel();
        }
        STATE.isSpeaking = false;
        
        // Clear chrome bug fix interval
        if (STATE.currentUtterance?.chromeBugFix) {
            clearInterval(STATE.currentUtterance.chromeBugFix);
        }
        STATE.currentUtterance = null;
        
        // Reset all listen buttons
        document.querySelectorAll('.msg-action-btn.speaking').forEach(btn => {
            btn.innerHTML = '<i class="fas fa-volume-up"></i> Listen';
            btn.classList.remove('speaking');
        });
        
        console.log('🔇 Speaking stopped');
    }

    // ============================================
    // Message Handling
    // ============================================

    async function sendMessage() {
        const text = elements.messageInput?.value?.trim();
        if (!text || STATE.isProcessing || !SESSION_ID) return;

        // Stop speaking if active
        if (STATE.isSpeaking) {
            stopSpeaking();
        }

        // Add user message
        addMessage('user', text);
        hideWelcomeScreen();

        // Clear input
        elements.messageInput.value = '';
        autoResizeTextarea();
        updateSendButton();

        // Send to API
        await sendToAPI(text);
    }

    async function sendToAPI(text) {
        STATE.isProcessing = true;
        updateSendButton();

        const typingId = addTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: SESSION_ID,
                    message: text
                })
            });

            const data = await response.json();
            removeTypingIndicator(typingId);

            if (data.success) {
                addMessage('ai', data.response);
            } else {
                addMessage('ai', "I apologize, but I encountered an error processing your request. Please try again.");
                showToast(data.error || 'Failed to get response', 'error');
            }
        } catch (error) {
            console.error('Chat error:', error);
            removeTypingIndicator(typingId);
            addMessage('ai', "I apologize, there was a network error. Please check your connection and try again.");
            showToast('Network error', 'error');
        }

        STATE.isProcessing = false;
        updateSendButton();
    }

    function addMessage(type, content, animate = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        if (!animate) messageDiv.style.animation = 'none';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = formatMessage(content);

        // Add actions for AI messages
        if (type === 'ai') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            actions.innerHTML = `
                <button class="msg-action-btn copy-msg-btn" title="Copy message">
                    <i class="fas fa-copy"></i> Copy
                </button>
                <button class="msg-action-btn listen-btn" title="Listen to message">
                    <i class="fas fa-volume-up"></i> Listen
                </button>
            `;
            contentDiv.appendChild(actions);

            // Add event listeners for the buttons
            const copyBtn = actions.querySelector('.copy-msg-btn');
            const listenBtn = actions.querySelector('.listen-btn');

            copyBtn.addEventListener('click', () => {
                navigator.clipboard.writeText(content).then(() => {
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    showToast('Message copied to clipboard', 'success');
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                    }, 2000);
                }).catch(() => {
                    showToast('Failed to copy message', 'error');
                });
            });

            listenBtn.addEventListener('click', () => {
                speakText(content, listenBtn);
            });
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        elements.chatMessages?.appendChild(messageDiv);

        // Store original content for reference
        messageDiv.dataset.content = content;

        scrollToBottom();
        STATE.messages.push({ type, content, timestamp: new Date() });
    }

    function formatMessage(content) {
        // Escape HTML first
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks with syntax highlighting header
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'code';
            const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
            return `
                <div class="code-block">
                    <div class="code-header">
                        <span class="code-lang">${language}</span>
                        <button class="copy-btn" data-code-id="${codeId}">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                    </div>
                    <div class="code-content">
                        <pre id="${codeId}">${code.trim()}</pre>
                    </div>
                </div>
            `;
        });

        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Headers
        formatted = formatted.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        formatted = formatted.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        formatted = formatted.replace(/^# (.+)$/gm, '<h2>$1</h2>');

        // Bold
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Lists (basic)
        formatted = formatted.replace(/^\- (.+)$/gm, '• $1');
        formatted = formatted.replace(/^\d+\. (.+)$/gm, '<span class="list-item">$&</span>');

        // Line breaks
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

        elements.chatMessages?.appendChild(messageDiv);
        scrollToBottom();

        return id;
    }

    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    }

    function scrollToBottom() {
        const wrapper = document.querySelector('.chat-messages-wrapper');
        if (wrapper) {
            wrapper.scrollTo({
                top: wrapper.scrollHeight,
                behavior: 'smooth'
            });
        }
    }

    function hideWelcomeScreen() {
        if (elements.welcomeScreen) {
            elements.welcomeScreen.style.display = 'none';
        }
    }

    // ============================================
    // Chat History
    // ============================================

    async function loadChatHistory() {
        if (!SESSION_ID) return;

        try {
            const response = await fetch(`/api/history/${SESSION_ID}`);
            const data = await response.json();

            if (data.success && data.messages?.length > 0) {
                hideWelcomeScreen();
                data.messages.forEach(msg => {
                    const type = msg.role === 'user' ? 'user' : 'ai';
                    addMessage(type, msg.content, false);
                });
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    function clearChat() {
        if (confirm('Clear all messages? This action cannot be undone.')) {
            // Stop any ongoing speech
            stopSpeaking();
            
            // Save welcome screen reference
            const welcomeScreen = elements.welcomeScreen;
            
            // Clear messages container
            elements.chatMessages.innerHTML = '';
            
            // Restore and show welcome screen
            if (welcomeScreen) {
                welcomeScreen.style.display = 'block';
                elements.chatMessages.appendChild(welcomeScreen);
            }
            
            // Clear state
            STATE.messages = [];
            
            showToast('Chat cleared successfully', 'success');
        }
    }

    // ============================================
    // Leave Chat
    // ============================================

    async function leaveChat() {
        try {
            // Stop any speaking
            stopSpeaking();
            
            // Cleanup session on server
            await fetch(`/api/session/${SESSION_ID}/cleanup`, {
                method: 'POST'
            });
            
            showToast('Session deleted', 'success');
            
            // Redirect to home after short delay
            setTimeout(() => {
                window.location.href = '/';
            }, 500);
        } catch (error) {
            console.error('Leave chat error:', error);
            showToast('Error leaving chat', 'error');
            // Still redirect even on error
            setTimeout(() => {
                window.location.href = '/';
            }, 1000);
        }
    }

    // ============================================
    // Theme
    // ============================================

    function loadTheme() {
        const savedTheme = localStorage.getItem('vectorplex-theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const newTheme = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('vectorplex-theme', newTheme);
        updateThemeIcon(newTheme);
        showToast(`Switched to ${newTheme} mode`, 'info');
    }

    function updateThemeIcon(theme) {
        const icon = elements.themeToggle?.querySelector('i');
        if (icon) {
            icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    // ============================================
    // Sidebar
    // ============================================

    function toggleSidebar() {
        elements.sidebar?.classList.toggle('active');
    }

    // ============================================
    // Modals
    // ============================================

    function openModal(modal) {
        if (modal) {
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    function closeModal(modal) {
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    // ============================================
    // Export Functions
    // ============================================

    function exportChat(format) {
        if (STATE.messages.length === 0) {
            showToast('No messages to export', 'warning');
            return;
        }

        const filename = `vectorplex-chat-${SESSION_ID?.slice(0, 8) || 'export'}`;

        switch (format) {
            case 'pdf':
                exportToPDF(filename);
                break;
            case 'json':
                downloadFile(
                    JSON.stringify({ 
                        exportDate: new Date().toISOString(), 
                        sessionId: SESSION_ID, 
                        videoTitle: window.VIDEO_TITLE || 'Unknown',
                        messages: STATE.messages 
                    }, null, 2),
                    `${filename}.json`,
                    'application/json'
                );
                showToast('Exported as JSON', 'success');
                break;
            case 'md':
                downloadFile(generateMarkdown(), `${filename}.md`, 'text/markdown');
                showToast('Exported as Markdown', 'success');
                break;
            default:
                downloadFile(generatePlainText(), `${filename}.txt`, 'text/plain');
                showToast('Exported as Text', 'success');
        }
    }

    function generateMarkdown() {
        let md = `# VectorPlex Chat Export\n\n`;
        md += `**Video:** ${window.VIDEO_TITLE || 'Unknown'}\n`;
        md += `**Session:** ${SESSION_ID}\n`;
        md += `**Date:** ${new Date().toLocaleString()}\n\n`;
        md += `---\n\n`;

        STATE.messages.forEach(msg => {
            const role = msg.type === 'user' ? '**You**' : '**VectorPlex**';
            md += `${role}:\n\n${msg.content}\n\n---\n\n`;
        });

        md += `\n---\n*Exported from VectorPlex Demo Model*\n`;
        md += `*AI/ML: Sangam Gautam | Frontend: Sushil Yadav*`;
        return md;
    }

    function generatePlainText() {
        let txt = `VECTORPLEX CHAT EXPORT\n`;
        txt += `${'='.repeat(50)}\n\n`;
        txt += `Video: ${window.VIDEO_TITLE || 'Unknown'}\n`;
        txt += `Session: ${SESSION_ID}\n`;
        txt += `Date: ${new Date().toLocaleString()}\n\n`;
        txt += `${'='.repeat(50)}\n\n`;

        STATE.messages.forEach(msg => {
            const role = msg.type === 'user' ? 'YOU' : 'VECTORPLEX';
            txt += `[${role}]\n`;
            txt += `${msg.content}\n\n`;
            txt += `${'-'.repeat(30)}\n\n`;
        });

        txt += `\n${'='.repeat(50)}\n`;
        txt += `Exported from VectorPlex Demo Model\n`;
        txt += `AI/ML: Sangam Gautam | Frontend: Sushil Yadav`;
        return txt;
    }

    function exportToPDF(filename) {
        // Check if jsPDF is available
        if (typeof window.jspdf === 'undefined') {
            showToast('PDF export not available', 'error');
            return;
        }

        try {
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 20;
            const maxWidth = pageWidth - (margin * 2);
            let y = margin;

            // Title
            doc.setFontSize(20);
            doc.setTextColor(139, 92, 246); // Purple
            doc.text('VectorPlex Chat Export', margin, y);
            y += 12;

            // Metadata
            doc.setFontSize(10);
            doc.setTextColor(100, 100, 100);
            doc.text(`Video: ${window.VIDEO_TITLE || 'Unknown'}`, margin, y);
            y += 6;
            doc.text(`Session: ${SESSION_ID}`, margin, y);
            y += 6;
            doc.text(`Date: ${new Date().toLocaleString()}`, margin, y);
            y += 10;

            // Divider line
            doc.setDrawColor(139, 92, 246);
            doc.setLineWidth(0.5);
            doc.line(margin, y, pageWidth - margin, y);
            y += 10;

            // Messages
            STATE.messages.forEach((msg, index) => {
                // Check if we need a new page
                if (y > pageHeight - 40) {
                    doc.addPage();
                    y = margin;
                }

                // Role header
                doc.setFontSize(11);
                doc.setTextColor(msg.type === 'user' ? 59 : 139, msg.type === 'user' ? 130 : 92, 246);
                doc.setFont(undefined, 'bold');
                doc.text(msg.type === 'user' ? 'You:' : 'VectorPlex:', margin, y);
                y += 6;

                // Message content
                doc.setFontSize(10);
                doc.setTextColor(50, 50, 50);
                doc.setFont(undefined, 'normal');

                // Clean content for PDF
                const cleanContent = msg.content
                    .replace(/```[\s\S]*?```/g, '[Code Block]')
                    .replace(/`([^`]+)`/g, '$1')
                    .replace(/\*\*([^*]+)\*\*/g, '$1')
                    .replace(/\*([^*]+)\*/g, '$1')
                    .replace(/#{1,6}\s/g, '');

                const lines = doc.splitTextToSize(cleanContent, maxWidth);
                
                lines.forEach(line => {
                    if (y > pageHeight - 20) {
                        doc.addPage();
                        y = margin;
                    }
                    doc.text(line, margin, y);
                    y += 5;
                });

                y += 8;
            });

            // Footer on each page
            const pageCount = doc.internal.getNumberOfPages();
            for (let i = 1; i <= pageCount; i++) {
                doc.setPage(i);
                doc.setFontSize(8);
                doc.setTextColor(150, 150, 150);
                doc.text('VectorPlex Demo Model | AI/ML: Sangam Gautam', margin, pageHeight - 10);
                doc.text(`Page ${i} of ${pageCount}`, pageWidth - margin - 20, pageHeight - 10);
            }

            // Save the PDF
            doc.save(`${filename}.pdf`);
            showToast('Exported as PDF', 'success');
        } catch (error) {
            console.error('PDF export error:', error);
            showToast('Failed to export PDF', 'error');
        }
    }

    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    // ============================================
    // Toast Notifications
    // ============================================

    function showToast(message, type = 'info') {
        const container = elements.toastContainer;
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };

        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
        `;

        container.appendChild(toast);

        // Auto remove after 4 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, 300);
        }, 4000);
    }

    // ============================================
    // Initialize the application
    // ============================================

    init();
});