/**
 * VectorPlex V2 - Algolix AI
 * Main JavaScript Application
 * Handles UI interactions and API communication with FastAPI backend
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        PROCESS_PLAYLIST: '/api/process-playlist',
        CHAT: '/api/chat',
        STATUS: '/api/status',
        END_SESSION: '/api/end-session',
        HEALTH: '/api/health'
    },
    POLLING_INTERVAL: 2000,
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000
};

// ============================================
// State Management
// ============================================
const AppState = {
    sessionId: null,
    isProcessing: false,
    isChatReady: false,
    isWaitingForResponse: false,
    chatHistory: [],
    currentPhase: 'input', // 'input', 'processing', 'chat'
    processingProgress: 0
};

// ============================================
// DOM Elements
// ============================================
const DOM = {
    // Navigation
    navbar: document.querySelector('.navbar'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    mobileMenu: document.getElementById('mobileMenu'),
    navLinks: document.querySelectorAll('.nav-link, .mobile-link'),

    // Input Phase
    inputPhase: document.getElementById('inputPhase'),
    playlistInput: document.getElementById('playlistInput'),
    clearBtn: document.getElementById('clearBtn'),
    processBtn: document.getElementById('processBtn'),

    // Processing Phase
    processingPhase: document.getElementById('processingPhase'),
    processingStatus: document.getElementById('processingStatus'),
    progressFill: document.getElementById('progressFill'),
    processingSteps: {
        step1: document.getElementById('pStep1'),
        step2: document.getElementById('pStep2'),
        step3: document.getElementById('pStep3'),
        step4: document.getElementById('pStep4'),
        step5: document.getElementById('pStep5')
    },

    // Chat Phase
    chatPhase: document.getElementById('chatPhase'),
    chatMessages: document.getElementById('chatMessages'),
    chatInput: document.getElementById('chatInput'),
    sendBtn: document.getElementById('sendBtn'),
    suggestions: document.getElementById('suggestions'),
    suggestionChips: document.querySelectorAll('.suggestion-chip'),
    newSessionBtn: document.getElementById('newSessionBtn'),
    endSessionBtn: document.getElementById('endSessionBtn'),

    // Toast
    toastContainer: document.getElementById('toastContainer')
};

// ============================================
// Utility Functions
// ============================================
const Utils = {
    /**
     * Generate unique session ID
     */
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    },

    /**
     * Validate YouTube playlist URL
     */
    isValidPlaylistUrl(url) {
        const patterns = [
            /^(https?:\/\/)?(www\.)?youtube\.com\/playlist\?list=[\w-]+/,
            /^(https?:\/\/)?(www\.)?youtube\.com\/watch\?v=[\w-]+&list=[\w-]+/,
            /^(https?:\/\/)?(www\.)?youtu\.be\/[\w-]+\?list=[\w-]+/
        ];
        return patterns.some(pattern => pattern.test(url));
    },

    /**
     * Format timestamp
     */
    formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        }).format(date);
    },

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Parse markdown-like formatting
     */
    parseMarkdown(text) {
        return text
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n/g, '<br>');
    },

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Sleep function for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

// ============================================
// API Service
// ============================================
const API = {
    /**
     * Make API request with retry logic
     */
    async request(endpoint, options = {}, retries = CONFIG.MAX_RETRIES) {
        const url = CONFIG.API_BASE_URL + endpoint;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const fetchOptions = { ...defaultOptions, ...options };

        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                const response = await fetch(url, fetchOptions);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}`);
                }

                return await response.json();
            } catch (error) {
                console.error(`API attempt ${attempt} failed:`, error);
                
                if (attempt === retries) {
                    throw error;
                }
                
                await Utils.sleep(CONFIG.RETRY_DELAY * attempt);
            }
        }
    },

    /**
     * Check API health
     */
    async checkHealth() {
        try {
            const response = await this.request(CONFIG.ENDPOINTS.HEALTH);
            return response.status === 'healthy';
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    },

    /**
     * Start playlist processing
     */
    async processPlaylist(playlistUrl, sessionId) {
        return this.request(CONFIG.ENDPOINTS.PROCESS_PLAYLIST, {
            method: 'POST',
            body: JSON.stringify({
                playlist_url: playlistUrl,
                session_id: sessionId
            })
        });
    },

    /**
     * Get processing status
     */
    async getStatus(sessionId) {
        return this.request(`${CONFIG.ENDPOINTS.STATUS}/${sessionId}`);
    },

    /**
     * Send chat message
     */
    async sendMessage(sessionId, message) {
        return this.request(CONFIG.ENDPOINTS.CHAT, {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });
    },

    /**
     * End session and cleanup
     */
    async endSession(sessionId) {
        return this.request(`${CONFIG.ENDPOINTS.END_SESSION}/${sessionId}`, {
            method: 'DELETE'
        });
    }
};

// ============================================
// Toast Notifications
// ============================================
const Toast = {
    /**
     * Show toast notification
     */
    show(message, type = 'info', duration = 5000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };

        toast.innerHTML = `
            <i class="fas ${icons[type]} toast-icon"></i>
            <span class="toast-message">${Utils.escapeHtml(message)}</span>
            <button class="toast-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => this.remove(toast));

        DOM.toastContainer.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => this.remove(toast), duration);
        }

        return toast;
    },

    /**
     * Remove toast
     */
    remove(toast) {
        toast.style.animation = 'toastSlide 0.3s ease reverse forwards';
        setTimeout(() => toast.remove(), 300);
    },

    success(message) { return this.show(message, 'success'); },
    error(message) { return this.show(message, 'error'); },
    warning(message) { return this.show(message, 'warning'); },
    info(message) { return this.show(message, 'info'); }
};

// ============================================
// UI Controller
// ============================================
const UI = {
    /**
     * Initialize UI
     */
    init() {
        this.setupNavigation();
        this.setupInputPhase();
        this.setupChatPhase();
        this.setupBeforeUnload();
        this.checkAPIConnection();
    },

    /**
     * Setup navigation
     */
    setupNavigation() {
        // Scroll effect
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                DOM.navbar.classList.add('scrolled');
            } else {
                DOM.navbar.classList.remove('scrolled');
            }
        });

        // Mobile menu
        DOM.mobileMenuBtn?.addEventListener('click', () => {
            DOM.mobileMenuBtn.classList.toggle('active');
            DOM.mobileMenu.classList.toggle('active');
            DOM.mobileMenu.style.display = DOM.mobileMenu.classList.contains('active') ? 'flex' : 'none';
        });

        // Smooth scroll for nav links
        DOM.navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href.startsWith('#')) {
                    e.preventDefault();
                    const target = document.querySelector(href);
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                        // Close mobile menu
                        DOM.mobileMenuBtn?.classList.remove('active');
                        DOM.mobileMenu?.classList.remove('active');
                        if (DOM.mobileMenu) DOM.mobileMenu.style.display = 'none';
                    }
                }
            });
        });
    },

    /**
     * Setup input phase
     */
    setupInputPhase() {
        // Clear button
        DOM.clearBtn?.addEventListener('click', () => {
            DOM.playlistInput.value = '';
            DOM.playlistInput.focus();
        });

        // Process button
        DOM.processBtn?.addEventListener('click', () => {
            this.handleProcessPlaylist();
        });

        // Enter key to submit
        DOM.playlistInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleProcessPlaylist();
            }
        });

        // Input validation on type
        DOM.playlistInput?.addEventListener('input', Utils.debounce(() => {
            const isValid = Utils.isValidPlaylistUrl(DOM.playlistInput.value);
            DOM.processBtn.disabled = !DOM.playlistInput.value.trim();
        }, 300));
    },

    /**
     * Setup chat phase
     */
    setupChatPhase() {
        // Send button
        DOM.sendBtn?.addEventListener('click', () => {
            this.handleSendMessage();
        });

        // Enter to send (Shift+Enter for new line)
        DOM.chatInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        // Auto-resize textarea
        DOM.chatInput?.addEventListener('input', () => {
            DOM.chatInput.style.height = 'auto';
            DOM.chatInput.style.height = Math.min(DOM.chatInput.scrollHeight, 150) + 'px';
        });

        // Suggestion chips
        DOM.suggestionChips?.forEach(chip => {
            chip.addEventListener('click', () => {
                DOM.chatInput.value = chip.textContent;
                this.handleSendMessage();
            });
        });

        // New session button
        DOM.newSessionBtn?.addEventListener('click', () => {
            this.handleNewSession();
        });

        // End session button
        DOM.endSessionBtn?.addEventListener('click', () => {
            this.handleEndSession();
        });
    },

    /**
     * Setup before unload warning
     */
    setupBeforeUnload() {
        window.addEventListener('beforeunload', async (e) => {
            if (AppState.sessionId && AppState.isChatReady) {
                // Try to cleanup session
                try {
                    await API.endSession(AppState.sessionId);
                } catch (error) {
                    console.error('Failed to cleanup session:', error);
                }
            }
        });
    },

    /**
     * Check API connection
     */
    async checkAPIConnection() {
        const isHealthy = await API.checkHealth();
        if (!isHealthy) {
            Toast.warning('Backend server is not responding. Please ensure the server is running.');
        }
    },

    /**
     * Switch between phases
     */
    switchPhase(phase) {
        AppState.currentPhase = phase;

        // Hide all phases
        DOM.inputPhase?.classList.add('hidden');
        DOM.processingPhase?.classList.add('hidden');
        DOM.chatPhase?.classList.add('hidden');

        // Show selected phase
        switch (phase) {
            case 'input':
                DOM.inputPhase?.classList.remove('hidden');
                break;
            case 'processing':
                DOM.processingPhase?.classList.remove('hidden');
                break;
            case 'chat':
                DOM.chatPhase?.classList.remove('hidden');
                break;
        }

        // Scroll to demo section
        const demoSection = document.getElementById('demo');
        demoSection?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },

    /**
     * Handle playlist processing
     */
    async handleProcessPlaylist() {
        const playlistUrl = DOM.playlistInput.value.trim();

        // Validate URL
        if (!playlistUrl) {
            Toast.warning('Please enter a playlist URL');
            return;
        }

        if (!Utils.isValidPlaylistUrl(playlistUrl)) {
            Toast.error('Please enter a valid YouTube playlist URL');
            return;
        }

        // Generate session ID
        AppState.sessionId = Utils.generateSessionId();
        AppState.isProcessing = true;

        // Switch to processing phase
        this.switchPhase('processing');
        this.resetProcessingUI();

        try {
            // Start processing
            const response = await API.processPlaylist(playlistUrl, AppState.sessionId);
            
            if (response.status === 'processing' || response.status === 'started') {
                Toast.info('Processing started! This may take a few minutes...');
                this.startStatusPolling();
            } else {
                throw new Error(response.message || 'Failed to start processing');
            }
        } catch (error) {
            console.error('Processing error:', error);
            Toast.error(`Failed to process playlist: ${error.message}`);
            this.switchPhase('input');
            AppState.isProcessing = false;
        }
    },

    /**
     * Reset processing UI
     */
    resetProcessingUI() {
        DOM.progressFill.style.width = '0%';
        DOM.processingStatus.textContent = 'Initializing...';
        
        Object.values(DOM.processingSteps).forEach(step => {
            step.classList.remove('active', 'completed');
        });
    },

    /**
     * Update processing progress
     */
    updateProcessingProgress(status) {
        const { step, progress, message } = status;
        
        DOM.processingStatus.textContent = message || 'Processing...';
        DOM.progressFill.style.width = `${progress}%`;

        // Update steps
        const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];
        const currentStepIndex = step - 1;

        steps.forEach((stepKey, index) => {
            const stepEl = DOM.processingSteps[stepKey];
            if (index < currentStepIndex) {
                stepEl.classList.remove('active');
                stepEl.classList.add('completed');
            } else if (index === currentStepIndex) {
                stepEl.classList.add('active');
                stepEl.classList.remove('completed');
            } else {
                stepEl.classList.remove('active', 'completed');
            }
        });
    },

    /**
     * Start status polling
     */
    startStatusPolling() {
        const pollStatus = async () => {
            if (!AppState.isProcessing) return;

            try {
                const status = await API.getStatus(AppState.sessionId);
                
                this.updateProcessingProgress(status);

                if (status.status === 'completed') {
                    AppState.isProcessing = false;
                    AppState.isChatReady = true;
                    Toast.success('Processing complete! You can now chat with your content.');
                    this.initializeChat();
                } else if (status.status === 'error') {
                    AppState.isProcessing = false;
                    Toast.error(`Processing failed: ${status.message}`);
                    this.switchPhase('input');
                } else {
                    // Continue polling
                    setTimeout(pollStatus, CONFIG.POLLING_INTERVAL);
                }
            } catch (error) {
                console.error('Status polling error:', error);
                // Retry polling
                setTimeout(pollStatus, CONFIG.POLLING_INTERVAL * 2);
            }
        };

        pollStatus();
    },

    /**
     * Initialize chat
     */
    initializeChat() {
        this.switchPhase('chat');
        this.clearChatMessages();
        
        // Add welcome message
        this.addMessage(
            "Hello! I've successfully processed your playlist and created a knowledge base from the video content. Feel free to ask me anything about the videos!",
            'ai'
        );

        // Show suggestions
        DOM.suggestions?.classList.remove('hidden');
        DOM.chatInput?.focus();
    },

    /**
     * Handle sending message
     */
    async handleSendMessage() {
        const message = DOM.chatInput.value.trim();

        if (!message || AppState.isWaitingForResponse) return;

        if (!AppState.isChatReady) {
            Toast.warning('Please process a playlist first');
            return;
        }

        // Add user message
        this.addMessage(message, 'user');
        DOM.chatInput.value = '';
        DOM.chatInput.style.height = 'auto';

        // Hide suggestions after first message
        DOM.suggestions?.classList.add('hidden');

        // Show loading
        AppState.isWaitingForResponse = true;
        const loadingMessage = this.addLoadingMessage();

        try {
            const response = await API.sendMessage(AppState.sessionId, message);
            
            // Remove loading message
            loadingMessage.remove();
            
            // Add AI response
            this.addMessage(response.response, 'ai');
        } catch (error) {
            console.error('Chat error:', error);
            loadingMessage.remove();
            this.addMessage(
                "I'm sorry, I encountered an error processing your request. Please try again.",
                'ai'
            );
            Toast.error('Failed to get response');
        } finally {
            AppState.isWaitingForResponse = false;
        }
    },

    /**
     * Add message to chat
     */
    addMessage(content, type) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}`;
        
        const avatar = type === 'ai' 
            ? '<i class="fas fa-robot"></i>' 
            : '<i class="fas fa-user"></i>';

        const formattedContent = type === 'ai' 
            ? Utils.parseMarkdown(Utils.escapeHtml(content))
            : Utils.escapeHtml(content);

        messageEl.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-bubble">${formattedContent}</div>
                <div class="message-time">${Utils.formatTime(new Date())}</div>
            </div>
        `;

        DOM.chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        // Store in history
        AppState.chatHistory.push({ type, content, timestamp: new Date() });

        return messageEl;
    },

    /**
     * Add loading message
     */
    addLoadingMessage() {
        const messageEl = document.createElement('div');
        messageEl.className = 'message ai loading';
        
        messageEl.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;

        DOM.chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        return messageEl;
    },

    /**
     * Clear chat messages
     */
    clearChatMessages() {
        DOM.chatMessages.innerHTML = '';
        AppState.chatHistory = [];
    },

    /**
     * Scroll chat to bottom
     */
    scrollToBottom() {
        DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    },

    /**
     * Handle new session
     */
    async handleNewSession() {
        if (AppState.sessionId) {
            try {
                await API.endSession(AppState.sessionId);
            } catch (error) {
                console.error('Failed to end previous session:', error);
            }
        }

        // Reset state
        AppState.sessionId = null;
        AppState.isProcessing = false;
        AppState.isChatReady = false;
        AppState.chatHistory = [];

        // Reset UI
        DOM.playlistInput.value = '';
        this.clearChatMessages();
        this.switchPhase('input');

        Toast.info('Ready for a new session');
    },

    /**
     * Handle end session
     */
    async handleEndSession() {
        if (!AppState.sessionId) {
            this.switchPhase('input');
            return;
        }

        try {
            await API.endSession(AppState.sessionId);
            Toast.success('Session ended. Your data has been deleted.');
        } catch (error) {
            console.error('Failed to end session:', error);
            Toast.warning('Session ended locally. Server cleanup may have failed.');
        }

        // Reset state
        AppState.sessionId = null;
        AppState.isProcessing = false;
        AppState.isChatReady = false;
        AppState.chatHistory = [];

        // Reset UI
        DOM.playlistInput.value = '';
        this.clearChatMessages();
        this.switchPhase('input');
    }
};

// ============================================
// Animations & Effects
// ============================================
const Effects = {
    /**
     * Initialize intersection observer for animations
     */
    initScrollAnimations() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        // Observe elements
        document.querySelectorAll('.feature-card, .pipeline-step, .tech-item').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        });
    },

    /**
     * Add animate-in class handler
     */
    setupAnimateIn() {
        const style = document.createElement('style');
        style.textContent = `
            .animate-in {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
        `;
        document.head.appendChild(style);
    }
};

// ============================================
// Initialize Application
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 VectorPlex V2 - Initializing...');
    
    // Initialize UI
    UI.init();
    
    // Initialize effects
    Effects.setupAnimateIn();
    Effects.initScrollAnimations();
    
    console.log('✅ VectorPlex V2 - Ready!');
});

// ============================================
// Export for debugging (optional)
// ============================================
window.VectorPlex = {
    state: AppState,
    api: API,
    ui: UI,
    toast: Toast,
    utils: Utils
};