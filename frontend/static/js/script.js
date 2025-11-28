// VectorPlex V2 - Main Script
document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const mobileMenu = document.getElementById("mobileMenu");
    const navbar = document.querySelector(".navbar");
    const videoUrlInput = document.getElementById("videoUrl");
    const clearInputBtn = document.getElementById("clearInput");
    const processBtn = document.getElementById("processBtn");
    const exampleChips = document.querySelectorAll(".example-chip");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const cancelLoadingBtn = document.getElementById("cancelLoadingBtn");
    const toastContainer = document.getElementById("toastContainer");

    // Loading elements
    const loadingProgressFill = document.getElementById("loadingProgressFill");
    const loadingPercentage = document.getElementById("loadingPercentage");
    const loadingSubtitle = document.getElementById("loadingSubtitle");
    const loadingTip = document.getElementById("loadingTip");
    const loadingSteps = {
        download: document.getElementById("stepDownload"),
        transcribe: document.getElementById("stepTranscribe"),
        embed: document.getElementById("stepEmbed"),
        ready: document.getElementById("stepReady"),
    };

    let currentSessionId = null;
    let processingCancelled = false;
    let pollingInterval = null;
    let tipInterval = null;

    // Tips for loading screen
    const tips = [
        "VectorPlex can process videos in 99+ languages!",
        "Ask follow-up questions to dive deeper into the content.",
        "You can export your chat history for later reference.",
        "Timestamps in responses link directly to video moments.",
        "ChromaDB enables semantic search through your video.",
        "Groq's LPU delivers responses in milliseconds!",
    ];

    // Initialize
    init();

    function init() {
        setupEventListeners();
        setupScrollEffects();
    }

    function setupEventListeners() {
        // Mobile Menu Toggle
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener("click", () => {
                mobileMenu.classList.toggle("active");
                mobileMenuBtn.classList.toggle("active");
            });
        }

        // Close mobile menu on link click
        document.querySelectorAll(".mobile-nav-link").forEach((link) => {
            link.addEventListener("click", () => {
                mobileMenu.classList.remove("active");
                mobileMenuBtn.classList.remove("active");
            });
        });

        // Video URL Input
        if (videoUrlInput) {
            videoUrlInput.addEventListener("input", () => {
                if (clearInputBtn) {
                    clearInputBtn.style.display = videoUrlInput.value ? "block" : "none";
                }
            });

            // Enter key to process
            videoUrlInput.addEventListener("keypress", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    startProcessing();
                }
            });
        }

        // Clear input button
        if (clearInputBtn) {
            clearInputBtn.addEventListener("click", () => {
                videoUrlInput.value = "";
                clearInputBtn.style.display = "none";
                videoUrlInput.focus();
            });
        }

        // Example Chips
        exampleChips.forEach((chip) => {
            chip.addEventListener("click", () => {
                const url = chip.dataset.url;
                if (videoUrlInput && url) {
                    videoUrlInput.value = url;
                    if (clearInputBtn) {
                        clearInputBtn.style.display = "block";
                    }
                }
            });
        });

        // Process Button
        if (processBtn) {
            processBtn.addEventListener("click", startProcessing);
        }

        // Cancel Loading
        if (cancelLoadingBtn) {
            cancelLoadingBtn.addEventListener("click", cancelProcessing);
        }

        // Smooth scroll for nav links
        document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
            anchor.addEventListener("click", function (e) {
                e.preventDefault();
                const targetId = this.getAttribute("href");
                const target = document.querySelector(targetId);
                if (target) {
                    const navHeight = navbar ? navbar.offsetHeight : 0;
                    const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight - 20;
                    window.scrollTo({
                        top: targetPosition,
                        behavior: "smooth"
                    });
                    // Close mobile menu
                    if (mobileMenu) mobileMenu.classList.remove("active");
                    if (mobileMenuBtn) mobileMenuBtn.classList.remove("active");
                }
            });
        });
    }

    function setupScrollEffects() {
        let lastScroll = 0;
        window.addEventListener("scroll", () => {
            const currentScroll = window.pageYOffset;
            if (navbar) {
                if (currentScroll > 100) {
                    navbar.style.background = "rgba(10, 10, 15, 0.95)";
                    navbar.style.boxShadow = "0 4px 20px rgba(0, 0, 0, 0.3)";
                } else {
                    navbar.style.background = "rgba(10, 10, 15, 0.8)";
                    navbar.style.boxShadow = "none";
                }
            }
            lastScroll = currentScroll;
        });
    }

    // URL Validation
    function isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    // Start Processing
    function startProcessing() {
        const url = videoUrlInput ? videoUrlInput.value.trim() : "";
        
        if (!url) {
            showToast("Please enter a video URL", "error");
            if (videoUrlInput) videoUrlInput.focus();
            return;
        }
        
        if (!isValidUrl(url)) {
            showToast("Please enter a valid URL", "error");
            return;
        }

        processingCancelled = false;
        showLoading();
        updateLoadingStep("download", "active", "Connecting...");
        updateProgress(5);

        // Make API call to start processing
        fetch("/api/process", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ video_url: url }),
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                throw new Error(data.error || "Failed to start processing");
            }
            currentSessionId = data.session_id;
            console.log("Session created:", currentSessionId);
            // Start polling for progress
            pollProgress();
        })
        .catch(error => {
            console.error("Processing error:", error);
            hideLoading();
            showToast(error.message || "Failed to process video", "error");
        });
    }

    // Poll Progress
    function pollProgress() {
        if (processingCancelled || !currentSessionId) return;

        pollingInterval = setInterval(() => {
            if (processingCancelled) {
                clearInterval(pollingInterval);
                return;
            }

            fetch(`/api/status/${currentSessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const progress = data.progress || 0;
                        const status = data.status || "processing";
                        const message = data.message || "Processing...";

                        updateProgress(progress);
                        updateLoadingSubtitle(message);

                        // Update steps based on status
                        if (status === "downloading" || progress >= 10) {
                            updateLoadingStep("download", "active", "Downloading...");
                        }
                        if (progress >= 25) {
                            updateLoadingStep("download", "completed", "Complete");
                            updateLoadingStep("transcribe", "active", "In progress...");
                        }
                        if (status === "transcribing" || progress >= 50) {
                            updateLoadingStep("transcribe", "active", "Transcribing...");
                        }
                        if (progress >= 60) {
                            updateLoadingStep("transcribe", "completed", "Complete");
                            updateLoadingStep("embed", "active", "In progress...");
                        }
                        if (status === "embedding" || progress >= 75) {
                            updateLoadingStep("embed", "active", "Embedding...");
                        }
                        if (progress >= 95) {
                            updateLoadingStep("embed", "completed", "Complete");
                            updateLoadingStep("ready", "active", "Finalizing...");
                        }

                        // Check if ready
                        if (data.ready || status === "ready" || progress >= 100) {
                            updateLoadingStep("ready", "completed", "Ready!");
                            updateProgress(100);
                            clearInterval(pollingInterval);

                            // Redirect to chat page
                            setTimeout(() => {
                                window.location.href = `/chat/${currentSessionId}`;
                            }, 1000);
                        }

                        // Check for error
                        if (status === "error" || data.error) {
                            clearInterval(pollingInterval);
                            hideLoading();
                            showToast(data.error || "Processing failed", "error");
                        }
                    }
                })
                .catch(error => {
                    console.error("Progress poll error:", error);
                });
        }, 1500);
    }

    // Cancel Processing
    function cancelProcessing() {
        processingCancelled = true;
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
        if (tipInterval) {
            clearInterval(tipInterval);
            tipInterval = null;
        }
        hideLoading();
        showToast("Processing cancelled", "error");
    }

    // Update Progress
    function updateProgress(percent) {
        if (loadingProgressFill) {
            loadingProgressFill.style.width = `${percent}%`;
        }
        if (loadingPercentage) {
            loadingPercentage.textContent = `${Math.round(percent)}%`;
        }
    }

    // Update Loading Step
    function updateLoadingStep(step, status, statusText) {
        const stepElement = loadingSteps[step];
        if (!stepElement) return;

        // Remove all status classes
        stepElement.classList.remove("active", "completed");

        // Add new status class
        if (status === "active" || status === "completed") {
            stepElement.classList.add(status);
        }

        // Update status text
        const statusSpan = stepElement.querySelector(".step-status");
        if (statusSpan) {
            statusSpan.textContent = statusText;
        }
    }

    // Update Loading Subtitle
    function updateLoadingSubtitle(text) {
        if (loadingSubtitle) {
            loadingSubtitle.textContent = text;
        }
    }

    // Show Loading
    function showLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.add("active");
            document.body.style.overflow = "hidden";
            
            // Reset all steps
            Object.keys(loadingSteps).forEach((step) => {
                updateLoadingStep(step, "", "Waiting...");
            });
            updateProgress(0);
            rotateTips();
        }
    }

    // Hide Loading
    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.remove("active");
            document.body.style.overflow = "";
        }
        if (tipInterval) {
            clearInterval(tipInterval);
            tipInterval = null;
        }
    }

    // Rotate Tips
    function rotateTips() {
        let tipIndex = 0;
        if (loadingTip) {
            loadingTip.textContent = tips[tipIndex];
        }
        
        tipInterval = setInterval(() => {
            if (loadingTip && loadingOverlay && loadingOverlay.classList.contains("active")) {
                tipIndex = (tipIndex + 1) % tips.length;
                loadingTip.style.opacity = "0";
                setTimeout(() => {
                    loadingTip.textContent = tips[tipIndex];
                    loadingTip.style.opacity = "1";
                }, 300);
            }
        }, 4000);
    }

    // Show Toast
    function showToast(message, type = "success") {
        if (!toastContainer) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas ${type === "success" ? "fa-check-circle" : "fa-exclamation-circle"} toast-icon"></i>
            <span class="toast-message">${message}</span>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});