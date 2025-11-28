"""
VectorPlex V2 - Flask Application
=================================
Main Flask server that integrates with pipeline.py
By Sangam Gautam & Sushil Yadav

Uses importlib.util for clean external module loading
"""

import os
import sys
import json
import time
import uuid
import threading
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS




class Config:
    """Application configuration"""
    
    # Paths
    CURRENT_DIR = Path(__file__).parent.resolve()
    PIPELINE_PATH = Path(r"D:\sangam\VectorPlex V2\VectorPlex-V2-DEMO\backend\pipeline.py")
    
    # Alternative paths to try
    ALTERNATIVE_PATHS = [
        Path(__file__).parent / "pipeline.py",
        Path(__file__).parent.parent / "backend" / "pipeline.py",
        Path.cwd() / "pipeline.py",
        Path.cwd() / "backend" / "pipeline.py",
    ]
    
    # Flask settings
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'vectorplex-v2-secret-key-2025')
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    
    # Processing settings
    STREAM_TIMEOUT = 300  # 5 minutes
    POLL_INTERVAL = 0.5  # seconds


# =============================================================================
# EXTERNAL MODULE LOADER
# =============================================================================

class ModuleLoader:
    """
    Utility class for loading Python modules from external file paths.
    Uses importlib.util - no Pylance warnings!
    """
    
    @staticmethod
    def load_from_path(module_name: str, file_path: Path) -> Optional[Any]:
        """
        Load a Python module from an absolute file path.
        
        Args:
            module_name: Name to assign to the module
            file_path: Path to the .py file
            
        Returns:
            Loaded module or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ Module file not found: {file_path}")
            return None
        
        if not file_path.suffix == '.py':
            print(f"❌ Not a Python file: {file_path}")
            return None
        
        try:
            # Create module spec from file location
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            
            if spec is None or spec.loader is None:
                print(f"❌ Cannot create spec for: {file_path}")
                return None
            
            # Create module from spec
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules so imports within the module work
            sys.modules[module_name] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            print(f"✅ Successfully loaded module: {module_name}")
            print(f"   From: {file_path}")
            
            return module
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}:")
            print(f"   Line {e.lineno}: {e.msg}")
            return None
            
        except Exception as e:
            print(f"❌ Failed to load module: {type(e).__name__}: {e}")
            return None
    
    @staticmethod
    def find_and_load(module_name: str, paths: List[Path]) -> Optional[Any]:
        """
        Try multiple paths to find and load a module.
        
        Args:
            module_name: Name to assign to the module
            paths: List of paths to try
            
        Returns:
            Loaded module or None if all paths failed
        """
        for path in paths:
            if path.exists():
                module = ModuleLoader.load_from_path(module_name, path)
                if module:
                    return module
        
        print(f"❌ Could not find {module_name}.py in any of the specified paths")
        return None
    
    @staticmethod
    def get_attribute(module: Any, attr_name: str, default: Any = None) -> Any:
        """Safely get an attribute from a module"""
        return getattr(module, attr_name, default)


# =============================================================================
# LOAD PIPELINE MODULE
# =============================================================================

print("\n" + "="*60)
print("   📦 Loading Pipeline Module")
print("="*60)

# Try to load the pipeline
pipeline_module = None
PIPELINE_AVAILABLE = False
GROQ_API_KEY = None

# First try the configured path
if Config.PIPELINE_PATH.exists():
    pipeline_module = ModuleLoader.load_from_path("pipeline", Config.PIPELINE_PATH)
else:
    # Try alternative paths
    print(f"⚠️ Primary path not found: {Config.PIPELINE_PATH}")
    print("   Trying alternative paths...")
    pipeline_module = ModuleLoader.find_and_load("pipeline", Config.ALTERNATIVE_PATHS)

# Extract classes if module loaded successfully
if pipeline_module:
    PIPELINE_AVAILABLE = True
    
    # Extract all needed components
    VectorPlexPipeline = ModuleLoader.get_attribute(pipeline_module, 'VectorPlexPipeline')
    VideoSession = ModuleLoader.get_attribute(pipeline_module, 'VideoSession')
    SessionStatus = ModuleLoader.get_attribute(pipeline_module, 'SessionStatus')
    ContentDetector = ModuleLoader.get_attribute(pipeline_module, 'ContentDetector')
    ContentType = ModuleLoader.get_attribute(pipeline_module, 'ContentType')
    GROQ_API_KEY = ModuleLoader.get_attribute(pipeline_module, 'GROQ_API_KEY')
    
    # Verify critical components
    if not VectorPlexPipeline:
        print("⚠️ VectorPlexPipeline class not found in module!")
        PIPELINE_AVAILABLE = False
    
    if PIPELINE_AVAILABLE:
        print(f"\n✅ Pipeline loaded successfully!")
        print(f"   - VectorPlexPipeline: {'✓' if VectorPlexPipeline else '✗'}")
        print(f"   - VideoSession: {'✓' if VideoSession else '✗'}")
        print(f"   - SessionStatus: {'✓' if SessionStatus else '✗'}")
        print(f"   - GROQ_API_KEY: {'✓ Set' if GROQ_API_KEY else '✗ Not Set'}")

else:
    print("\n⚠️ Pipeline module not available")
    print("🔧 Running in DEMO MODE...\n")


# =============================================================================
# FALLBACK MOCK CLASSES (Demo Mode)
# =============================================================================

if not PIPELINE_AVAILABLE:
    
    class SessionStatus:
        """Mock SessionStatus for demo mode"""
        PENDING = "pending"
        PROCESSING = "processing" 
        READY = "ready"
        ERROR = "error"
    
    class VideoSession:
        """Mock VideoSession for demo mode"""
        def __init__(self, session_id: str = "", video_url: str = ""):
            self.session_id = session_id
            self.video_url = video_url
            self.video_title = ""
            self.audio_path = ""
            self.transcript = ""
            self.duration = 0
            self.word_count = 0
            self.chunk_count = 0
            self.language = "en"
            self.status = SessionStatus.PENDING
            self.chat_history = []
    
    class ContentType:
        """Mock ContentType for demo mode"""
        EDUCATIONAL = "educational"
        ENTERTAINMENT = "entertainment"
        NEWS = "news"
        OTHER = "other"
    
    class ContentDetector:
        """Mock ContentDetector for demo mode"""
        @staticmethod
        def detect(text: str) -> str:
            return ContentType.OTHER
    
    class MockDownloader:
        """Mock downloader for demo mode"""
        def download(self, url: str, session_id: str) -> Dict[str, Any]:
            return {'success': False, 'error': 'Demo mode'}
    
    class MockTranscriber:
        """Mock transcriber for demo mode"""
        def transcribe(self, audio_path: str, session_id: str) -> Dict[str, Any]:
            return {'success': False, 'error': 'Demo mode'}
    
    class MockChunker:
        """Mock chunker for demo mode"""
        def chunk(self, text: str) -> List[str]:
            return []
    
    class MockVectorStore:
        """Mock vector store for demo mode"""
        def create_collection(self, session_id: str) -> None:
            pass
        
        def add_chunks(self, session_id: str, chunks: List[str], transcript: str) -> None:
            pass
    
    class VectorPlexPipeline:
        """Mock VectorPlexPipeline for demo mode"""
        def __init__(self):
            self.session = None
            self.downloader = MockDownloader()
            self.transcriber = MockTranscriber()
            self.chunker = MockChunker()
            self.vectorstore = MockVectorStore()
        
        def chat(self, message: str) -> str:
            return "Demo mode - pipeline not available"
        
        def get_info(self) -> Dict[str, Any]:
            return {"mode": "demo", "available": False}
        
        def cleanup(self) -> None:
            pass


# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
CORS(app)


# =============================================================================
# SESSION MANAGER - Thread-Safe Pipeline Management
# =============================================================================

class SessionManager:
    """
    Manages multiple user sessions and their pipelines.
    Thread-safe for concurrent users.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for session manager"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._pipelines: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._global_lock = threading.Lock()
        self._initialized = True
    
    def create_session(self, video_url: str) -> str:
        """Create a new session for video processing"""
        session_id = str(uuid.uuid4())
        
        with self._global_lock:
            self._sessions[session_id] = {
                'id': session_id,
                'video_url': video_url,
                'video_title': '',
                'status': 'pending',
                'progress': 0,
                'current_step': 'Initializing...',
                'error': None,
                'created_at': datetime.now().isoformat(),
                'ready': False,
                'messages': [],
                'word_count': 0,
                'chunk_count': 0,
                'language': 'unknown',
                'duration': 0
            }
            self._locks[session_id] = threading.Lock()
            self._progress[session_id] = {
                'step': 'pending',
                'progress': 0,
                'message': 'Waiting to start...'
            }
        
        print(f"📝 Created session: {session_id[:8]}...")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self._sessions.get(session_id)
    
    def update_session(self, session_id: str, **kwargs) -> None:
        """Update session data"""
        if session_id not in self._sessions:
            return
            
        lock = self._locks.get(session_id, self._global_lock)
        with lock:
            self._sessions[session_id].update(kwargs)
    
    def get_pipeline(self, session_id: str) -> Optional[Any]:
        """Get pipeline for session"""
        return self._pipelines.get(session_id)
    
    def set_pipeline(self, session_id: str, pipeline: Any) -> None:
        """Set pipeline for session"""
        with self._global_lock:
            self._pipelines[session_id] = pipeline
    
    def update_progress(self, session_id: str, step: str, progress: int, message: str) -> None:
        """Update processing progress"""
        if session_id not in self._progress:
            return
            
        self._progress[session_id] = {
            'step': step,
            'progress': progress,
            'message': message
        }
        self.update_session(
            session_id,
            status=step,
            progress=progress,
            current_step=message
        )
    
    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get current progress"""
        return self._progress.get(session_id, {
            'step': 'unknown',
            'progress': 0,
            'message': 'Unknown status'
        })
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to chat history"""
        if session_id not in self._sessions:
            return
            
        lock = self._locks.get(session_id, self._global_lock)
        with lock:
            self._sessions[session_id]['messages'].append({
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat messages"""
        session = self._sessions.get(session_id)
        if session:
            return session.get('messages', [])
        return []
    
    def clear_messages(self, session_id: str) -> None:
        """Clear chat messages"""
        if session_id not in self._sessions:
            return
            
        lock = self._locks.get(session_id, self._global_lock)
        with lock:
            self._sessions[session_id]['messages'] = []
    
    def cleanup_session(self, session_id: str) -> None:
        """Cleanup a specific session"""
        with self._global_lock:
            # Cleanup pipeline
            if session_id in self._pipelines:
                try:
                    self._pipelines[session_id].cleanup()
                except Exception as e:
                    print(f"⚠️ Error cleaning up pipeline: {e}")
                del self._pipelines[session_id]
            
            # Remove session data
            self._sessions.pop(session_id, None)
            self._locks.pop(session_id, None)
            self._progress.pop(session_id, None)
        
        print(f"🧹 Cleaned up session: {session_id[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        return {
            'active_sessions': len(self._sessions),
            'active_pipelines': len(self._pipelines),
            'sessions': list(self._sessions.keys())
        }


# Global session manager (singleton)
session_manager = SessionManager()


# =============================================================================
# VIDEO PROCESSOR - Background Processing
# =============================================================================

class VideoProcessor:
    """
    Handles video processing in background thread.
    Provides real-time progress updates.
    """
    
    @staticmethod
    def process_async(session_id: str, video_url: str) -> threading.Thread:
        """Start video processing in background thread"""
        thread = threading.Thread(
            target=VideoProcessor._process,
            args=(session_id, video_url),
            daemon=True,
            name=f"processor-{session_id[:8]}"
        )
        thread.start()
        return thread
    
    @staticmethod
    def _process(session_id: str, video_url: str) -> None:
        """Main processing function - runs in background"""
        log_prefix = f"[{session_id[:8]}]"
        
        try:
            print(f"\n{log_prefix} 🚀 Starting video processing...")
            print(f"{log_prefix} URL: {video_url[:50]}...")
            
            # Initialize
            session_manager.update_progress(
                session_id, 'initializing', 5, 'Initializing pipeline...'
            )
            time.sleep(0.5)
            
            # Check if pipeline is available
            if not PIPELINE_AVAILABLE:
                print(f"{log_prefix} ⚠️ Pipeline not available, using demo mode")
                VideoProcessor._process_demo(session_id, video_url)
                return
            
            # Create pipeline instance
            print(f"{log_prefix} Creating VectorPlexPipeline...")
            pipeline = VectorPlexPipeline()
            session_manager.set_pipeline(session_id, pipeline)
            
            # Initialize session
            pipeline.session = VideoSession(
                session_id=session_id,
                video_url=video_url
            )
            
            # ==================== STEP 1: DOWNLOAD ====================
            session_manager.update_progress(
                session_id, 'downloading', 10, 'Fetching video information...'
            )
            
            print(f"{log_prefix} 📥 Downloading video...")
            dl_result = pipeline.downloader.download(video_url, session_id)
            
            if not dl_result.get('success'):
                raise Exception(f"Download failed: {dl_result.get('error', 'Unknown error')}")
            
            # Update session with video info
            pipeline.session.video_title = dl_result['title']
            pipeline.session.audio_path = dl_result['audio_path']
            pipeline.session.duration = dl_result['duration']
            
            session_manager.update_session(session_id, video_title=dl_result['title'])
            
            title_preview = dl_result['title'][:40] + '...' if len(dl_result['title']) > 40 else dl_result['title']
            session_manager.update_progress(
                session_id, 'downloading', 25, f'Downloaded: {title_preview}'
            )
            print(f"{log_prefix} ✓ Downloaded: {dl_result['title'][:50]}...")
            
            # ==================== STEP 2: TRANSCRIBE ====================
            session_manager.update_progress(
                session_id, 'transcribing', 30, 'Loading Whisper model...'
            )
            
            print(f"{log_prefix} 🎤 Transcribing audio...")
            session_manager.update_progress(
                session_id, 'transcribing', 40, 'Transcribing audio with Whisper AI...'
            )
            
            tr_result = pipeline.transcriber.transcribe(dl_result['audio_path'], session_id)
            
            if not tr_result.get('success'):
                raise Exception(f"Transcription failed: {tr_result.get('error', 'Unknown error')}")
            
            # Update session with transcript info
            pipeline.session.transcript = tr_result['transcript']
            pipeline.session.word_count = tr_result['word_count']
            pipeline.session.language = tr_result['language']
            
            session_manager.update_progress(
                session_id, 'transcribing', 55,
                f'Transcribed {tr_result["word_count"]:,} words in {tr_result["language"]}'
            )
            print(f"{log_prefix} ✓ Transcribed: {tr_result['word_count']:,} words")
            
            # ==================== STEP 3: CHUNK TEXT ====================
            session_manager.update_progress(
                session_id, 'processing', 60, 'Splitting transcript into chunks...'
            )
            
            print(f"{log_prefix} 📄 Chunking transcript...")
            chunks = pipeline.chunker.chunk(tr_result['transcript'])
            pipeline.session.chunk_count = len(chunks)
            
            session_manager.update_progress(
                session_id, 'processing', 70, f'Created {len(chunks)} semantic chunks'
            )
            print(f"{log_prefix} ✓ Created {len(chunks)} chunks")
            
            # ==================== STEP 4: CREATE EMBEDDINGS ====================
            session_manager.update_progress(
                session_id, 'embedding', 75, 'Creating vector embeddings...'
            )
            
            print(f"{log_prefix} 🧠 Creating embeddings...")
            pipeline.vectorstore.create_collection(session_id)
            
            session_manager.update_progress(
                session_id, 'embedding', 85, 'Storing vectors in ChromaDB...'
            )
            
            pipeline.vectorstore.add_chunks(session_id, chunks, tr_result['transcript'])
            
            session_manager.update_progress(
                session_id, 'embedding', 95, 'Finalizing knowledge base...'
            )
            print(f"{log_prefix} ✓ Embeddings stored")
            
            # ==================== COMPLETE ====================
            pipeline.session.status = SessionStatus.READY
            
            session_manager.update_session(
                session_id,
                ready=True,
                word_count=tr_result['word_count'],
                chunk_count=len(chunks),
                language=tr_result['language'],
                duration=dl_result['duration']
            )
            
            session_manager.update_progress(
                session_id, 'ready', 100, 'Ready to chat!'
            )
            
            print(f"\n{log_prefix} ✅ Processing complete!")
            print(f"{log_prefix}    Title: {dl_result['title'][:50]}...")
            print(f"{log_prefix}    Words: {tr_result['word_count']:,}")
            print(f"{log_prefix}    Chunks: {len(chunks)}")
            print(f"{log_prefix}    Language: {tr_result['language']}\n")
            
        except Exception as e:
            error_msg = str(e)
            print(f"{log_prefix} ❌ Error: {error_msg}")
            
            session_manager.update_session(
                session_id,
                status='error',
                error=error_msg,
                ready=False
            )
            
            session_manager.update_progress(
                session_id, 'error', 0, f'Error: {error_msg[:100]}'
            )
            
            # Cleanup on error
            pipeline = session_manager.get_pipeline(session_id)
            if pipeline:
                try:
                    pipeline.cleanup()
                except:
                    pass
    
    @staticmethod
    def _process_demo(session_id: str, video_url: str) -> None:
        """Demo processing when pipeline is not available"""
        log_prefix = f"[{session_id[:8]}]"
        print(f"{log_prefix} 🎭 Running demo processing...")
        
        # Simulated processing steps
        steps = [
            ('downloading', 15, 'Connecting to video source...', 1.0),
            ('downloading', 25, 'Downloading video...', 1.5),
            ('downloading', 35, 'Extracting audio...', 1.0),
            ('transcribing', 45, 'Loading Whisper model...', 1.0),
            ('transcribing', 55, 'Transcribing audio...', 2.0),
            ('transcribing', 65, 'Processing transcript...', 1.0),
            ('processing', 75, 'Creating semantic chunks...', 1.0),
            ('embedding', 85, 'Generating embeddings...', 1.5),
            ('embedding', 92, 'Storing in ChromaDB...', 1.0),
            ('embedding', 98, 'Finalizing...', 0.5),
            ('ready', 100, 'Ready to chat!', 0.3),
        ]
        
        for step, progress, message, delay in steps:
            session_manager.update_progress(session_id, step, progress, message)
            time.sleep(delay)
        
        # Extract video ID for demo title
        video_id = video_url.split('v=')[-1][:11] if 'v=' in video_url else session_id[:8]
        
        # Set demo data
        session_manager.update_session(
            session_id,
            video_title=f'Demo Video [{video_id}]',
            ready=True,
            word_count=5000,
            chunk_count=25,
            language='en',
            duration=600
        )
        
        print(f"{log_prefix} ✅ Demo processing complete!")


# =============================================================================
# CHAT HANDLER
# =============================================================================

class ChatHandler:
    """Handles chat interactions using the pipeline's chat functionality."""
    
    @staticmethod
    def get_response(session_id: str, message: str) -> Dict[str, Any]:
        """Get AI response for user message"""
        try:
            session_data = session_manager.get_session(session_id)
            
            if not session_data:
                return {'success': False, 'error': 'Session not found'}
            
            if not session_data.get('ready'):
                return {'success': False, 'error': 'Video not yet processed'}
            
            # Save user message
            session_manager.add_message(session_id, 'user', message)
            
            # Get response
            if PIPELINE_AVAILABLE:
                response = ChatHandler._get_pipeline_response(session_id, message)
            else:
                response = ChatHandler._get_demo_response(message, session_data)
            
            if response:
                session_manager.add_message(session_id, 'assistant', response)
                return {
                    'success': True,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': 'Failed to generate response'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _get_pipeline_response(session_id: str, message: str) -> Optional[str]:
        """Get response from actual pipeline"""
        pipeline = session_manager.get_pipeline(session_id)
        
        if not pipeline:
            return None
            
        if not pipeline.session:
            return None
            
        if pipeline.session.status != SessionStatus.READY:
            return None
        
        return pipeline.chat(message)
    
    @staticmethod
    def _get_demo_response(message: str, session_data: Dict[str, Any]) -> str:
        """Generate demo response"""
        message_lower = message.lower().strip()
        video_title = session_data.get('video_title', 'this video')
        
        # Response templates
        templates = {
            'greeting': ChatHandler._greeting_response,
            'identity': ChatHandler._identity_response,
            'summary': ChatHandler._summary_response,
            'default': ChatHandler._default_response
        }
        
        # Detect intent
        if any(message_lower.startswith(g) for g in ['hi', 'hello', 'hey', 'howdy']):
            return templates['greeting'](video_title)
        
        if any(phrase in message_lower for phrase in ['who are you', 'what are you', 'your name']):
            return templates['identity'](video_title)
        
        if any(word in message_lower for word in ['summary', 'summarize', 'overview', 'about']):
            return templates['summary'](video_title)
        
        return templates['default'](message, video_title)
    
    @staticmethod
    def _greeting_response(video_title: str) -> str:
        return f"""Hello! 👋 I'm **VectorPlex V2**, your AI video learning assistant!

I'm here to help you understand and explore the content from "{video_title}".

**What I can do:**
- 📝 Summarize the video content
- 🔍 Answer specific questions
- 💡 Explain complex concepts
- ⏱️ Find specific moments

What would you like to know?

---
✨ **VectorPlex V2** | 🧠 **Sangam Gautam** | 🎨 **Sushil Yadav**
"""
    
    @staticmethod
    def _identity_response(video_title: str) -> str:
        return f"""I'm **VectorPlex V2**, an intelligent AI assistant! 🎬

**My Capabilities:**
- 🎤 **Whisper AI** for transcription
- 🧠 **ChromaDB** for semantic search  
- ⚡ **Groq** for fast responses

**Created by Team SB:**
- 🧠 **Sangam Gautam** - Backend & AI
- 🎨 **Sushil Yadav** - Frontend

Currently analyzing: "{video_title}"

---
✨ **VectorPlex V2** | [GitHub](https://github.com/sangamgautam1111)
"""
    
    @staticmethod
    def _summary_response(video_title: str) -> str:
        return f"""## 📝 Video Summary: {video_title}

### 🎯 Overview
This video covers important concepts and provides valuable insights.

### 📚 Main Topics
1. **Introduction** - Setting up the foundation
2. **Core Concepts** - Deep dive into main ideas
3. **Examples** - Real-world applications
4. **Takeaways** - Key points to remember

### 💡 Key Insights
- Important concept from the video
- Practical application tip
- Connecting theme

### 🎓 Conclusion
Comprehensive coverage with practical examples.

---
✨ **VectorPlex V2** | 🧠 **Sangam Gautam** | 🎨 **Sushil Yadav**
"""
    
    @staticmethod
    def _default_response(message: str, video_title: str) -> str:
        return f"""Based on "{video_title}", here's what I found:

**Your question:** "{message}"

The video discusses this topic. Key points:

1. **Context** - Background information provided
2. **Main Points** - Important concepts covered
3. **Examples** - Practical demonstrations
4. **Conclusion** - Summary with takeaways

**Would you like me to:**
- 📝 Provide more details?
- 🔍 Focus on a specific aspect?
- ⏱️ Find timestamps?

---
✨ **VectorPlex V2** | 🧠 **Sangam Gautam** | 🎨 **Sushil Yadav**
"""


# =============================================================================
# API RESPONSE HELPERS
# =============================================================================

def api_success(data: Dict[str, Any] = None, **kwargs) -> Response:
    """Create successful API response"""
    response = {'success': True}
    if data:
        response.update(data)
    response.update(kwargs)
    return jsonify(response)


def api_error(error: str, status_code: int = 400) -> tuple:
    """Create error API response"""
    return jsonify({'success': False, 'error': error}), status_code


# =============================================================================
# FLASK ROUTES - PAGES
# =============================================================================

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')


@app.route('/chat/<session_id>')
def chat_page(session_id: str):
    """Chat UI page"""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        session_data = {
            'id': session_id,
            'video_title': 'Video Chat Session',
            'video_url': '',
            'messages': [],
            'ready': False
        }
    
    return render_template(
        'ui.html',
        session_id=session_id,
        chat_data={
            'video_url': session_data.get('video_url', ''),
            'video_title': session_data.get('video_title', 'Video Chat'),
            'messages': session_data.get('messages', [])
        }
    )


# =============================================================================
# FLASK ROUTES - API
# =============================================================================

@app.route('/api/process', methods=['POST'])
def api_process_video():
    """Start video processing"""
    try:
        data = request.get_json() or {}
        video_url = data.get('video_url', '').strip()
        
        # Validation
        if not video_url:
            return api_error('No video URL provided')
        
        if not video_url.startswith(('http://', 'https://')):
            return api_error('Invalid URL format')
        
        # Create session and start processing
        session_id = session_manager.create_session(video_url)
        VideoProcessor.process_async(session_id, video_url)
        
        return api_success(
            session_id=session_id,
            message='Processing started',
            status_url=f'/api/status/{session_id}',
            chat_url=f'/chat/{session_id}'
        )
        
    except Exception as e:
        return api_error(str(e), 500)


@app.route('/api/status/<session_id>')
def api_get_status(session_id: str):
    """Get processing status"""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        return api_error('Session not found', 404)
    
    progress = session_manager.get_progress(session_id)
    
    return api_success(
        session_id=session_id,
        status=progress.get('step', 'unknown'),
        progress=progress.get('progress', 0),
        message=progress.get('message', ''),
        ready=session_data.get('ready', False),
        error=session_data.get('error'),
        video_title=session_data.get('video_title', ''),
        word_count=session_data.get('word_count', 0),
        chunk_count=session_data.get('chunk_count', 0)
    )


@app.route('/api/status/<session_id>/stream')
def api_stream_status(session_id: str):
    """Server-Sent Events for real-time status updates"""
    
    def generate():
        last_progress = -1
        start_time = time.time()
        
        while True:
            # Timeout check
            if time.time() - start_time > Config.STREAM_TIMEOUT:
                yield f"data: {json.dumps({'done': True, 'timeout': True})}\n\n"
                break
            
            session_data = session_manager.get_session(session_id)
            
            if not session_data:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break
            
            progress = session_manager.get_progress(session_id)
            current_progress = progress.get('progress', 0)
            
            # Send update if progress changed
            if current_progress != last_progress:
                last_progress = current_progress
                
                data = {
                    'status': progress.get('step', 'unknown'),
                    'progress': current_progress,
                    'message': progress.get('message', ''),
                    'ready': session_data.get('ready', False),
                    'error': session_data.get('error'),
                    'video_title': session_data.get('video_title', '')
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop if complete or error
                if session_data.get('ready') or session_data.get('error'):
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    break
            
            time.sleep(Config.POLL_INTERVAL)
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Send message and get AI response"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id', '').strip()
        message = data.get('message', '').strip()
        
        if not session_id:
            return api_error('No session ID provided')
        
        if not message:
            return api_error('No message provided')
        
        result = ChatHandler.get_response(session_id, message)
        
        if result['success']:
            return jsonify(result)
        else:
            return api_error(result.get('error', 'Unknown error'))
            
    except Exception as e:
        return api_error(str(e), 500)


@app.route('/api/history/<session_id>')
def api_get_history(session_id: str):
    """Get chat history"""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        return api_error('Session not found', 404)
    
    return api_success(
        messages=session_data.get('messages', []),
        video_title=session_data.get('video_title', ''),
        ready=session_data.get('ready', False)
    )


@app.route('/api/clear/<session_id>', methods=['POST'])
def api_clear_history(session_id: str):
    """Clear chat history"""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        return api_error('Session not found', 404)
    
    session_manager.clear_messages(session_id)
    
    # Also clear pipeline chat history
    pipeline = session_manager.get_pipeline(session_id)
    if pipeline and hasattr(pipeline, 'session') and pipeline.session:
        if hasattr(pipeline.session, 'chat_history'):
            pipeline.session.chat_history = []
    
    return api_success(message='Chat history cleared')


@app.route('/api/session/<session_id>/info')
def api_session_info(session_id: str):
    """Get detailed session information"""
    session_data = session_manager.get_session(session_id)
    
    if not session_data:
        return api_error('Session not found', 404)
    
    pipeline = session_manager.get_pipeline(session_id)
    pipeline_info = {}
    
    if pipeline and hasattr(pipeline, 'get_info'):
        pipeline_info = pipeline.get_info()
    
    return api_success(
        session={
            'id': session_id,
            'video_url': session_data.get('video_url', ''),
            'video_title': session_data.get('video_title', ''),
            'status': session_data.get('status', 'unknown'),
            'ready': session_data.get('ready', False),
            'word_count': session_data.get('word_count', 0),
            'chunk_count': session_data.get('chunk_count', 0),
            'language': session_data.get('language', 'unknown'),
            'duration': session_data.get('duration', 0),
            'message_count': len(session_data.get('messages', [])),
            'created_at': session_data.get('created_at', ''),
            'pipeline_active': pipeline is not None
        },
        pipeline_info=pipeline_info
    )


@app.route('/api/session/<session_id>/cleanup', methods=['POST'])
def api_cleanup_session(session_id: str):
    """Cleanup and delete a session"""
    try:
        session_manager.cleanup_session(session_id)
        return api_success(message='Session cleaned up successfully')
    except Exception as e:
        return api_error(str(e), 500)


@app.route('/api/health')
def api_health():
    """API health check"""
    stats = session_manager.get_stats()
    
    return api_success(
        status='healthy',
        pipeline_available=PIPELINE_AVAILABLE,
        groq_configured=bool(GROQ_API_KEY) if PIPELINE_AVAILABLE else False,
        active_sessions=stats['active_sessions'],
        timestamp=datetime.now().isoformat(),
        version='2.0.0'
    )


@app.route('/api/debug')
def api_debug():
    """Debug information (disable in production)"""
    if not Config.DEBUG:
        return api_error('Debug endpoint disabled', 403)
    
    return api_success(
        config={
            'pipeline_path': str(Config.PIPELINE_PATH),
            'current_dir': str(Config.CURRENT_DIR),
            'pipeline_exists': Config.PIPELINE_PATH.exists(),
            'pipeline_available': PIPELINE_AVAILABLE
        },
        session_manager=session_manager.get_stats(),
        python_path=sys.path[:5]
    )


# =============================================================================
# LEGACY ROUTES (Backward Compatibility)
# =============================================================================

@app.route('/process', methods=['POST'])
def legacy_process():
    """Legacy process endpoint"""
    try:
        data = request.get_json() or {}
        video_url = data.get('video_url', '').strip()
        
        if not video_url:
            return api_error('No video URL provided')
        
        session_id = session_manager.create_session(video_url)
        VideoProcessor.process_async(session_id, video_url)
        
        return api_success(
            session_id=session_id,
            redirect_url=f'/chat/{session_id}'
        )
        
    except Exception as e:
        return api_error(str(e), 500)


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def handle_404(e):
    if request.path.startswith('/api/'):
        return api_error('Endpoint not found', 404)
    return render_template('index.html'), 404


@app.errorhandler(500)
def handle_500(e):
    return api_error('Internal server error', 500)


@app.errorhandler(Exception)
def handle_exception(e):
    print(f"❌ Unhandled exception: {type(e).__name__}: {e}")
    if request.path.startswith('/api/'):
        return api_error(str(e), 500)
    return render_template('index.html'), 500


# =============================================================================
# MAIN
# =============================================================================

def print_banner():
    """Print startup banner"""
    print("\n" + "="*65)
    print("   🚀 VectorPlex V2 - Flask Server")
    print("   By Sangam Gautam & Sushil Yadav")
    print("="*65)
    print(f"\n   📦 Pipeline: {'✅ Loaded' if PIPELINE_AVAILABLE else '❌ Demo Mode'}")
    
    if PIPELINE_AVAILABLE:
        print(f"   🔑 Groq API: {'✅ Configured' if GROQ_API_KEY else '⚠️ Not Set'}")
    
    print(f"\n   📁 Pipeline Path: {Config.PIPELINE_PATH}")
    print(f"   📁 Current Dir: {Config.CURRENT_DIR}")
    print(f"\n   🌐 Server: http://localhost:{Config.PORT}")
    print(f"   🔧 Debug: {'Enabled' if Config.DEBUG else 'Disabled'}")
    print("="*65 + "\n")


if __name__ == '__main__':
    print_banner()
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )