

import os
import sys
import json
import re
import time
import uuid
import warnings
import shutil
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================

from dotenv import load_dotenv

# Load from backend directory
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# =============================================================================
# CONFIGURATION FROM .env
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Paths from .env
AUDIO_PATH = Path(os.getenv("AUDIO_PATH", "./data/audio"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./data/chroma_db"))
TRANSCRIPT_PATH = Path(os.getenv("TRANSCRIPT_PATH", "./data/transcriptions"))

# Create directories
for path in [AUDIO_PATH, CHROMA_PATH, TRANSCRIPT_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEPENDENCY IMPORTS
# =============================================================================

print("[VectorPlex V2] Loading dependencies...")

try:
    import yt_dlp
    import whisper
    import torch
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    print("[VectorPlex V2] ✅ All dependencies loaded!")
except ImportError as e:
    print(f"\n❌ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install yt-dlp openai-whisper torch chromadb sentence-transformers groq python-dotenv")
    sys.exit(1)

# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SessionStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class ContentType(str, Enum):
    GENERAL = "general"
    CODING = "coding"
    MATH = "math"
    SCIENCE = "science"
    SUMMARY = "summary"
    GREETING = "greeting"


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class VideoSession:
    session_id: str
    video_url: str
    video_title: str = ""
    transcript: str = ""
    word_count: int = 0
    chunk_count: int = 0
    duration: int = 0
    language: str = "unknown"
    status: SessionStatus = SessionStatus.PENDING
    error: str = ""
    audio_path: str = ""
    chat_history: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# =============================================================================
# CREATOR FOOTER
# =============================================================================

CREATOR_FOOTER = """

---
✨ **Powered by VectorPlex V2** ✨
🧠 Created by **Sangam Gautam** - [GitHub](https://github.com/sangamgautam1111)
🎨 Frontend by **Sushil Yadav** - [GitHub](https://github.com/sushilraey)
"""


# =============================================================================
# INTELLIGENT CONTENT DETECTOR - FIXED
# =============================================================================

class ContentDetector:
    """
    Intelligent content type detection.
    FIXED: Now properly detects greetings and casual questions.
    """
    
    # Greeting patterns - These should NOT trigger code responses
    GREETING_PATTERNS = [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^howdy\b", r"^greetings\b",
        r"^good morning\b", r"^good afternoon\b", r"^good evening\b",
        r"^what'?s up\b", r"^sup\b", r"^yo\b",
        r"^who are you", r"^what are you", r"^introduce yourself",
        r"^tell me about yourself", r"^what can you do",
        r"^how are you", r"^how do you do", r"^how's it going",
        r"^nice to meet you", r"^pleased to meet you",
        r"^thank you", r"^thanks", r"^bye", r"^goodbye", r"^see you",
        r"^help\b", r"^help me\b"
    ]
    
    # Explicit code request patterns
    CODE_REQUEST_PATTERNS = [
        r"write.*code", r"show.*code", r"give.*code", r"create.*code",
        r"write.*function", r"write.*program", r"write.*script",
        r"code for", r"code to", r"implement", r"implementation",
        r"how to code", r"how to program", r"how to implement",
        r"syntax for", r"example code", r"code example",
        r"in python", r"in javascript", r"in java", r"in c\+\+",
        r"programming", r"algorithm for", r"function for",
        r"class for", r"method for", r"write a.*that"
    ]
    
    CODING_KEYWORDS = [
        "function", "class", "variable", "loop", "array", "list",
        "dictionary", "object", "api", "debug", "error", "bug",
        "compile", "runtime", "syntax", "import", "export",
        "git", "github", "docker", "server", "client",
        "frontend", "backend", "framework", "library", "module"
    ]
    
    MATH_KEYWORDS = [
        "equation", "formula", "calculate", "solve", "derivative",
        "integral", "matrix", "algebra", "calculus", "geometry",
        "probability", "statistics", "theorem", "proof", "quadratic"
    ]
    
    SCIENCE_KEYWORDS = [
        "physics", "chemistry", "biology", "experiment", "hypothesis",
        "theory", "force", "energy", "mass", "velocity", "atom",
        "molecule", "element", "reaction", "cell", "dna"
    ]
    
    SUMMARY_KEYWORDS = [
        "summarize", "summary", "overview", "main points", "key points",
        "takeaways", "what is this about", "tldr", "recap", "brief"
    ]
    
    @classmethod
    def detect(cls, question: str, context: str = "") -> ContentType:
        """
        Detect content type from question.
        PRIORITY ORDER:
        1. Greetings/Casual questions → GREETING (no code)
        2. Explicit code requests → CODING
        3. Summary requests → SUMMARY
        4. Math keywords → MATH
        5. Science keywords → SCIENCE
        6. General coding keywords (only if context suggests code) → CODING
        7. Default → GENERAL
        """
        q_lower = question.lower().strip()
        
        # 1. CHECK FOR GREETINGS FIRST - Highest priority
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.GREETING
        
        # 2. Check for explicit code requests
        for pattern in cls.CODE_REQUEST_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.CODING
        
        # 3. Check for summary requests
        if any(kw in q_lower for kw in cls.SUMMARY_KEYWORDS):
            return ContentType.SUMMARY
        
        # 4. Check for math (explicit in question)
        math_in_question = sum(1 for kw in cls.MATH_KEYWORDS if kw in q_lower)
        if math_in_question >= 2:
            return ContentType.MATH
        
        # 5. Check for science (explicit in question)
        science_in_question = sum(1 for kw in cls.SCIENCE_KEYWORDS if kw in q_lower)
        if science_in_question >= 2:
            return ContentType.SCIENCE
        
        # 6. Check for coding ONLY if explicitly asking about code concepts
        coding_in_question = sum(1 for kw in cls.CODING_KEYWORDS if kw in q_lower)
        if coding_in_question >= 2:
            return ContentType.CODING
        
        # 7. Default to general
        return ContentType.GENERAL


# =============================================================================
# ELITE SYSTEM PROMPTS - FIXED
# =============================================================================

class EliteSystemPrompts:
    """
    Elite system prompts for intelligent responses.
    FIXED: Proper handling of different content types.
    """
    
    @staticmethod
    def get_prompt(video_title: str, content_type: ContentType) -> str:
        """Generate appropriate system prompt based on content type"""
        
        # Base identity
        identity = f'''You are **VectorPlex V2**, an intelligent AI video learning assistant.

**Created by Team Algolix AI:**
- 🧠 **Sangam Gautam** - Backend & AI
- 🎨 **Sushil Yadav** - Frontend Design

**Current Video:** "{video_title}"

'''
        
        # GREETING prompt - Simple, friendly, NO CODE
        if content_type == ContentType.GREETING:
            return identity + '''**YOUR ROLE:** You are having a friendly conversation.

**IMPORTANT RULES:**
1. Be friendly and conversational
2. DO NOT include any code blocks
3. DO NOT provide technical explanations unless explicitly asked
4. Keep responses concise and warm
5. If asked "who are you", introduce yourself briefly
6. If asked about the video, give a brief overview

**RESPONSE STYLE:**
- Warm and friendly tone
- Short paragraphs
- Use emojis sparingly (1-2 max)
- 100-300 words maximum
- NO code, NO technical deep-dives

**EXAMPLE RESPONSES:**

For "Hi, who are you?":
"Hello! 👋 I'm VectorPlex V2, your AI learning assistant! I'm here to help you understand and explore the content from videos. I was created by Sangam Gautam and Sushil Yadav.

Right now, I'm ready to answer any questions you have about the video you've loaded. Feel free to ask me to summarize it, explain specific concepts, or dive deep into any topic covered!

What would you like to know?"

For "How are you?":
"I'm doing great, thanks for asking! 😊 I'm ready and excited to help you learn from this video. Is there anything specific you'd like to know about it?"'''

        # GENERAL prompt - Balanced, informative
        elif content_type == ContentType.GENERAL:
            return identity + '''**YOUR ROLE:** Answer questions about the video content clearly and helpfully.

**RESPONSE GUIDELINES:**
1. Answer based on the video transcript provided
2. Be informative but not overwhelming
3. Use clear structure with headers when needed
4. Include code ONLY if the user explicitly asks for it
5. Keep responses focused and relevant

**FORMATTING:**
- Use **bold** for key terms
- Use bullet points for lists
- Use headers (##) for sections when appropriate
- Keep responses 200-600 words unless more detail is needed

**IMPORTANT:**
- If the question is simple, give a simple answer
- If the question needs depth, provide depth
- Match your response length to the question complexity
- Don't include code unless specifically requested'''

        # CODING prompt - Only when explicitly requested
        elif content_type == ContentType.CODING:
            return identity + '''**YOUR ROLE:** Provide coding explanations and examples.

**WHEN TO INCLUDE CODE:**
- User explicitly asks for code examples
- User asks "how to implement" something
- User asks about syntax or programming concepts
- User wants to see code from the video

**CODE FORMATTING:**
```language
// Always specify the language
// Add helpful comments
// Keep code clean and readable
```

**RESPONSE STRUCTURE:**
1. Brief explanation of the concept
2. Code example (if requested)
3. Line-by-line explanation (for complex code)
4. Common mistakes to avoid
5. Best practices

**IMPORTANT:**
- Only include code if it's relevant to the question
- Explain the code, don't just dump it
- Keep explanations clear and educational'''

        # MATH prompt
        elif content_type == ContentType.MATH:
            return identity + '''**YOUR ROLE:** Explain mathematical concepts clearly.

**RESPONSE STRUCTURE:**
1. Explain the concept in simple terms
2. Show the formula/equation
3. Walk through step-by-step solutions
4. Provide examples
5. Note common mistakes

**FORMATTING:**
- Use clear step numbering
- Bold important formulas
- Explain each step
- Verify answers when possible'''

        # SCIENCE prompt
        elif content_type == ContentType.SCIENCE:
            return identity + '''**YOUR ROLE:** Explain scientific concepts engagingly.

**RESPONSE STRUCTURE:**
1. Simple explanation first
2. Technical details
3. Real-world examples
4. Key takeaways

**FORMATTING:**
- Use analogies to explain complex ideas
- Connect to everyday experiences
- Highlight cause and effect'''

        # SUMMARY prompt
        elif content_type == ContentType.SUMMARY:
            return identity + '''**YOUR ROLE:** Provide comprehensive video summaries.

**SUMMARY STRUCTURE:**

## 🎬 Overview
Brief 2-3 sentence overview

## 📚 Main Topics
List and explain each major topic

## 💡 Key Takeaways
- Important point 1
- Important point 2
- Important point 3

## 🎓 Conclusion
What you should remember

**GUIDELINES:**
- Cover all major points from the video
- Be thorough but organized
- Use the transcript as your source'''

        return identity


# =============================================================================
# VIDEO DOWNLOADER
# =============================================================================

class VideoDownloader:
    """Downloads and extracts audio from videos"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def download(self, url: str, session_id: str) -> Dict[str, Any]:
        """Download video and extract MP3 audio"""
        try:
            audio_file = self.output_path / f"{session_id}.mp3"
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.output_path / f'{session_id}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            # Get video info
            print("  📥 Fetching video info...")
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown Video')
                duration = info.get('duration', 0)
            
            print(f"  📹 Title: {title[:55]}{'...' if len(title) > 55 else ''}")
            print(f"  ⏱️  Duration: {duration // 60}m {duration % 60}s")
            
            # Download
            print("  📥 Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Convert if needed
            if not audio_file.exists():
                for ext in ['webm', 'm4a', 'opus', 'ogg', 'wav']:
                    alt = self.output_path / f"{session_id}.{ext}"
                    if alt.exists():
                        print(f"  🔄 Converting {ext} → MP3...")
                        import subprocess
                        subprocess.run([
                            'ffmpeg', '-i', str(alt), '-vn',
                            '-acodec', 'libmp3lame', '-q:a', '2',
                            str(audio_file), '-y'
                        ], capture_output=True)
                        alt.unlink(missing_ok=True)
                        break
            
            if audio_file.exists():
                print(f"  ✅ Audio saved: {audio_file.name}")
                return {
                    'success': True,
                    'audio_path': str(audio_file),
                    'title': title,
                    'duration': duration
                }
            
            return {'success': False, 'error': 'Audio file not created'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =============================================================================
# WHISPER TRANSCRIBER
# =============================================================================

class WhisperTranscriber:
    """Transcribes audio using Whisper"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Device: {self.device.upper()}")
    
    def _load_model(self):
        if self.model is None:
            print(f"  🔄 Loading Whisper '{WHISPER_MODEL}' model...")
            self.model = whisper.load_model(WHISPER_MODEL, device=self.device)
            print("  ✅ Model loaded")
        return self.model
    
    def transcribe(self, audio_path: str, session_id: str) -> Dict[str, Any]:
        """Transcribe audio to text"""
        try:
            model = self._load_model()
            
            print("  🎤 Transcribing...")
            start = time.time()
            
            result = model.transcribe(
                audio_path,
                fp16=(self.device == "cuda"),
                verbose=False
            )
            
            transcript = result["text"].strip()
            elapsed = time.time() - start
            word_count = len(transcript.split())
            language = result.get('language', 'unknown')
            
            # Save transcript
            transcript_file = TRANSCRIPT_PATH / f"{session_id}.txt"
            transcript_file.write_text(transcript, encoding='utf-8')
            
            print(f"  ✅ Transcription complete!")
            print(f"  📝 Words: {word_count} | Language: {language} | Time: {elapsed:.1f}s")
            
            return {
                'success': True,
                'transcript': transcript,
                'word_count': word_count,
                'language': language,
                'time': round(elapsed, 2)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =============================================================================
# TEXT CHUNKER
# =============================================================================

class TextChunker:
    """Intelligent text chunking"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text with overlap"""
        if not text.strip():
            return []
        
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size * 4 and current:
                chunk_text = ' '.join(current)
                chunks.append({
                    'text': chunk_text,
                    'index': len(chunks),
                    'word_count': len(chunk_text.split())
                })
                
                # Overlap
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.overlap * 4:
                        overlap_sents.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_sents
                current_len = overlap_len
            
            current.append(sentence)
            current_len += len(sentence)
        
        if current:
            chunk_text = ' '.join(current)
            chunks.append({
                'text': chunk_text,
                'index': len(chunks),
                'word_count': len(chunk_text.split())
            })
        
        return chunks


# =============================================================================
# VECTOR STORE - WITH PROPER CLEANUP
# =============================================================================

class VectorStore:
    """ChromaDB vector storage with proper cleanup"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.client = None
        self.embedding_model = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client"""
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _load_embeddings(self):
        if self.embedding_model is None:
            print("  🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  ✅ Embeddings ready")
        return self.embedding_model
    
    def create_collection(self, session_id: str) -> bool:
        try:
            name = f"session_{session_id}"
            try:
                self.client.delete_collection(name)
            except:
                pass
            self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
            return True
        except Exception as e:
            print(f"  ❌ Collection error: {e}")
            return False
    
    def add_chunks(self, session_id: str, chunks: List[Dict], full_transcript: str) -> bool:
        try:
            collection = self.client.get_collection(f"session_{session_id}")
            model = self._load_embeddings()
            
            ids, docs, metas = [], [], []
            for chunk in chunks:
                text = chunk.get('text', '')
                if text:
                    ids.append(f"{session_id}_chunk_{chunk['index']}")
                    docs.append(text)
                    metas.append({
                        'chunk_index': chunk['index'],
                        'word_count': chunk['word_count'],
                        'start_pos': full_transcript.find(text[:50])
                    })
            
            if docs:
                print(f"  🔄 Embedding {len(docs)} chunks...")
                embeddings = model.encode(docs, show_progress_bar=False).tolist()
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
                print(f"  ✅ {len(docs)} chunks stored")
            
            return True
        except Exception as e:
            print(f"  ❌ Storage error: {e}")
            return False
    
    def search(self, session_id: str, query: str, full_transcript: str, top_k: int = 5) -> List[Dict]:
        try:
            collection = self.client.get_collection(f"session_{session_id}")
            model = self._load_embeddings()
            
            results = collection.query(
                query_embeddings=[model.encode(query).tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i] if results.get('metadatas') else {}
                    start = meta.get('start_pos', -1)
                    
                    if start >= 0 and full_transcript:
                        exp_start = max(0, start - 400)
                        exp_end = min(len(full_transcript), start + len(doc) + 400)
                        expanded = full_transcript[exp_start:exp_end]
                    else:
                        expanded = doc
                    
                    formatted.append({
                        'text': doc,
                        'expanded': expanded,
                        'score': round(1 - results['distances'][0][i], 4),
                        'index': meta.get('chunk_index', i)
                    })
            
            return formatted
        except Exception as e:
            print(f"  ❌ Search error: {e}")
            return []
    
    def close_and_cleanup(self):
        """Properly close and cleanup the database"""
        try:
            # Clear embedding model
            if self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            # Reset database
            if self.client is not None:
                try:
                    self.client.reset()
                except:
                    pass
                
                # Clear client reference
                del self.client
                self.client = None
            
            # Force garbage collection
            gc.collect()
            
            # Wait for file handles to release
            time.sleep(1)
            
            # Now delete files
            if self.db_path.exists():
                # Try to delete chroma.sqlite3
                sqlite_file = self.db_path / "chroma.sqlite3"
                for attempt in range(5):
                    if sqlite_file.exists():
                        try:
                            sqlite_file.unlink()
                            print("  ✅ Deleted chroma.sqlite3")
                            break
                        except:
                            time.sleep(0.5)
                            gc.collect()
                
                # Delete UUID folders
                for item in self.db_path.iterdir():
                    if item.is_dir() and len(item.name) == 36 and item.name.count('-') == 4:
                        try:
                            shutil.rmtree(item)
                            print(f"  ✅ Deleted: {item.name[:12]}...")
                        except:
                            pass
            
            return True
        except Exception as e:
            print(f"  ⚠️ Cleanup warning: {e}")
            return False


# =============================================================================
# GROQ LLM - FIXED
# =============================================================================

class GroqLLM:
    """Groq LLM interface with intelligent prompting"""
    
    def __init__(self):
        self.client = None
        self.model = GROQ_MODEL
    
    def _get_client(self):
        if self.client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set in .env file!")
            self.client = Groq(api_key=GROQ_API_KEY)
        return self.client
    
    def generate(self, question: str, context: str, video_title: str,
                 content_type: ContentType, chat_history: List[ChatMessage] = None) -> Dict:
        """Generate response with appropriate prompting"""
        try:
            client = self._get_client()
            
            # Get appropriate system prompt
            system_prompt = EliteSystemPrompts.get_prompt(video_title, content_type)
            
            # Build user message based on content type
            if content_type == ContentType.GREETING:
                # Simple message for greetings
                user_message = f"The user said: {question}"
            else:
                # Full context for other types
                user_message = f'''**Video Transcript Context:**
{context[:8000]}

---

**User Question:** {question}

Please respond appropriately based on the question type.'''
            
            # Build messages
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add history (limited)
            if chat_history and content_type != ContentType.GREETING:
                for msg in chat_history[-4:]:
                    content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                    messages.append({"role": msg.role, "content": content})
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048 if content_type in [ContentType.GREETING, ContentType.GENERAL] else 4096,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content + CREATOR_FOOTER
            
            return {
                'success': True,
                'answer': answer,
                'time': round(time.time() - start, 2),
                'tokens': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_connection(self) -> bool:
        try:
            self._get_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            return True
        except:
            return False


# =============================================================================
# MAIN PIPELINE - CLEAN OUTPUT
# =============================================================================

class VectorPlexPipeline:
    """Main pipeline with clean output"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("   🚀 VectorPlex V2 - RAG Pipeline")
        print("   By Sangam Gautam & Sushil Yadav")
        print("="*60)
        
        self.downloader = VideoDownloader(AUDIO_PATH)
        self.transcriber = WhisperTranscriber()
        self.chunker = TextChunker()
        self.vectorstore = VectorStore(CHROMA_PATH)
        self.llm = GroqLLM()
        
        self.session: Optional[VideoSession] = None
        
        print(f"\n[Config] Model: {GROQ_MODEL}")
        print(f"[Config] Whisper: {WHISPER_MODEL}")
        print("[Pipeline] Ready!\n")
    
    def process_video(self, url: str) -> bool:
        """Process video through complete pipeline"""
        session_id = str(uuid.uuid4())[:8]
        
        self.session = VideoSession(
            session_id=session_id,
            video_url=url
        )
        
        print(f"\n{'='*60}")
        print(f"📹 PROCESSING VIDEO")
        print(f"{'='*60}")
        print(f"Session: {session_id}\n")
        
        try:
            # Step 1: Download
            print("📥 STEP 1/4: Download")
            print("-" * 40)
            self.session.status = SessionStatus.DOWNLOADING
            
            dl = self.downloader.download(url, session_id)
            if not dl.get('success'):
                raise Exception(f"Download failed: {dl.get('error')}")
            
            self.session.video_title = dl['title']
            self.session.audio_path = dl['audio_path']
            self.session.duration = dl['duration']
            print()
            
            # Step 2: Transcribe
            print("🎤 STEP 2/4: Transcribe")
            print("-" * 40)
            self.session.status = SessionStatus.TRANSCRIBING
            
            tr = self.transcriber.transcribe(dl['audio_path'], session_id)
            if not tr.get('success'):
                raise Exception(f"Transcription failed: {tr.get('error')}")
            
            self.session.transcript = tr['transcript']
            self.session.word_count = tr['word_count']
            self.session.language = tr['language']
            print()
            
            # Step 3: Chunk
            print("✂️  STEP 3/4: Chunk")
            print("-" * 40)
            self.session.status = SessionStatus.PROCESSING
            
            chunks = self.chunker.chunk(tr['transcript'])
            self.session.chunk_count = len(chunks)
            print(f"  ✅ Created {len(chunks)} chunks")
            print()
            
            # Step 4: Store
            print("🗄️  STEP 4/4: Vector Store")
            print("-" * 40)
            
            self.vectorstore.create_collection(session_id)
            self.vectorstore.add_chunks(session_id, chunks, tr['transcript'])
            print()
            
            self.session.status = SessionStatus.READY
            
            print("="*60)
            print("✅ READY TO CHAT!")
            print("="*60)
            print(f"📹 {self.session.video_title}")
            print(f"📝 {self.session.word_count} words | {self.session.chunk_count} chunks")
            print("="*60 + "\n")
            
            return True
            
        except Exception as e:
            self.session.status = SessionStatus.ERROR
            self.session.error = str(e)
            print(f"\n❌ ERROR: {e}\n")
            self.cleanup()
            return False
    
    def chat(self, question: str) -> Optional[str]:
        """Chat with processed video - CLEAN OUTPUT"""
        if not self.session or self.session.status != SessionStatus.READY:
            print("❌ No video processed!")
            return None
        
        try:
            # Detect content type
            content_type = ContentDetector.detect(question, self.session.transcript[:2000])
            
            # For greetings, minimal context needed
            if content_type == ContentType.GREETING:
                context = f"Video title: {self.session.video_title}"
            elif content_type == ContentType.SUMMARY:
                context = self.session.transcript[:18000]
            else:
                # Use vector search
                results = self.vectorstore.search(
                    self.session.session_id,
                    question,
                    self.session.transcript,
                    top_k=4
                )
                
                if results:
                    parts = [r['expanded'] for r in results]
                    context = "\n\n---\n\n".join(parts)
                else:
                    context = self.session.transcript[:12000]
            
            # Generate response (no debug output)
            result = self.llm.generate(
                question=question,
                context=context,
                video_title=self.session.video_title,
                content_type=content_type,
                chat_history=self.session.chat_history
            )
            
            if not result.get('success'):
                print(f"❌ Error: {result.get('error')}")
                return None
            
            # Save to history
            self.session.chat_history.append(ChatMessage(role="user", content=question))
            self.session.chat_history.append(ChatMessage(role="assistant", content=result['answer']))
            
            return result['answer']
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def cleanup(self):
        """Complete cleanup"""
        print("\n🧹 Cleaning up...")
        
        session_id = self.session.session_id if self.session else None
        
        # Delete audio files
        if session_id:
            for ext in ['mp3', 'webm', 'm4a', 'opus', 'ogg', 'wav', 'mp4']:
                audio_file = AUDIO_PATH / f"{session_id}.{ext}"
                if audio_file.exists():
                    try:
                        audio_file.unlink()
                    except:
                        pass
        print("  ✅ Audio cleaned")
        
        # Delete transcripts
        if session_id:
            transcript_file = TRANSCRIPT_PATH / f"{session_id}.txt"
            if transcript_file.exists():
                try:
                    transcript_file.unlink()
                except:
                    pass
        print("  ✅ Transcripts cleaned")
        
        # Close and cleanup vector store
        self.vectorstore.close_and_cleanup()
        print("  ✅ Vectors cleaned")
        
        self.session = None
        print("✅ Cleanup complete!\n")
    
    def get_info(self) -> Dict:
        """Get session info"""
        if not self.session:
            return {'active': False}
        return {
            'title': self.session.video_title,
            'words': self.session.word_count,
            'chunks': self.session.chunk_count
        }


# =============================================================================
# INTERACTIVE CLI - CLEAN OUTPUT
# =============================================================================

def run_cli():
    """Run interactive CLI with clean output"""
    print("\n" + "="*60)
    print("   🎬 VECTORPLEX V2")
    print("   By Sangam Gautam & Sushil Yadav")
    print("="*60)
    print("\n1. Enter a video URL")
    print("2. Ask questions about the video")
    print("3. Type /quit to exit")
    print("="*60 + "\n")
    
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env!")
        return
    
    pipeline = VectorPlexPipeline()
    
    print("Testing API...")
    if pipeline.llm.test_connection():
        print("✅ API connected!\n")
    else:
        print("❌ API failed!")
        return
    
    try:
        while True:
            if not pipeline.session or pipeline.session.status != SessionStatus.READY:
                url = input("📹 Video URL: ").strip()
                
                if url.lower() in ['quit', '/quit', 'exit', '/exit', 'q']:
                    break
                
                if not url:
                    continue
                
                if not pipeline.process_video(url):
                    continue
            
            # Chat loop
            print("\n💬 Ask anything about the video (/quit to exit, /new for new video)\n")
            
            while True:
                try:
                    question = input("You: ").strip()
                    
                    if not question:
                        continue
                    
                    if question.lower() in ['/quit', '/exit', 'quit', 'exit']:
                        raise KeyboardInterrupt
                    
                    if question.lower() in ['/new', 'new']:
                        pipeline.cleanup()
                        break
                    
                    if question.lower() in ['/info', 'info']:
                        info = pipeline.get_info()
                        print(f"\n📊 {info}\n")
                        continue
                    
                    # Get response
                    response = pipeline.chat(question)
                    if response:
                        print(f"\n{'─'*60}")
                        print(response)
                        print(f"{'─'*60}\n")
                    
                except EOFError:
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    finally:
        pipeline.cleanup()
        print("Thanks for using VectorPlex V2! 🚀\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ["--help", "-h"]:
            print("""
VectorPlex V2 - Video Chat Pipeline

Usage:
  python pipeline.py              Interactive mode
  python pipeline.py <url>        Process URL directly
  python pipeline.py --cleanup    Force cleanup all data
  python pipeline.py --help       Show help
            """)
        
        elif arg == "--cleanup":
            print("\n🧹 Force cleanup...")
            
            for f in AUDIO_PATH.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            print("  ✅ Audio cleaned")
            
            for f in TRANSCRIPT_PATH.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            print("  ✅ Transcripts cleaned")
            
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH, ignore_errors=True)
                CHROMA_PATH.mkdir(exist_ok=True)
            print("  ✅ Vectors cleaned")
            
            print("✅ Done!\n")
        
        elif arg.startswith("http"):
            pipeline = VectorPlexPipeline()
            try:
                if pipeline.process_video(arg):
                    print("\n💬 Chat (Ctrl+C to exit):\n")
                    while True:
                        q = input("You: ").strip()
                        if q:
                            r = pipeline.chat(q)
                            if r:
                                print(f"\n{r}\n")
            except KeyboardInterrupt:
                pass
            finally:
                pipeline.cleanup()
        else:
            print(f"Unknown: {arg}")
    else:
        run_cli()


