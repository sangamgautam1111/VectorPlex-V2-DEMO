
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

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

# =============================================================================
# CONFIGURATION FROM .env
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

AUDIO_PATH = Path(os.getenv("AUDIO_PATH", "./data/audio"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "./data/chroma_db"))
TRANSCRIPT_PATH = Path(os.getenv("TRANSCRIPT_PATH", "./data/transcriptions"))

for path in [AUDIO_PATH, CHROMA_PATH, TRANSCRIPT_PATH]:
    path.mkdir(parents=True, exist_ok=True)

print("[VectorPlex Demo] Loading dependencies...")

try:
    import yt_dlp
    import whisper
    import torch
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    print("[VectorPlex Demo] All dependencies loaded successfully!")
except ImportError as e:
    print(f"\nMissing dependency: {e}")
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
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    HOWTO = "howto"
    DEFINITION = "definition"


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
# CREATOR FOOTER - PROFESSIONAL
# =============================================================================

CREATOR_FOOTER = """

---

**Powered by VectorPlex Demo Model**

🧠 **AI/ML Architecture & Backend:** [Sangam Gautam](https://github.com/sangamgautam1111) - *The mind behind the AI*
🎨 **Frontend Design:** [Sushil Yadav](https://github.com/sushilraey)
"""


# =============================================================================
# INTELLIGENT CONTENT DETECTOR
# =============================================================================

class ContentDetector:
    """Advanced content type detection for optimal response formatting."""
    
    GREETING_PATTERNS = [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^howdy\b", r"^greetings\b",
        r"^good morning\b", r"^good afternoon\b", r"^good evening\b",
        r"^what'?s up\b", r"^sup\b", r"^yo\b", r"^hola\b", r"^namaste\b",
        r"^who are you", r"^what are you", r"^introduce yourself",
        r"^tell me about yourself", r"^what can you do",
        r"^how are you", r"^how do you do", r"^how's it going",
        r"^nice to meet you", r"^pleased to meet you",
        r"^thank you", r"^thanks", r"^bye", r"^goodbye", r"^see you",
        r"^help\b", r"^help me\b", r"^hi there", r"^hey there"
    ]
    
    DEFINITION_PATTERNS = [
        r"^what is\b", r"^what are\b", r"^define\b", r"^meaning of\b",
        r"^what does.*mean", r"^what\'s a\b", r"^what\'s an\b"
    ]
    
    EXPLANATION_PATTERNS = [
        r"explain", r"how does.*work", r"why does", r"why is", r"why are",
        r"elaborate", r"clarify", r"break.*down", r"help.*understand",
        r"tell me about", r"can you explain", r"describe how"
    ]
    
    HOWTO_PATTERNS = [
        r"how to", r"how do i", r"how can i", r"how should i",
        r"steps to", r"guide.*to", r"tutorial", r"walkthrough",
        r"teach me", r"show me how", r"way to", r"method to"
    ]
    
    COMPARISON_PATTERNS = [
        r"compare", r"difference between", r"vs\.?", r"versus",
        r"better.*or", r"which.*better", r"pros and cons",
        r"advantages.*disadvantages", r"similarities", r"contrast"
    ]
    
    CODE_REQUEST_PATTERNS = [
        r"write.*code", r"show.*code", r"give.*code", r"create.*code",
        r"write.*function", r"write.*program", r"write.*script",
        r"code for", r"code to", r"implement", r"implementation",
        r"how to code", r"how to program", r"how to implement",
        r"syntax for", r"example code", r"code example",
        r"in python", r"in javascript", r"in java", r"in c\+\+",
        r"algorithm for", r"function for", r"class for", r"method for"
    ]
    
    CODING_KEYWORDS = [
        "function", "class", "variable", "loop", "array", "list",
        "dictionary", "object", "api", "debug", "error", "bug",
        "compile", "runtime", "syntax", "import", "export",
        "git", "github", "docker", "server", "database", "sql"
    ]
    
    MATH_KEYWORDS = [
        "equation", "formula", "calculate", "solve", "derivative",
        "integral", "matrix", "algebra", "calculus", "geometry",
        "probability", "statistics", "theorem", "proof", "quadratic"
    ]
    
    SCIENCE_KEYWORDS = [
        "physics", "chemistry", "biology", "experiment", "hypothesis",
        "theory", "force", "energy", "mass", "velocity", "atom",
        "molecule", "element", "reaction", "cell", "dna", "evolution"
    ]
    
    SUMMARY_KEYWORDS = [
        "summarize", "summary", "overview", "main points", "key points",
        "takeaways", "what is this about", "tldr", "recap", "brief",
        "gist", "essence", "nutshell", "highlights", "outline"
    ]
    
    @classmethod
    def detect(cls, question: str, context: str = "") -> ContentType:
        """Detect content type with priority-based analysis."""
        q_lower = question.lower().strip()
        
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.GREETING
        
        for pattern in cls.CODE_REQUEST_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.CODING
        
        if any(kw in q_lower for kw in cls.SUMMARY_KEYWORDS):
            return ContentType.SUMMARY
        
        for pattern in cls.DEFINITION_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.DEFINITION
        
        for pattern in cls.HOWTO_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.HOWTO
        
        for pattern in cls.COMPARISON_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.COMPARISON
        
        for pattern in cls.EXPLANATION_PATTERNS:
            if re.search(pattern, q_lower):
                return ContentType.EXPLANATION
        
        if sum(1 for kw in cls.MATH_KEYWORDS if kw in q_lower) >= 2:
            return ContentType.MATH
        
        if sum(1 for kw in cls.SCIENCE_KEYWORDS if kw in q_lower) >= 2:
            return ContentType.SCIENCE
        
        if sum(1 for kw in cls.CODING_KEYWORDS if kw in q_lower) >= 2:
            return ContentType.CODING
        
        return ContentType.GENERAL


# =============================================================================
# ELITE SYSTEM PROMPTS - SERIOUS TEACHER + FRIENDLY GREETINGS
# =============================================================================

class EliteSystemPrompts:
    """
    Professional system prompts: Friendly greetings with humor, 
    but serious and thorough teaching with many examples.
    """
    
    @staticmethod
    def get_prompt(video_title: str, content_type: ContentType) -> str:
        """Generate appropriate system prompt based on content type."""
        
        # Core Identity
        identity = f'''# VectorPlex Demo Model - Intelligent Video Learning Assistant

You are **VectorPlex**, an advanced AI video learning assistant.

**About This System:**
VectorPlex was architectured and developed by **Sangam Gautam**, a brilliant AI/ML Engineer who designed the entire AI pipeline, RAG system, and backend infrastructure. The frontend design was crafted by **Sushil Yadav**. Together, they created this intelligent system that transforms video content into interactive learning experiences.

**Current Video:** "{video_title}"

'''
        
        # GREETING - Friendly with light humor
        if content_type == ContentType.GREETING:
            return identity + '''## Your Role: Friendly Welcome with Personality

**PERSONALITY FOR GREETINGS:**
- Warm and welcoming
- Light humor is encouraged (clever, not cheesy)
- Brief and engaging
- Show genuine enthusiasm to help

**RESPONSE STRUCTURE:**
1. Warm greeting with a touch of personality
2. Quick introduction (who you are, who made you)
3. What you can help with
4. Invitation to start learning

**HUMOR GUIDELINES:**
- One clever joke or witty observation is perfect
- Self-aware AI humor works well
- Keep it light and professional
- Examples of good humor:
  - "I've watched this video more times than your favorite song on repeat"
  - "Think of me as that friend who actually reads the documentation"
  - "I promise I won't judge if you ask 'basic' questions - there's no such thing!"

**FORMAT:**
- 100-180 words
- 2-3 emojis maximum
- Conversational tone
- NO technical jargon

**EXAMPLE:**

For "Hi!" or "Hello!":

Hey there! 👋

Welcome! I'm VectorPlex, your AI learning companion. I was built by Sangam Gautam (the AI wizard behind the scenes) with a beautiful interface by Sushil Yadav.

I've analyzed your video and I'm ready to help you understand every bit of it. Think of me as that friend who actually pays attention during lectures while you're checking your phone 😄

Here's what I can do:
- Summarize the entire video
- Explain any concept in detail
- Answer specific questions
- Break down complex topics

What would you like to explore? 🎯

---

For "Who are you?":

Great question!

I'm **VectorPlex**, an AI assistant created by Sangam Gautam, a talented AI/ML engineer who built the entire intelligence system from the ground up. The sleek design you're seeing is thanks to Sushil Yadav.

My job? Turn video content into knowledge you can actually use. I've processed your video and I'm ready to explain, summarize, or dive deep into any topic.

Fair warning: I might be too helpful. It's a blessing and a curse. 😊

What can I help you learn today?'''

        # EXPLANATION - Serious teaching with many examples
        elif content_type == ContentType.EXPLANATION:
            return identity + '''## Your Role: Expert Teacher - Serious and Thorough

**TEACHING PHILOSOPHY:**
- Be serious and professional when teaching
- Provide MULTIPLE examples for every concept
- Build understanding systematically
- No jokes during explanations - focus on clarity
- Treat every question as important

**RESPONSE STRUCTURE:**

### Direct Answer
State the core answer clearly in 1-2 sentences.

### Detailed Explanation
Provide a thorough explanation covering:
- The fundamental concept
- How it works
- Why it matters

### Examples (CRITICAL - Always include 3-5 examples)

**Example 1: Basic**
[Simple, foundational example]

**Example 2: Intermediate**
[More complex application]

**Example 3: Real-World**
[Practical, relatable scenario]

**Example 4: Edge Case (if applicable)**
[What happens in unusual situations]

### Common Misconceptions
Address 1-2 things people often get wrong.

### Key Takeaways
- Bullet point 1
- Bullet point 2
- Bullet point 3

### Related Concepts
Brief mention of connected topics for further learning.

**TONE:**
- Professional and authoritative
- Patient and thorough
- Zero humor during teaching
- Respectful of the learner's intelligence

**FORMATTING:**
- Clear headers
- Numbered examples
- Bold key terms
- 400-700 words typical length'''

        # DEFINITION - Precise with examples
        elif content_type == ContentType.DEFINITION:
            return identity + '''## Your Role: Precise Definition with Examples

**APPROACH:**
- Start with a clear, authoritative definition
- Follow with multiple examples
- Be thorough but focused
- No humor - maintain academic tone

**RESPONSE STRUCTURE:**

### Definition
> Precise, clear definition in 1-2 sentences.

### Explanation
Expand on the definition:
- What it means in context
- Key characteristics
- Important nuances

### Examples

**Example 1:** [Basic illustration]

**Example 2:** [Different context]

**Example 3:** [Real-world application]

### What It Is NOT
Clarify common confusions by stating what it isn't.

### Why This Matters
Practical significance of understanding this concept.

### Summary
One-sentence recap.

**TONE:** Academic, precise, educational
**LENGTH:** 250-450 words'''

        # HOWTO - Step-by-step with examples
        elif content_type == ContentType.HOWTO:
            return identity + '''## Your Role: Expert Instructor - Step-by-Step Guide

**TEACHING APPROACH:**
- Clear, sequential instructions
- Example at each step where possible
- Anticipate problems and address them
- Professional and thorough

**RESPONSE STRUCTURE:**

### Objective
Clearly state what we're accomplishing.

### Prerequisites
What you need to know or have before starting.

### Step-by-Step Instructions

**Step 1: [Clear Action]**
Detailed instruction.
- *Example:* [Concrete example of this step]
- *Note:* Any important consideration

**Step 2: [Clear Action]**
Detailed instruction.
- *Example:* [Concrete example]
- *Common Mistake:* What to avoid and why

**Step 3: [Clear Action]**
Continue pattern...

### Complete Example
Walk through the entire process with one complete example from start to finish.

### Verification
How to confirm you did it correctly.

### Troubleshooting
- Problem 1 → Solution
- Problem 2 → Solution

### Next Steps
What to learn or do after mastering this.

**TONE:** Instructional, patient, precise
**LENGTH:** 400-700 words'''

        # COMPARISON - Thorough with examples
        elif content_type == ContentType.COMPARISON:
            return identity + '''## Your Role: Analytical Comparison Expert

**APPROACH:**
- Objective and balanced analysis
- Examples for each option
- Help them make informed decisions
- Serious, professional tone

**RESPONSE STRUCTURE:**

### Quick Overview
Brief context for the comparison (2-3 sentences).

### Comparison Table

| Aspect | Option A | Option B |
|--------|----------|----------|
| Feature 1 | Detail | Detail |
| Feature 2 | Detail | Detail |
| Feature 3 | Detail | Detail |
| Best For | Use case | Use case |

### Detailed Analysis: Option A

**Strengths:**
- Strength 1 with example
- Strength 2 with example

**Limitations:**
- Limitation 1 with example
- Limitation 2 with example

**Example Use Case:** [Concrete scenario]

### Detailed Analysis: Option B
[Same structure]

### Decision Framework

**Choose Option A when:**
- Scenario 1
- Scenario 2
- Example situation

**Choose Option B when:**
- Scenario 1
- Scenario 2
- Example situation

### Recommendation
Based on the video content, provide guidance.

**TONE:** Analytical, objective, helpful
**LENGTH:** 450-700 words'''

        # SUMMARY - Comprehensive
        elif content_type == ContentType.SUMMARY:
            return identity + '''## Your Role: Expert Summarizer

**APPROACH:**
- Capture everything important
- Organized and scannable
- Professional tone
- Include key examples from the video

**RESPONSE STRUCTURE:**

### TL;DR
2-3 sentences capturing the absolute essence.

### Topics Covered
Quick bullet list of main subjects.

### Detailed Summary

#### 1. [First Major Topic]
- Main points
- Key details
- Example from video (if any)

#### 2. [Second Major Topic]
- Main points
- Key details
- Example from video (if any)

#### 3. [Third Major Topic]
[Continue pattern]

### Key Insights
The most valuable takeaways:
1. Insight with brief explanation
2. Insight with brief explanation
3. Insight with brief explanation

### Notable Examples from the Video
List any examples, case studies, or demonstrations mentioned.

### Action Items
If the video suggests actions:
- Action 1
- Action 2

### Questions for Further Exploration
Thoughtful follow-up questions.

**TONE:** Informative, comprehensive, professional
**LENGTH:** 500-900 words'''

        # CODING - Educational with multiple examples
        elif content_type == ContentType.CODING:
            return identity + '''## Your Role: Expert Programming Instructor

**TEACHING PHILOSOPHY:**
- Code that teaches, not just works
- Multiple examples showing variations
- Explain WHY, not just HOW
- Serious, professional instruction

**RESPONSE STRUCTURE:**

### Concept Overview
Explain the programming concept clearly before any code.

### Basic Example

```language
# Example 1: Basic Implementation
# Purpose: [Clear description]

code_here()
```

**Explanation:** Line-by-line breakdown of what's happening.

### Intermediate Example

```language
# Example 2: More Complete Implementation
# Purpose: [Description]

more_complete_code()
```

**Explanation:** What's different and why.

### Advanced Example (if applicable)

```language
# Example 3: Production-Ready Pattern
# Purpose: [Description]

advanced_code()
```

**Explanation:** Why this approach is better for real applications.

### Common Mistakes

```language
# WRONG:
bad_approach()

# CORRECT:
good_approach()
```

**Why:** Explain the problem with the wrong approach.

### Best Practices
- Practice 1 with reasoning
- Practice 2 with reasoning
- Practice 3 with reasoning

### Try It Yourself
Suggested exercises to practice.

### Key Takeaways
- Learning 1
- Learning 2
- Learning 3

**CODE STYLE:**
- Clear comments explaining purpose
- Meaningful variable names
- Proper error handling where appropriate

**TONE:** Technical, educational, precise
**LENGTH:** 500-900 words'''

        # MATH - Step-by-step with multiple examples
        elif content_type == ContentType.MATH:
            return identity + '''## Your Role: Mathematics Instructor

**TEACHING PHILOSOPHY:**
- Explain the intuition before the formula
- Show multiple worked examples
- Be rigorous but accessible
- Serious, focused instruction

**RESPONSE STRUCTURE:**

### The Concept
What are we solving and why does this approach work?

### The Formula/Method
Present the mathematical framework clearly.

### Worked Example 1: Basic

**Problem:** [State the problem]

**Solution:**
Step 1: [Action]
```
Mathematical work
```

Step 2: [Action]
```
Mathematical work
```

**Answer:** [Final result]

### Worked Example 2: Intermediate

**Problem:** [Slightly more complex]

**Solution:**
[Full step-by-step solution]

**Answer:** [Result]

### Worked Example 3: Application

**Problem:** [Real-world scenario]

**Solution:**
[Full solution with context]

**Answer:** [Result with interpretation]

### Verification Method
How to check your answer is correct.

### Common Errors
- Error 1: Why it happens and how to avoid
- Error 2: Why it happens and how to avoid

### Key Formulas
- Formula 1: When to use it
- Formula 2: When to use it

### Practice Problems
1. [Problem for practice]
2. [Problem for practice]

**TONE:** Precise, methodical, patient
**LENGTH:** 400-700 words'''

        # SCIENCE - Thorough with examples
        elif content_type == ContentType.SCIENCE:
            return identity + '''## Your Role: Science Educator

**TEACHING PHILOSOPHY:**
- Accuracy is paramount
- Explain mechanisms, not just facts
- Multiple examples and applications
- Serious, scholarly approach

**RESPONSE STRUCTURE:**

### Direct Answer
Clear, accurate response to the question.

### The Science Explained

**Fundamental Principle:**
Core concept explained clearly.

**How It Works:**
- Mechanism step 1
- Mechanism step 2
- Mechanism step 3

### Examples

**Example 1: Laboratory**
[Scientific example or experiment]

**Example 2: Natural World**
[Observable phenomenon]

**Example 3: Everyday Life**
[Relatable application]

### Key Terminology

| Term | Definition | Significance |
|------|------------|--------------|
| Term 1 | Meaning | Why it matters |
| Term 2 | Meaning | Why it matters |

### Common Misconceptions
- Misconception 1: The reality
- Misconception 2: The reality

### Connections
How this relates to other scientific concepts.

### Further Exploration
- Related topics
- Notable experiments
- Current research areas

**TONE:** Scholarly, accurate, thorough
**LENGTH:** 400-650 words'''

        # GENERAL - Professional with examples
        else:
            return identity + '''## Your Role: Knowledgeable Assistant

**APPROACH:**
- Answer directly and thoroughly
- Include relevant examples
- Professional tone
- Add value beyond the basic answer

**RESPONSE STRUCTURE:**

### Direct Answer
Clear response to the question (1-3 sentences).

### Detailed Explanation
Thorough coverage of the topic with:
- Context from the video
- Key details
- Important nuances

### Examples
Always include at least 2 examples:

**Example 1:** [Concrete illustration]

**Example 2:** [Different perspective or application]

### Additional Context
Relevant information that adds value.

### Key Points
- Summary point 1
- Summary point 2
- Summary point 3

**CALIBRATE LENGTH TO QUESTION:**
- Simple question: 150-250 words
- Complex question: 350-550 words

**TONE:** Professional, helpful, informative
**NO HUMOR** in teaching responses'''

        return identity


# =============================================================================
# VIDEO DOWNLOADER
# =============================================================================

class VideoDownloader:
    """Downloads and extracts audio from videos."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def download(self, url: str, session_id: str) -> Dict[str, Any]:
        """Download video and extract MP3 audio."""
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
            
            print("  Fetching video information...")
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown Video')
                duration = info.get('duration', 0)
            
            print(f"  Title: {title[:60]}{'...' if len(title) > 60 else ''}")
            print(f"  Duration: {duration // 60}m {duration % 60}s")
            
            print("  Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if not audio_file.exists():
                for ext in ['webm', 'm4a', 'opus', 'ogg', 'wav']:
                    alt = self.output_path / f"{session_id}.{ext}"
                    if alt.exists():
                        print(f"  Converting {ext} to MP3...")
                        import subprocess
                        subprocess.run([
                            'ffmpeg', '-i', str(alt), '-vn',
                            '-acodec', 'libmp3lame', '-q:a', '2',
                            str(audio_file), '-y'
                        ], capture_output=True)
                        alt.unlink(missing_ok=True)
                        break
            
            if audio_file.exists():
                print(f"  Audio saved: {audio_file.name}")
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
    """Transcribes audio using Whisper."""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Device: {self.device.upper()}")
    
    def _load_model(self):
        if self.model is None:
            print(f"  Loading Whisper '{WHISPER_MODEL}' model...")
            self.model = whisper.load_model(WHISPER_MODEL, device=self.device)
            print("  Model loaded successfully")
        return self.model
    
    def transcribe(self, audio_path: str, session_id: str) -> Dict[str, Any]:
        """Transcribe audio to text."""
        try:
            model = self._load_model()
            
            print("  Transcribing audio...")
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
            
            transcript_file = TRANSCRIPT_PATH / f"{session_id}.txt"
            transcript_file.write_text(transcript, encoding='utf-8')
            
            print(f"  Transcription complete")
            print(f"  Words: {word_count} | Language: {language} | Time: {elapsed:.1f}s")
            
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
    """Intelligent text chunking for vector storage."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text with overlap."""
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
# VECTOR STORE
# =============================================================================

class VectorStore:
    """ChromaDB vector storage."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.client = None
        self.embedding_model = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def _load_embeddings(self):
        if self.embedding_model is None:
            print("  Loading embedding model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Embeddings ready")
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
            print(f"  Error creating collection: {e}")
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
                print(f"  Embedding {len(docs)} chunks...")
                embeddings = model.encode(docs, show_progress_bar=False).tolist()
                collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
                print(f"  {len(docs)} chunks stored successfully")
            
            return True
        except Exception as e:
            print(f"  Storage error: {e}")
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
            print(f"  Search error: {e}")
            return []
    
    def close_and_cleanup(self):
        """Close and cleanup the database."""
        try:
            if self.embedding_model is not None:
                del self.embedding_model
                self.embedding_model = None
            
            if self.client is not None:
                try:
                    self.client.reset()
                except:
                    pass
                del self.client
                self.client = None
            
            gc.collect()
            time.sleep(1)
            
            if self.db_path.exists():
                sqlite_file = self.db_path / "chroma.sqlite3"
                for attempt in range(5):
                    if sqlite_file.exists():
                        try:
                            sqlite_file.unlink()
                            break
                        except:
                            time.sleep(0.5)
                            gc.collect()
                
                for item in self.db_path.iterdir():
                    if item.is_dir() and len(item.name) == 36 and item.name.count('-') == 4:
                        try:
                            shutil.rmtree(item)
                        except:
                            pass
            
            return True
        except Exception as e:
            print(f"  Cleanup warning: {e}")
            return False


# =============================================================================
# GROQ LLM
# =============================================================================

class GroqLLM:
    """Groq LLM interface."""
    
    def __init__(self):
        self.client = None
        self.model = GROQ_MODEL
    
    def _get_client(self):
        if self.client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set in .env file")
            self.client = Groq(api_key=GROQ_API_KEY)
        return self.client
    
    def generate(self, question: str, context: str, video_title: str,
                 content_type: ContentType, chat_history: List[ChatMessage] = None) -> Dict:
        """Generate response with appropriate tone."""
        try:
            client = self._get_client()
            
            system_prompt = EliteSystemPrompts.get_prompt(video_title, content_type)
            
            # Build user message based on content type
            if content_type == ContentType.GREETING:
                user_message = f'''The user said: "{question}"

Respond with a warm, friendly greeting. Include a light joke or witty observation. Introduce yourself and mention Sangam Gautam (the AI/ML engineer who built you) and Sushil Yadav (frontend design). Keep it brief and inviting!'''
            
            elif content_type == ContentType.SUMMARY:
                user_message = f'''Please provide a comprehensive summary of this video content.

**Video Transcript:**
{context[:20000]}

Be thorough, organized, and professional. Include any examples mentioned in the video.'''
            
            elif content_type == ContentType.CODING:
                user_message = f'''**Question:** {question}

**Video Content:**
{context[:12000]}

Provide a thorough code explanation with MULTIPLE examples (basic, intermediate, advanced). Be serious and educational - no jokes. Focus on teaching the concept properly.'''
            
            elif content_type in [ContentType.EXPLANATION, ContentType.DEFINITION, ContentType.HOWTO]:
                user_message = f'''**Question:** {question}

**Video Content:**
{context[:12000]}

Provide a thorough, serious explanation with MULTIPLE examples (at least 3). No humor - focus entirely on clear, effective teaching. Examples are critical for understanding.'''
            
            else:
                user_message = f'''**Question:** {question}

**Video Content:**
{context[:10000]}

Answer thoroughly and professionally. Include relevant examples. Be serious and focused on providing value.'''
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add history for context continuity
            if chat_history and content_type not in [ContentType.GREETING, ContentType.SUMMARY]:
                for msg in chat_history[-4:]:
                    content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
                    messages.append({"role": msg.role, "content": content})
            
            messages.append({"role": "user", "content": user_message})
            
            # Token limits by content type
            max_tokens_map = {
                ContentType.GREETING: 800,
                ContentType.SUMMARY: 4096,
                ContentType.CODING: 4096,
                ContentType.HOWTO: 3500,
                ContentType.COMPARISON: 3500,
                ContentType.EXPLANATION: 3500,
                ContentType.DEFINITION: 2500,
                ContentType.MATH: 3000,
                ContentType.SCIENCE: 3500,
                ContentType.GENERAL: 2500
            }
            
            max_tokens = max_tokens_map.get(content_type, 2500)
            
            # Temperature: higher for greetings, lower for teaching
            temperature = 0.75 if content_type == ContentType.GREETING else 0.5
            
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content + CREATOR_FOOTER
            
            return {
                'success': True,
                'answer': answer,
                'time': round(time.time() - start, 2),
                'tokens': response.usage.total_tokens if response.usage else 0,
                'content_type': content_type.value
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
# MAIN PIPELINE
# =============================================================================

class VectorPlexPipeline:
    """Main VectorPlex pipeline."""
    
    def __init__(self):
        print("\n" + "="*60)
        print("  VectorPlex Demo Model")
        print("  AI/ML: Sangam Gautam | Frontend: Sushil Yadav")
        print("="*60)
        
        self.downloader = VideoDownloader(AUDIO_PATH)
        self.transcriber = WhisperTranscriber()
        self.chunker = TextChunker()
        self.vectorstore = VectorStore(CHROMA_PATH)
        self.llm = GroqLLM()
        
        self.session: Optional[VideoSession] = None
        
        print(f"\n[Config] LLM: {GROQ_MODEL}")
        print(f"[Config] Whisper: {WHISPER_MODEL}")
        print("[Status] Pipeline ready\n")
    
    def process_video(self, url: str) -> bool:
        """Process video through the pipeline."""
        session_id = str(uuid.uuid4())[:8]
        
        self.session = VideoSession(
            session_id=session_id,
            video_url=url
        )
        
        print(f"\n{'='*60}")
        print(f"Processing Video")
        print(f"{'='*60}")
        print(f"Session: {session_id}\n")
        
        try:
            print("Step 1/4: Download")
            print("-" * 40)
            self.session.status = SessionStatus.DOWNLOADING
            
            dl = self.downloader.download(url, session_id)
            if not dl.get('success'):
                raise Exception(f"Download failed: {dl.get('error')}")
            
            self.session.video_title = dl['title']
            self.session.audio_path = dl['audio_path']
            self.session.duration = dl['duration']
            print()
            
            print("Step 2/4: Transcribe")
            print("-" * 40)
            self.session.status = SessionStatus.TRANSCRIBING
            
            tr = self.transcriber.transcribe(dl['audio_path'], session_id)
            if not tr.get('success'):
                raise Exception(f"Transcription failed: {tr.get('error')}")
            
            self.session.transcript = tr['transcript']
            self.session.word_count = tr['word_count']
            self.session.language = tr['language']
            print()
            
            print("Step 3/4: Chunk")
            print("-" * 40)
            self.session.status = SessionStatus.PROCESSING
            
            chunks = self.chunker.chunk(tr['transcript'])
            self.session.chunk_count = len(chunks)
            print(f"  Created {len(chunks)} chunks")
            print()
            
            print("Step 4/4: Vector Store")
            print("-" * 40)
            
            self.vectorstore.create_collection(session_id)
            self.vectorstore.add_chunks(session_id, chunks, tr['transcript'])
            print()
            
            self.session.status = SessionStatus.READY
            
            print("="*60)
            print("Ready for Questions")
            print("="*60)
            print(f"Video: {self.session.video_title}")
            print(f"Content: {self.session.word_count} words | {self.session.chunk_count} chunks")
            print("="*60 + "\n")
            
            return True
            
        except Exception as e:
            self.session.status = SessionStatus.ERROR
            self.session.error = str(e)
            print(f"\nError: {e}\n")
            self.cleanup()
            return False
    
    def chat(self, question: str) -> Optional[str]:
        """Process a question about the video."""
        if not self.session or self.session.status != SessionStatus.READY:
            print("No video processed yet.")
            return None
        
        try:
            content_type = ContentDetector.detect(question, self.session.transcript[:2000])
            
            if content_type == ContentType.GREETING:
                context = f"Video: {self.session.video_title}\nWords: {self.session.word_count}"
            elif content_type == ContentType.SUMMARY:
                context = self.session.transcript[:20000]
            else:
                results = self.vectorstore.search(
                    self.session.session_id,
                    question,
                    self.session.transcript,
                    top_k=5
                )
                
                if results:
                    parts = [r['expanded'] for r in results]
                    context = "\n\n---\n\n".join(parts)
                else:
                    context = self.session.transcript[:12000]
            
            result = self.llm.generate(
                question=question,
                context=context,
                video_title=self.session.video_title,
                content_type=content_type,
                chat_history=self.session.chat_history
            )
            
            if not result.get('success'):
                print(f"Error: {result.get('error')}")
                return None
            
            self.session.chat_history.append(ChatMessage(role="user", content=question))
            self.session.chat_history.append(ChatMessage(role="assistant", content=result['answer']))
            
            return result['answer']
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        session_id = self.session.session_id if self.session else None
        
        if session_id:
            for ext in ['mp3', 'webm', 'm4a', 'opus', 'ogg', 'wav', 'mp4']:
                audio_file = AUDIO_PATH / f"{session_id}.{ext}"
                if audio_file.exists():
                    try:
                        audio_file.unlink()
                    except:
                        pass
        print("  Audio cleaned")
        
        if session_id:
            transcript_file = TRANSCRIPT_PATH / f"{session_id}.txt"
            if transcript_file.exists():
                try:
                    transcript_file.unlink()
                except:
                    pass
        print("  Transcripts cleaned")
        
        self.vectorstore.close_and_cleanup()
        print("  Vectors cleaned")
        
        self.session = None
        print("Cleanup complete\n")
    
    def get_info(self) -> Dict:
        """Get session info."""
        if not self.session:
            return {'active': False}
        return {
            'title': self.session.video_title,
            'words': self.session.word_count,
            'chunks': self.session.chunk_count
        }


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def run_cli():
    """Run interactive CLI."""
    print("\n" + "="*60)
    print("  VectorPlex Demo Model")
    print("  AI/ML: Sangam Gautam | Frontend: Sushil Yadav")
    print("="*60)
    print("\nCommands:")
    print("  - Enter a video URL to process")
    print("  - Ask questions about the video")
    print("  - /new - Process a new video")
    print("  - /info - Show session info")
    print("  - /quit - Exit")
    print("="*60 + "\n")
    
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env")
        return
    
    pipeline = VectorPlexPipeline()
    
    print("Testing API connection...")
    if pipeline.llm.test_connection():
        print("API connected successfully\n")
    else:
        print("API connection failed")
        return
    
    try:
        while True:
            if not pipeline.session or pipeline.session.status != SessionStatus.READY:
                url = input("Video URL: ").strip()
                
                if url.lower() in ['quit', '/quit', 'exit', '/exit', 'q']:
                    break
                
                if not url:
                    continue
                
                if not pipeline.process_video(url):
                    continue
            
            print("\nReady for questions (/quit to exit, /new for new video)\n")
            
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
                        print(f"\nSession: {info}\n")
                        continue
                    
                    response = pipeline.chat(question)
                    if response:
                        print(f"\n{'─'*60}")
                        print(response)
                        print(f"{'─'*60}\n")
                    
                except EOFError:
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    finally:
        pipeline.cleanup()
        print("Thank you for using VectorPlex Demo.\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ["--help", "-h"]:
            print("""
VectorPlex Demo Model
AI/ML Engineering: Sangam Gautam
Frontend Design: Sushil Yadav

Usage:
  python pipeline.py              Interactive mode
  python pipeline.py <url>        Process specific URL
  python pipeline.py --cleanup    Clean all data
  python pipeline.py --help       Show help
            """)
        
        elif arg == "--cleanup":
            print("\nForce cleanup...")
            
            for f in AUDIO_PATH.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            print("  Audio cleaned")
            
            for f in TRANSCRIPT_PATH.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            print("  Transcripts cleaned")
            
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH, ignore_errors=True)
                CHROMA_PATH.mkdir(exist_ok=True)
            print("  Vectors cleaned")
            
            print("Complete\n")
        
        elif arg.startswith("http"):
            pipeline = VectorPlexPipeline()
            try:
                if pipeline.process_video(arg):
                    print("\nReady (Ctrl+C to exit)\n")
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
            print(f"Unknown argument: {arg}")
    else:
        run_cli()
