import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_engine import RAGTeachingAssistant

# Initialize RAG Engine
print("[SERVER] Initializing RAG Teaching Assistant...")
embeddings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings.joblib")
assistant = None

try:
    assistant = RAGTeachingAssistant(embeddings_path=embeddings_file)
    print(f"[SERVER] RAG Engine ready with {len(assistant.df)} chunks.")
except Exception as e:
    print(f"[SERVER] Error initializing RAG Engine: {e}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sigma Web Development - AI Teaching Assistant (RAG)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b0f19; color: #f1f5f9; }
    code, pre { font-family: 'JetBrains Mono', monospace; }
    .glass-card { background: rgba(18, 24, 38, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glass-card-hover:hover { border-color: rgba(99, 102, 241, 0.4); transform: translateY(-2px); transition: all 0.2s ease; }
    .gradient-text { background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .glow-purple { box-shadow: 0 0 40px -10px rgba(168, 85, 247, 0.3); }
    .prose p { margin-bottom: 0.75rem; line-height: 1.65; }
    .prose strong { color: #f8fafc; }
    .prose ul { list-style-type: disc; padding-left: 1.5rem; margin-bottom: 0.75rem; }
    .prose li { margin-bottom: 0.25rem; }
    .prose h3 { font-size: 1.15rem; font-weight: 700; color: #a5b4fc; margin-top: 1rem; margin-bottom: 0.5rem; }
  </style>
</head>
<body class="min-h-screen flex flex-col justify-between">
  
  <!-- Navigation Header -->
  <header class="border-b border-slate-800/80 bg-slate-950/60 sticky top-0 z-50 backdrop-blur-md">
    <div class="max-w-6xl mx-auto px-4 py-3.5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center text-white text-lg shadow-md glow-purple">
          <i class="fa-solid fa-graduation-cap"></i>
        </div>
        <div>
          <h1 class="text-base md:text-lg font-bold text-white flex items-center gap-2">
            Sigma AI Teaching Assistant
            <span class="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">RAG Engine</span>
          </h1>
          <p class="text-xs text-slate-400">Indexed from 7,252 video subtitle chunks across the complete web development course</p>
        </div>
      </div>
      <div class="flex items-center gap-2 text-xs">
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          7,252 Chunks Active
        </span>
      </div>
    </div>
  </header>

  <!-- Main Content Area -->
  <main class="max-w-5xl mx-auto px-4 py-8 flex-1 w-full space-y-8">
    
    <!-- Hero Banner -->
    <div class="text-center space-y-3 pt-2">
      <h2 class="text-3xl md:text-4xl font-extrabold tracking-tight">
        Ask Any Question from the <span class="gradient-text">Course Videos</span>
      </h2>
      <p class="text-slate-400 text-sm md:text-base max-w-2xl mx-auto">
        Retrieval-Augmented Generation searches through all lectures and extracts the exact video number, title, and timestamp interval for your query.
      </p>
    </div>

    <!-- Search Input Form -->
    <div class="glass-card rounded-2xl p-3 md:p-4 shadow-2xl relative glow-purple">
      <form id="ragForm" class="flex flex-col md:flex-row gap-3">
        <div class="relative flex-1">
          <i class="fa-solid fa-magnifying-glass absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 text-sm"></i>
          <input 
            type="text" 
            id="queryInput" 
            placeholder="e.g., Where is CSS Box Model explained? Or How do HTML forms work?" 
            class="w-full pl-11 pr-10 py-3.5 bg-slate-900/90 rounded-xl border border-slate-700/80 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm md:text-base transition"
            required
            autocomplete="off"
          />
          <button type="button" id="clearBtn" class="hidden absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white p-1">
            <i class="fa-solid fa-xmark text-sm"></i>
          </button>
        </div>
        <button 
          type="submit" 
          id="searchBtn"
          class="px-6 py-3.5 bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold rounded-xl text-sm shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition active:scale-98 whitespace-nowrap cursor-pointer"
        >
          <span id="btnText">Ask Assistant</span>
          <i id="btnIcon" class="fa-solid fa-arrow-right text-xs"></i>
          <span id="btnSpinner" class="hidden"><i class="fa-solid fa-circle-notch fa-spin"></i></span>
        </button>
      </form>

      <!-- Quick Suggestion Chips -->
      <div class="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap items-center gap-2 text-xs">
        <span class="text-slate-400 font-medium flex items-center gap-1">
          <i class="fa-regular fa-lightbulb text-amber-400"></i> Suggestions:
        </span>
        <button class="chip px-3 py-1 rounded-lg bg-slate-800/80 hover:bg-indigo-600/30 text-slate-300 hover:text-white border border-slate-700/60 transition cursor-pointer">
          Where is CSS Box Model explained?
        </button>
        <button class="chip px-3 py-1 rounded-lg bg-slate-800/80 hover:bg-indigo-600/30 text-slate-300 hover:text-white border border-slate-700/60 transition cursor-pointer">
          How do HTML forms action and method work?
        </button>
        <button class="chip px-3 py-1 rounded-lg bg-slate-800/80 hover:bg-indigo-600/30 text-slate-300 hover:text-white border border-slate-700/60 transition cursor-pointer">
          CSS Grid vs Flexbox differences
        </button>
        <button class="chip px-3 py-1 rounded-lg bg-slate-800/80 hover:bg-indigo-600/30 text-slate-300 hover:text-white border border-slate-700/60 transition cursor-pointer">
          What are semantic HTML tags?
        </button>
      </div>
    </div>

    <!-- Results Section -->
    <div id="resultsContainer" class="hidden space-y-6">
      
      <!-- AI Answer Card -->
      <div class="glass-card rounded-2xl p-6 md:p-7 shadow-xl border border-indigo-500/20 relative overflow-hidden">
        <div class="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        
        <!-- Meta Badges -->
        <div class="flex flex-wrap items-center justify-between gap-3 pb-4 mb-4 border-b border-slate-800">
          <div class="flex items-center gap-2">
            <span class="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center text-sm">
              <i class="fa-solid fa-robot"></i>
            </span>
            <span class="font-bold text-white text-sm">Teaching Assistant Response</span>
          </div>
          <div class="flex flex-wrap items-center gap-2 text-xs">
            <span id="providerBadge" class="px-2.5 py-1 rounded-md bg-purple-500/15 text-purple-300 border border-purple-500/30 font-medium"></span>
            <span id="retrievalBadge" class="px-2.5 py-1 rounded-md bg-blue-500/15 text-blue-300 border border-blue-500/30 font-medium"></span>
          </div>
        </div>

        <!-- Rendered Answer -->
        <div id="answerContent" class="prose text-slate-200 text-sm md:text-base leading-relaxed"></div>
      </div>

      <!-- Source Videos & Transcripts -->
      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-film text-indigo-400"></i>
            Retrieved Lecture Segments & Timestamps (<span id="sourceCount">0</span>)
          </h3>
          <span class="text-xs text-slate-400">Ranked by cosine similarity</span>
        </div>

        <div id="sourcesGrid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
      </div>

    </div>

    <!-- Empty State / Features Overview -->
    <div id="emptyState" class="grid grid-cols-1 md:grid-cols-3 gap-5 pt-4">
      <div class="glass-card glass-card-hover rounded-2xl p-5 space-y-2.5">
        <div class="w-10 h-10 rounded-xl bg-blue-500/15 text-blue-400 flex items-center justify-center text-lg">
          <i class="fa-solid fa-layer-group"></i>
        </div>
        <h4 class="font-bold text-white text-sm">7,252 Transcript Chunks</h4>
        <p class="text-xs text-slate-400 leading-relaxed">
          Every video from the course was transcribed, segmented into overlapping semantic intervals, and indexed for rapid nearest-neighbor lookup.
        </p>
      </div>

      <div class="glass-card glass-card-hover rounded-2xl p-5 space-y-2.5">
        <div class="w-10 h-10 rounded-xl bg-purple-500/15 text-purple-400 flex items-center justify-center text-lg">
          <i class="fa-solid fa-stopwatch"></i>
        </div>
        <h4 class="font-bold text-white text-sm">Exact MM:SS Timestamps</h4>
        <p class="text-xs text-slate-400 leading-relaxed">
          Never waste time scrubbing videos. The assistant tells you the exact lecture title and minute mark where the concept is coded live.
        </p>
      </div>

      <div class="glass-card glass-card-hover rounded-2xl p-5 space-y-2.5">
        <div class="w-10 h-10 rounded-xl bg-pink-500/15 text-pink-400 flex items-center justify-center text-lg">
          <i class="fa-solid fa-cloud-arrow-up"></i>
        </div>
        <h4 class="font-bold text-white text-sm">Cloud & Offline Resilient</h4>
        <p class="text-xs text-slate-400 leading-relaxed">
          Engineered with an automatic hybrid fallback: dense vector matching when available, with instant TF-IDF semantic search for 100% cloud uptime.
        </p>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800/80 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
    <div class="max-w-5xl mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-2">
      <p>Sigma Web Development RAG Teaching Assistant &bull; Built with Python, Scikit-Learn & Joblib</p>
      <div class="flex items-center gap-4">
        <a href="/api/health" target="_blank" class="hover:text-indigo-400 transition flex items-center gap-1">
          <i class="fa-solid fa-heart-pulse text-emerald-400"></i> Health API
        </a>
      </div>
    </div>
  </footer>

  <script>
    const form = document.getElementById('ragForm');
    const queryInput = document.getElementById('queryInput');
    const clearBtn = document.getElementById('clearBtn');
    const searchBtn = document.getElementById('searchBtn');
    const btnText = document.getElementById('btnText');
    const btnIcon = document.getElementById('btnIcon');
    const btnSpinner = document.getElementById('btnSpinner');
    const resultsContainer = document.getElementById('resultsContainer');
    const emptyState = document.getElementById('emptyState');
    const answerContent = document.getElementById('answerContent');
    const providerBadge = document.getElementById('providerBadge');
    const retrievalBadge = document.getElementById('retrievalBadge');
    const sourcesGrid = document.getElementById('sourcesGrid');
    const sourceCount = document.getElementById('sourceCount');

    queryInput.addEventListener('input', () => {
      clearBtn.classList.toggle('hidden', !queryInput.value);
    });

    clearBtn.addEventListener('click', () => {
      queryInput.value = '';
      clearBtn.classList.add('hidden');
      queryInput.focus();
    });

    document.querySelectorAll('.chip').forEach(chip => {
      chip.addEventListener('click', () => {
        queryInput.value = chip.innerText.trim();
        clearBtn.classList.remove('hidden');
        performSearch(queryInput.value);
      });
    });

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const q = queryInput.value.trim();
      if (q) performSearch(q);
    });

    async function performSearch(query) {
      setLoading(true);
      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, top_k: 6 })
        });

        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`);
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert("Failed to retrieve answer: " + err.message);
      } finally {
        setLoading(false);
      }
    }

    function setLoading(isLoading) {
      if (isLoading) {
        btnText.innerText = "Searching...";
        btnIcon.classList.add('hidden');
        btnSpinner.classList.remove('hidden');
        searchBtn.disabled = true;
        searchBtn.classList.add('opacity-75');
      } else {
        btnText.innerText = "Ask Assistant";
        btnIcon.classList.remove('hidden');
        btnSpinner.classList.add('hidden');
        searchBtn.disabled = false;
        searchBtn.classList.remove('opacity-75');
      }
    }

    function renderResults(data) {
      emptyState.classList.add('hidden');
      resultsContainer.classList.remove('hidden');

      providerBadge.innerHTML = `<i class="fa-solid fa-microchip mr-1"></i> Provider: ${data.provider}`;
      retrievalBadge.innerHTML = `<i class="fa-solid fa-bolt mr-1"></i> Retrieval: ${data.retrieval_mode}`;

      // Render Markdown Answer
      if (window.marked) {
        answerContent.innerHTML = marked.parse(data.answer);
      } else {
        answerContent.innerText = data.answer;
      }

      // Render Sources
      sourceCount.innerText = data.sources.length;
      sourcesGrid.innerHTML = '';

      data.sources.forEach(source => {
        const card = document.createElement('div');
        card.className = "glass-card glass-card-hover rounded-xl p-4 space-y-3 flex flex-col justify-between";

        const scorePercent = Math.round(source.similarity_score * 100);

        card.innerHTML = `
          <div class="space-y-2">
            <div class="flex items-start justify-between gap-2">
              <span class="text-xs font-bold px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Video #${source.video_number}
              </span>
              <span class="text-[11px] font-medium text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/50">
                Match: ${scorePercent}%
              </span>
            </div>
            <h5 class="text-sm font-bold text-white leading-snug line-clamp-2">
              ${escapeHtml(source.video_title)}
            </h5>
            <div class="inline-flex items-center gap-1.5 text-xs text-amber-300/90 font-mono bg-amber-500/10 px-2.5 py-1 rounded-md border border-amber-500/20">
              <i class="fa-regular fa-clock text-amber-400"></i>
              <span>Timestamp: ${source.timestamp_formatted}</span>
            </div>
            <p class="text-xs text-slate-300 italic bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/80 line-clamp-3">
              "${escapeHtml(source.text)}"
            </p>
          </div>
          <div class="pt-2 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/60">
            <span>Duration: ${Math.round(source.end_seconds - source.start_seconds)}s</span>
            <button onclick="navigator.clipboard.writeText('${escapeHtml(source.video_title)} [${source.timestamp_formatted}]'); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy Marker', 1500);" class="text-indigo-400 hover:text-indigo-300 transition cursor-pointer">
              <i class="fa-regular fa-copy"></i> Copy Marker
            </button>
          </div>
        `;
        sourcesGrid.appendChild(card);
      });

      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function escapeHtml(text) {
      if (!text) return '';
      const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
      return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
  </script>
</body>
</html>
"""

class RAGHTTPHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            health_data = {
                "status": "ok",
                "service": "Sigma Web Development RAG Teaching Assistant",
                "total_chunks_indexed": len(assistant.df) if assistant else 0,
                "engine_ready": assistant is not None
            }
            self.wfile.write(json.dumps(health_data).encode("utf-8"))
            return

        elif path == "/api/query":
            params = urllib.parse.parse_qs(parsed.query)
            q = params.get("q", [""])[0]
            if not q:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing query parameter 'q'"}).encode("utf-8"))
                return
            
            top_k = int(params.get("top_k", [5])[0])
            result = assistant.answer_query(q, top_k=top_k)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        else:
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
                query = payload.get("query", "").strip()
                top_k = int(payload.get("top_k", 5))

                if not query:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Field 'query' cannot be empty"}).encode("utf-8"))
                    return

                result = assistant.answer_query(query, top_k=top_k)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(b"Not Found")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

def run_server(port: int = 8000):
    server = ThreadedHTTPServer(("0.0.0.0", port), RAGHTTPHandler)
    print(f"[SERVER] Sigma RAG Assistant running live on http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Server shutting down cleanly.")
        server.shutdown()

if __name__ == "__main__":
    env_port = os.environ.get("PORT")
    if env_port and env_port.isdigit():
        port = int(env_port)
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    else:
        port = 8000
    run_server(port)
