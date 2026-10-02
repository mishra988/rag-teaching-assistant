import os
import sys
import re
import json
import joblib
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def format_timestamp(seconds: float) -> str:
    """Converts seconds float into MM:SS or HH:MM:SS format."""
    total_sec = int(round(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

class RAGTeachingAssistant:
    def __init__(self, embeddings_path: str = "embeddings.joblib", ollama_url: str = "http://localhost:11434"):
        self.ollama_url = os.environ.get("OLLAMA_URL", ollama_url)
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

        if os.path.exists(embeddings_path):
            print(f"[RAG] Loading precomputed video subtitle embeddings from {embeddings_path}...")
            self.df = joblib.load(embeddings_path)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            jsons_dir = os.path.join(current_dir, "jsons")
            if os.path.exists(jsons_dir):
                print(f"[RAG] Loading video subtitle chunks from {jsons_dir}...")
                my_dicts = []
                chunk_id = 0
                for json_file in sorted(os.listdir(jsons_dir)):
                    if json_file.endswith(".json"):
                        with open(os.path.join(jsons_dir, json_file), encoding="utf-8") as f:
                            content = json.load(f)
                        for chunk in content.get("chunks", []):
                            chunk_copy = dict(chunk)
                            chunk_copy["chunk_id"] = chunk_id
                            chunk_id += 1
                            my_dicts.append(chunk_copy)
                self.df = pd.DataFrame.from_records(my_dicts)
            else:
                raise FileNotFoundError(f"Neither {embeddings_path} nor '{jsons_dir}' directory found.")

        print(f"[RAG] Loaded {len(self.df)} subtitle chunks across all video lessons.")

        # Cache vector stack for cosine similarity if embeddings are present
        if "embedding" in self.df.columns and len(self.df) > 0 and hasattr(self.df["embedding"].iloc[0], "__len__") and len(self.df["embedding"].iloc[0]) > 0:
            print("[RAG] Stacking precomputed dense vectors...")
            self.embedding_matrix = np.vstack(self.df["embedding"].values)
        else:
            self.embedding_matrix = None

        # Build fallback TF-IDF vectorizer over all chunk texts for robust offline/cloud search
        print("[RAG] Indexing TF-IDF corpus for cloud & fallback retrieval...")
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=25000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["text"].fillna(""))
        print("[RAG] RAG Teaching Assistant Engine is fully initialized.")

    def _get_ollama_embedding(self, query: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": "bge-m3", "input": [query]},
                timeout=4
            )
            if r.status_code == 200:
                data = r.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    return data["embeddings"][0]
        except Exception:
            pass
        return None

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Retrieves the most relevant video subtitle chunks using dense embeddings (Ollama)
        or high-accuracy TF-IDF fallback if Ollama is not accessible.
        """
        retrieval_mode = "dense_vector (bge-m3)"
        query_vec = self._get_ollama_embedding(query)

        if query_vec is not None and self.embedding_matrix is not None:
            sims = cosine_similarity(self.embedding_matrix, [query_vec]).flatten()
        else:
            retrieval_mode = "semantic_tfidf (cloud_fallback)"
            query_tfidf = self.tfidf.transform([query])
            sims = cosine_similarity(self.tfidf_matrix, query_tfidf).flatten()

        top_indices = sims.argsort()[::-1][:top_k]
        top_df = self.df.iloc[top_indices].copy()
        top_df["similarity_score"] = sims[top_indices]
        return top_df, retrieval_mode

    def _query_groq_llm(self, prompt: str) -> str:
        if not self.groq_api_key:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are an expert, friendly teaching assistant for the Sigma Web Development course."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 800
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=12)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] Groq error: {e}")
        return None

    def _query_openai_llm(self, prompt: str) -> str:
        if not self.openai_api_key:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            body = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are an expert teaching assistant for the Sigma Web Development course."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=12)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] OpenAI error: {e}")
        return None

    def _query_ollama_llm(self, prompt: str) -> str:
        try:
            res = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=15
            )
            if res.status_code == 200:
                return res.json().get("response")
        except Exception:
            pass
        return None

    def _generate_extractive_summary(self, query: str, top_df: pd.DataFrame) -> str:
        """
        Intelligent fallback response generator if external LLMs / Ollama are unavailable,
        ensuring guaranteed 100% uptime on Render with exact video timestamps and context!
        """
        # Group by video
        videos_dict = {}
        for _, row in top_df.iterrows():
            vid = f"Video {row['number']}: {row['title']}"
            if vid not in videos_dict:
                videos_dict[vid] = []
            videos_dict[vid].append(row)

        primary_vid = list(videos_dict.keys())[0]
        
        lines = [
            f"Here is where **\"{query}\"** is taught in the Sigma Web Development Course:\n"
        ]

        for vid_name, chunks in videos_dict.items():
            lines.append(f"### 📹 **{vid_name}**")
            for chunk in chunks:
                start_fmt = format_timestamp(chunk['start'])
                end_fmt = format_timestamp(chunk['end'])
                excerpt = chunk['text'].strip()
                lines.append(f"- ⏱️ **Timestamp [{start_fmt} - {end_fmt}]**: *\"{excerpt}\"*")
            lines.append("")

        lines.append(f"💡 **Recommended Learning Path:** Start by watching **{primary_vid}**, where the instructor explains this concept in depth. Follow along in VS Code and try out the examples shown in the video!")
        return "\n".join(lines)

    def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline:
        Query -> Retrieve Top Chunks -> Prompt Synthesis -> LLM Generation / Fallback -> Formatted Output
        """
        clean_query = query.strip()
        top_df, retrieval_mode = self.retrieve_relevant_chunks(clean_query, top_k=top_k)

        # Prepare source chunks list with formatted timestamps
        sources = []
        for _, row in top_df.iterrows():
            sources.append({
                "video_number": str(row["number"]),
                "video_title": str(row["title"]),
                "start_seconds": float(row["start"]),
                "end_seconds": float(row["end"]),
                "timestamp_formatted": f"{format_timestamp(row['start'])} - {format_timestamp(row['end'])}",
                "text": str(row["text"]).strip(),
                "similarity_score": round(float(row.get("similarity_score", 0.0)), 4)
            })

        # Construct RAG Prompt
        prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{top_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{clean_query}"
User asked this question related to the video chunks. You have to answer in a warm, helpful teaching assistant tone (don't mention the raw JSON or format, it is just for you).
Explain clearly where and how much content is taught in which video (mention the exact video title, number, and timestamps formatted nicely) and guide the student to go to that particular video. If the query is completely unrelated to web development or this course, politely guide them back to course topics.
'''

        # Try LLM providers in order: Groq -> OpenAI -> Ollama -> Structured Extractive Summary
        response_text = None
        provider_used = "Ollama Local"

        if self.groq_api_key:
            response_text = self._query_groq_llm(prompt)
            if response_text:
                provider_used = "Groq (Llama-3.3-70B)"

        if not response_text and self.openai_api_key:
            response_text = self._query_openai_llm(prompt)
            if response_text:
                provider_used = "OpenAI (GPT-4o-mini)"

        if not response_text:
            response_text = self._query_ollama_llm(prompt)
            if response_text:
                provider_used = "Ollama (Llama-3.2)"

        if not response_text:
            response_text = self._generate_extractive_summary(clean_query, top_df)
            provider_used = "Course RAG Indexer (High-Precision Extractive Synthesis)"

        return {
            "query": clean_query,
            "answer": response_text,
            "provider": provider_used,
            "retrieval_mode": retrieval_mode,
            "sources": sources
        }
