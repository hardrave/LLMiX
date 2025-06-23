"""
LLMIX  –  Raspberry-Pi-5 “LLM-in-a-box”
FastAPI · WebSockets · llama-cpp-python (token streaming) + FAISS RAG
▸ 2025-06  ·  Granite | Medicanite | Qwen | Wikipedia RAG
▸ Created by Dominika & Michal with <3
"""

import asyncio, uuid, datetime, pickle, threading
from collections import deque
from typing import Optional, List, Dict, Deque, Tuple

import faiss, numpy as np
from fastapi             import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles  import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic             import BaseModel
from llama_cpp            import Llama
from sentence_transformers import SentenceTransformer

# ───────── CONFIG ────────────────────────────────────────────────────
MODELS = {
    "Granite"    : "models/granite.gguf",
    "Qwen"       : "models/qwen.gguf",      
    "RAG"        : "models/granite.gguf",
    "Medicanite" : "models/medicanite.gguf",
}

N_THREADS, N_CTX, N_BATCH = 4, 2048, 1024
USE_MLOCK, MAX_HIST       = False, 30               # Pi-5 friendly

# token-budget knobs
MAX_GEN_TOKENS         = 512    # how many tokens we let the model emit
RESERVED_ANSWER_TOKENS = 512    # same number — keep that much free in context

INDEX_PATH = "data/wiki.faiss"
CHUNK_PATH = "data/wiki.pkl"
TOP_K      = 3
# ─────────────────────────────────────────────────────────────────────


# ────────  Pydantic helpers ──────────────────────────────────────────
class User   (BaseModel): name: str; sid: str
class Msg    (BaseModel):
    user: str
    text: str
    is_bot: bool
    qid: int
    chunks: Optional[List[str]] = None
class Prompt (BaseModel): user: str; text: str; model: str; qid: int


# ────────  Global state ─────────────────────────────────────────────
class State:
    def __init__(self):
        self.loaded_model : Optional[str]   = None   # what’s in RAM
        self.users        : List [User]     = []
        self.convo        : List [Msg]      = []
        self.queue        : Deque[Prompt]   = deque()
        self.next_qid     : int             = 1
        self.busy         : bool            = False
        self.lock                           = asyncio.Lock()
S = State()

SOCKETS: Dict[str, WebSocket] = {}
async def broadcast(ev: str, data):
    dead=[]
    for sid, ws in SOCKETS.items():
        try:   await ws.send_json({"event": ev, "data": data})
        except Exception: dead.append(sid)
    for sid in dead: SOCKETS.pop(sid, None)


# ────────  llama-cpp singleton loader ───────────────────────────────
_LLAMA: Optional[Llama] = None
_LL_LOCK = asyncio.Lock()

async def get_llama(requested: str) -> Llama:
    global _LLAMA
    async with _LL_LOCK:
        if S.loaded_model != requested:
            _LLAMA = None
            S.loaded_model = requested
        if _LLAMA is None:
            _LLAMA = Llama(model_path = MODELS[requested],
                           n_threads  = N_THREADS,
                           n_ctx      = N_CTX,
                           n_batch    = N_BATCH,
                           use_mlock  = USE_MLOCK)
        return _LLAMA


# ────────  RAG helpers ──────────────────────────────────────────────
INDEX = CHUNKS = EMB = None
RAG_LOCK = threading.Lock()

def load_rag():
    global INDEX, CHUNKS, EMB
    if INDEX is not None: return
    with RAG_LOCK:
        if INDEX is None:
            print("[RAG] loading FAISS + MiniLM …")
            INDEX  = faiss.read_index(INDEX_PATH)
            CHUNKS = pickle.load(open(CHUNK_PATH, "rb"))
            EMB    = SentenceTransformer("models/bge-small-en-v1.5", device="cpu")
            EMB.max_seq_length = 1024

def retrieve_chunks(query: str, k: int = TOP_K) -> list[str]:
    load_rag()
    INDEX.hnsw.efSearch = 128                     # ❶ deeper search

    vec = EMB.encode([query], normalize_embeddings=True).astype("float32")
    _, idx = INDEX.search(vec, 32)                # ❷ wider candidate set
    cands = [CHUNKS[i] for i in idx[0] if i != -1]

    # ❸ simple lexical filter
    kw = {w.lower() for w in query.split() if len(w) > 3}
    filtered = [c for c in cands if any(w in c.lower() for w in kw)]
    return (filtered or cands)[:k]


# ────────  prompt / streaming utils ─────────────────────────────────
def sys_msg(model: str) -> str:
    today = datetime.datetime.now().strftime("%B %d %Y")
    if model == "Granite":     return f"You are a helpful assistant. Today is {today}."
    if model == "Qwen":       return f"You are a helpful coding assistant. Today is {today}."
    if model == "Medicanite":  return f"You are a helpful medical assistant. Today is {today}."
    return f"Wikipedia-RAG assistant · answer **only** from context · {today}"

# model → visible assistant name
def bot_alias(model: str) -> str:
    return {
        "Granite"    : "llmix",
        "Qwen"       : "llmix-code",
        "RAG"        : "llmix-wiki",
        "Medicanite" : "llmix-med",
    }.get(model, "llmix")

def token_of(chunk) -> str:
    try:   return chunk.choices[0].delta.content or ""
    except: return (chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                   or chunk.get("choices", [{}])[0].get("text") or "")

def num_tokens(llm: Llama, txt: str) -> int:
    """Count tokens the model sees for given text."""
    return len(llm.tokenize(txt.encode("utf-8")))


# ────────  generate answer (true token streaming) ───────────────────
async def llm_answer(pr: Prompt) -> Tuple[str, str, Optional[List[str]]]:
    llama  = await get_llama(pr.model)
    alias  = bot_alias(pr.model)

    context = ""
    top     = None
    if pr.model == "RAG":
        top     = retrieve_chunks(pr.text)
        context = "\n\n---\n".join(top)

    # build dialogue history
    hist=[{"role":"system","content":sys_msg(pr.model)}]
    if context:
        hist.append({"role":"system",
                     "content":f"Context (Simple-English Wikipedia):\n---\n{context}\n---"})
    for m in S.convo[-MAX_HIST:]:
        hist.append({"role":"assistant" if m.is_bot else "user", "content": m.text})
    hist.append({"role":"user", "content": pr.text})

    # ── trim oldest turns to guarantee room for answer ──────────────
    allowed_ctx = N_CTX - RESERVED_ANSWER_TOKENS
    while True:
        used = sum(num_tokens(llama, m["content"]) for m in hist)
        if used <= allowed_ctx or len(hist) <= 3:  # always keep sys+latest user
            break
        hist.pop(2)                                # drop oldest real turn

    # streaming worker
    q   = asyncio.Queue()
    loop = asyncio.get_running_loop()
    def worker():
        for ch in llama.create_chat_completion(messages   = hist,
                                               temperature = 0.7,
                                               max_tokens  = MAX_GEN_TOKENS,
                                               stream      = True):
            tok = token_of(ch)
            if tok:
                loop.call_soon_threadsafe(q.put_nowait, tok)

    # notify UI
    await broadcast("assistant_started", {"qid": pr.qid, "alias": alias})
    if context:
        await broadcast("rag_context",
                        {"qid": pr.qid, "chunks": context.split("---\n")})

    fut  = loop.run_in_executor(None, worker)
    full = ""
    while True:
        try:
            tok = await asyncio.wait_for(q.get(), 0.1)
            full += tok
            await broadcast("stream_token", {"qid": pr.qid, "token": tok})
            q.task_done()
        except asyncio.TimeoutError:
            if fut.done() and q.empty():
                break
    await fut
    return full.strip(), alias, (top if pr.model == "RAG" else None)


# ────────  queue processor ──────────────────────────────────────────
async def process_queue():
    if S.busy: return
    S.busy = True
    while True:
        async with S.lock:
            if not S.queue:
                S.busy = False
                return
            pr = S.queue.popleft()

        # user message
        S.convo.append(Msg(user=pr.user, text=pr.text, is_bot=False, qid=pr.qid))
        await broadcast("conversation_update", [m.dict() for m in S.convo])
        await broadcast("queue_update",
                        [f"{i+1}:{q.user}" for i, q in enumerate(S.queue)])

        # LLM / RAG answer
        try:
            ans, alias, chunks = await llm_answer(pr)
        except Exception as e:
            ans, alias, chunks = f"(error: {e})", bot_alias(pr.model), None

        S.convo.append(
            Msg(user=alias, text=ans, is_bot=True, qid=pr.qid, chunks=chunks)
        )
        await broadcast("conversation_update", [m.dict() for m in S.convo])


# ────────  FastAPI app & WS handler ─────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    sid = str(uuid.uuid4())
    SOCKETS[sid] = ws

    # initial sync
    await ws.send_json({"event": "conversation_update",
                        "data" : [m.dict() for m in S.convo]})
    await ws.send_json({"event": "queue_update",
                        "data" : [f"{i+1}:{q.user}" for i, q in enumerate(S.queue)]})
    await ws.send_json({"event": "user_list",
                        "data" : [u.name for u in S.users]})

    uname = None
    try:
        while True:
            msg = await ws.receive_json()
            ev  = msg.get("event")
            dat = msg.get("data", {})

            if ev == "login":
                n = dat.get("name", "").strip()
                if not n or any(u.name == n for u in S.users):
                    await ws.send_json({"event": "error",
                                        "data" : {"message": "name invalid/taken"}})
                    continue
                uname = n
                async with S.lock:
                    S.users.append(User(name=n, sid=sid))
                await broadcast("user_list", [u.name for u in S.users])

            elif ev == "submit_prompt":
                txt = dat.get("text", "").strip()
                mdl = dat.get("model") or "Granite"
                if not (uname and txt and mdl in MODELS):
                    await ws.send_json({"event": "error",
                                        "data" : {"message": "bad request"}})
                    continue
                async with S.lock:
                    if any(q.user == uname for q in S.queue):
                        await ws.send_json({"event": "error",
                                            "data" : {"message": "wait for answer"}})
                        continue
                    S.queue.append(Prompt(user=uname, text=txt, model=mdl,
                                          qid=S.next_qid))
                    S.next_qid += 1
                await broadcast("queue_update",
                                [f"{i+1}:{q.user}" for i, q in enumerate(S.queue)])
                asyncio.create_task(process_queue())

    except WebSocketDisconnect:
        SOCKETS.pop(sid, None)
        if uname:
            async with S.lock:
                S.users = [u for u in S.users if u.name != uname]
        await broadcast("user_list", [u.name for u in S.users])

