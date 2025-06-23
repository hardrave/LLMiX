# LLMix Documentation  
*A lightweight “LLM-in-a-box” for Raspberry Pi 5*  

---

## · What is LLMix?

LLMIX turns a Raspberry Pi 5 into a self-contained AI hotspot that anyone nearby can join over Wi-Fi.

| Mode | Purpose | Model |
|------|---------|-------|
| **Chat** | Everyday Q&A and conversation | `Granite-3.3-2B` |
| **Med-Chat** | Medical explanations & symptom questions | **Medicanite** (fine-tuned Granite) |
| **Code-Chat** | Programming help & examples | `Qwen 2.5 Coder 1.5B` |
| **Wiki-Chat** | Wikipedia-grounded fact answers | Granite + FAISS RAG |

No internet, no GPU, no cloud calls – everything runs **entirely on-device** with `llama.cpp`.

---

## · Why you might care

* **2.6 B** people still lack reliable internet (ITU 2024).  
* **Kiwix** & **Internet-in-a-Box** proved offline knowledge matters.  
* LLMix adds **reasoning** to that idea: live chat, code help, medical answers, RAG – anywhere a Pi can boot.

---

| Requirement | Minimum |
|-------------|---------|
| Board       | **Raspberry Pi 5** (8 GB or more recommended) |
| Storage     | 32 GB micro-SD (image ≈ 14 GB inc. models & FAISS) |
| Power       | 25.5W USB-C (5.1V - 5A) or battery bank |
| Clients     | Any phone laptop / phone / tablet / device with Wi-Fi & browser |

```text
1. Download the latest LLMIX image
2. Flash it with Raspberry Pi Imager
3. Insert SD, power on
4. Connect to Wi-Fi  ➜  SSID: LLMIX  (password: llmix/org)
5. Open http://llmix.local  ⇒   Enter a name   ⇒   Start chatting
```

## · Image Download

[LLMIX OS 1.0 (2025-06-23) — Download `.img.xz`](https://sourceforge.net/projects/llmix/files/llmix-os-20250623.img.xz/download)

---

## · Using the Web Interface

| Step | What happens |
|------|--------------|
| **Open llmix.local** | Enter a name   ⇒   Start chatting |
| **Select model** | Click *Chat*, *Med-Chat*, *Code-Chat*, or *Wiki-Chat* |
| **Type a question** | Press **Ask**; your prompt enters a queue |
| **Watch the stream** | Tokens appear live – RAG mode shows “📖 Sources” |
| **Multi-user** | Each device gets a colour; sidebar shows who’s waiting |

---

## · Architecture Snapshot

```
┌──────── Pi 5 ────────────┐
│ FastAPI  + WebSockets    │
│  ├── llama.cpp           │  (Granite / Medicanite / Qwen)
│  ├── FAISS (wiki.idx)    │  ← bge-small-en embeddings
│  └── Static UI (HTML/CSS)│
└────────────┬─────────────┘
             │ 802.11 AP (hostapd)
      Phones / Laptops connect here
```

* **Token streaming** – true incremental updates via WebSockets.  
* **hot-swap** – changing models just reloads GGUF weights.  
* **Memory** – stays under 4 GB even with RAG context.

---

## · Benchmark Highlights

| Benchmark (↓) | **Medicanite** | Granite-3.3-2B | MedPhi2-2.7B |
|---------------|--------------:|---------------:|-------------:|
| Average (9 tasks) | **55.82 %** | 54.86 % | 54.36 % |

Medicanite is currently the **highest-scoring open medical model &lt; 3 B params** on the Open Medical LLM Leaderboard benchmark – and it runs entirely offline on this setup.

---

## · FAQ

| Question | Answer |
|----------|--------|
| **Does LLMix need the internet?** | No. Everything – models, embeddings, FAISS – is baked into the image. |
| **How many users can connect?** | Up to ~10 concurrent devices is comfortable on a Pi 5. |
| **Is Medicanite suitable for clinical use?** | No. Educational only. Always consult a professional. |
| **Can I add my own model?** | Drop a `.gguf` into `models/`, extend `MODELS{}` in `main.py`, reboot. |


---

### 📜 LLMIX source code  
All original source files in this repository are released under the **Apache License 2.0**.  

LLMIX OS image is based on **Raspberry Pi OS (Debian GNU/Linux)**.

### 📦 Third-party components bundled in the image

| Component | Licence(s) | Notes |
|-----------|------------|-------|
| **Raspberry Pi OS (Debian GNU/Linux)** | GPL-2.0, GPL-3.0, BSD, MIT, permissive | Full licence texts are preserved under `/usr/share/doc/*/copyright`. |
| **Raspberry Pi firmware & VideoCore blobs** | Redistributable proprietary licence (see `/licenses/` in image) | Required for Pi boot and hardware acceleration. |
| **Raspberry Pi trademarks** | Trademark © Raspberry Pi Ltd. | LLMix is **not** affiliated with or endorsed by the Raspberry Pi Foundation. |
| **Granite-3.3-2B** | Apache 2.0 | Default Chat Model |
| **Medicanite**  | Apache 2.0 | Med-Chat Model |
| **Qwen 2.5 Coder** | Apache 2.0 | Code-Chat |
| **BGE Small EN v1.5** | MIT | Embedding Model |
| **Wikipedia text snippets** | CC BY-SA 4.0 | Used via offline FAISS index; attribution retained via “📖 Sources” UI. |

### 🐍 Python dependencies (bundled wheels)

| Package | Licence |
|---------|-----------|
| `faiss-cpu` | MIT |
| `numpy` | BSD 3-Clause |
| `fastapi` | MIT |
| `pydantic` | MIT |
| `llama-cpp-python` | MIT |
| `sentence-transformers` | Apache 2.0 |
| `uvicorn` | BSD 3-Clause |

All Python packages are redistributed unmodified; their LICENCE files are preserved inside the virtual-env site-packages directory.

### 🧩 Extra system packages (installed via `apt`)

| Package            | Primary licence |
|--------------------|------------------|
| python3-venv       | PSF License v2 |
| python3-dev        | PSF License v2 |
| build-essential    | GNU AGPL v3 or later |
| cmake              | BSD 3-Clause |
| ninja-build        | Apache License 2.0 |
| pkg-config         | GPL 2.0 or later |
| git                | GPL 2.0 |
| curl               | “curl” (MIT/ISC-style) |
| rsync              | GPL 3.0 |
| python3-torch      | BSD 3-Clause |
| libtorch2          | BSD 3-Clause |
| hostapd            | BSD or GPL v2 (dual) |
| dnsmasq            | GPL 2 or GPL 3 |
| dhcpcd5            | BSD 2-Clause     |
| network-manager    | GPL 2 or later (daemon/CLI) LGPL 2.1 or later (libnm) |

All of these are standard Debian/Raspberry Pi OS packages; their licence texts reside in  
`/usr/share/doc/<package>/copyright`.  
No modifications are made.

### 🔗 Source availability for GPL components  
Raspberry Pi OS and all Debian packages are unmodified builds; corresponding source code is available from:  
<https://archive.raspberrypi.org/debian/> or any official Debian mirror.


### 🙏 Best-effort notice  

We have made every reasonable effort to include or reference the licences for all third-party software and data distributed with LLMIX.  
If you believe a required notice is missing or incomplete, please open an issue or email us at contact@llmix.org and we will correct it promptly.

---

**Made with ❤️ by Dominika & Michal – bringing AI to those who need it, anywhere, anytime.**