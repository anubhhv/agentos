# AgentOS — Autonomous AI Agent System

A full-stack autonomous AI agent powered by a custom GPT model trained from scratch. No external AI APIs.

---

## 🚀 Overview

AgentOS is an intelligent system that takes a goal in plain English, breaks it down, uses tools (search, code, files, weather), and generates a final answer using a locally trained language model — all in real time.

---

## ✨ Features

* 🧠 **Custom GPT Model**: Built and trained from scratch using PyTorch
* 🔄 **Autonomous Agent Loop**: Reason → Act → Observe → Respond
* 🌐 **Tool Integration**: Web search, Python execution, file reading, weather
* ⚡ **Real-time Streaming**: Live responses using Server-Sent Events (SSE)
* 🔐 **Authentication System**: JWT-based login and user management
* 📊 **Full-Stack System**: Backend + frontend with interactive UI

---

## 🏗️ Project Structure

```id="as91kd"
AgentOS/
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   └── agent.html
│
└── backend/
    ├── main.py
    ├── agent.py
    ├── local_model.py
    ├── auth.py
    ├── model.pt
    ├── vocab.json
    ├── users.json
    ├── requirements.txt
    └── tools/
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anubhhv/agentos.git
   cd agentos
   ```

2. Setup backend:

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Add model files:

   * `model.pt`
   * `vocab.json`

4. Run backend:

   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. Run frontend:

   ```bash
   cd ../frontend
   python -m http.server 5500
   ```

6. Open:

   ```
   http://localhost:5500
   ```

---

## 📈 Usage

* Sign up / log in
* Start a new session
* Enter a goal in natural language
* Agent will:

  * Analyze intent
  * Call tools if needed
  * Generate final response

---

## 🧠 AI Model

* Architecture: GPT (Transformer)
* Parameters: ~10M
* Layers: 6
* Context Length: 256 tokens
* Training Data: Wikipedia + Shakespeare
* Framework: PyTorch

---

## 🔌 API Highlights

* `POST /chat` → Main agent interaction (SSE streaming)
* `POST /auth/login` → User authentication
* `POST /session/new` → Create session
* `GET /health` → System status

---

## ⚙️ How It Works

1. User input → intent detection
2. Agent selects appropriate tools
3. Tools execute (search, code, files, etc.)
4. Results passed to local GPT
5. Final response streamed to UI

---

## 🛠️ Tools Available

* Web search
* Web content fetch
* Python execution
* File reader (PDF, CSV, JSON, TXT)
* Calculator
* Weather API

---

## 🔐 Security

* JWT-based authentication
* Password hashing with bcrypt
* Environment-based API keys

---

## 📄 License

MIT License

---

## 💡 Key Learnings

* Built a GPT model from scratch
* Implemented agent-based reasoning (ReAct pattern)
* Integrated real-time streaming with SSE
* Developed full-stack AI system end-to-end

---

**No APIs. No shortcuts. Just your own AI system.** 
