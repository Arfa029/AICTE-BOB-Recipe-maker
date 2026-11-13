# 🍳 Recipe Preparation Agent
### Problem Statement No. 16 — AICTE Project

> **A RAG-based AI cooking assistant powered by IBM WatsonX Granite AI and ChromaDB.**  
> Input your available ingredients → get tailored recipes, substitutions, dietary adjustments, and step-by-step instructions.

---

## 🏗️ Architecture

```
User Browser (HTML/CSS/JS)
        │
        ▼
Flask REST API  (app.py)
        │
        ├─── RAG Engine (rag_engine.py)
        │         ├── ChromaDB Vector Store  ← recipe embeddings
        │         └── sentence-transformers  ← local embedding model
        │
        └─── IBM Granite LLM (ibm_granite.py)
                  └── ibm-watsonx-ai SDK → IBM WatsonX.ai (Cloud Lite)
                            └── ibm/granite-13b-instruct-v2
```

---

## 📁 Project Structure

```
Receipe Maker/
├── app.py               # Flask REST API server
├── agent.py             # Core agent orchestration logic
├── ibm_granite.py       # IBM WatsonX / Granite AI client & prompt templates
├── rag_engine.py        # RAG vector store (ChromaDB + sentence-transformers)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (IBM credentials) ← CONFIGURE THIS
├── data/
│   ├── recipes.json     # Recipe knowledge base (15 recipes)
│   └── chroma_db/       # ChromaDB vector store (auto-created on first run)
├── static/
│   └── index.html       # Frontend SPA (HTML/CSS/JS)
└── README.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure IBM Cloud Credentials

Edit the `.env` file:

```env
IBM_API_KEY=your_actual_ibm_cloud_api_key
WATSONX_PROJECT_ID=your_actual_watsonx_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2
```

**How to get credentials:**
- **IBM API Key:** [https://cloud.ibm.com/iam/apikeys](https://cloud.ibm.com/iam/apikeys)
- **WatsonX Project ID:** [https://dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com) → Create a project → Copy Project ID

> ✅ **IBM Cloud Lite tier** supports WatsonX.ai with limited free usage.  
> The IBM Granite model `ibm/granite-13b-instruct-v2` is available on the Lite tier.

### 3. Run the Application

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/status` | IBM config status |
| `POST` | `/api/suggest` | Get recipe suggestions (main RAG + AI endpoint) |
| `POST` | `/api/substitution` | Get ingredient substitutions via Granite |
| `POST` | `/api/tips` | Get cooking tips via Granite |
| `POST` | `/api/dietary-adjust` | Adjust recipe for dietary needs |
| `GET` | `/api/recipes` | List/filter recipes from knowledge base |
| `GET` | `/api/ingredients` | List all known ingredients |
| `POST` | `/api/rebuild-index` | Force rebuild ChromaDB vector index |

### Example: Suggest Recipes

```bash
curl -X POST http://localhost:5000/api/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": ["eggs", "spinach", "cheese", "butter"],
    "preferences": {
      "dietary": ["vegetarian"],
      "difficulty": "Easy",
      "servings": 2
    }
  }'
```

---

## 🤖 IBM Technology Stack

| Technology | Usage | IBM Service |
|------------|-------|-------------|
| **IBM Granite 13B Instruct v2** | LLM for recipe generation, substitutions, tips | WatsonX.ai (Cloud Lite) |
| **ibm-watsonx-ai SDK** | Python client for WatsonX API | IBM Cloud |
| **IBM Cloud IAM** | API key authentication | IBM Cloud |

---

## 🔍 How RAG Works

1. **Indexing (startup):** All 15 recipes in `data/recipes.json` are converted to rich text documents and embedded using `sentence-transformers/all-MiniLM-L6-v2`, then stored in ChromaDB.

2. **Retrieval (per query):** The user's ingredient list is embedded and compared against the recipe vectors using cosine similarity. The top-K most relevant recipes are retrieved.

3. **Ingredient matching:** Each retrieved recipe is scored by how many of the user's actual ingredients overlap with the recipe's requirements.

4. **Generation (IBM Granite):** The retrieved recipes + user's ingredient list + preferences are assembled into a structured prompt sent to IBM Granite, which generates a natural-language recipe recommendation with substitutions and tips.

---

## 🥘 Recipe Knowledge Base

The knowledge base (`data/recipes.json`) contains **15 diverse recipes**:

| ID | Recipe | Cuisine | Dietary |
|----|--------|---------|---------|
| R001 | Spaghetti Aglio e Olio | Italian | Vegetarian |
| R002 | Vegetable Fried Rice | Asian | Vegetarian |
| R003 | Chicken Stir-Fry | Asian | Dairy-Free |
| R004 | Tomato Basil Pasta | Italian | Vegetarian |
| R005 | Lentil Soup | Mediterranean | Vegan, GF |
| R006 | Banana Oat Pancakes | American | Vegetarian |
| R007 | Egg Bhurji | Indian | Vegetarian, GF |
| R008 | Chana Masala | Indian | Vegan, GF |
| R009 | Avocado Toast + Egg | Modern | Vegetarian |
| R010 | Garlic Butter Mushroom Rice | Fusion | Vegetarian, GF |
| R011 | Spinach Cheese Omelette | French | Vegetarian, GF |
| R012 | Black Bean Tacos | Mexican | Vegan, GF |
| R013 | Creamy Potato Soup | American | Vegetarian, GF |
| R014 | Honey Garlic Salmon | Asian-Fusion | GF, Dairy-Free |
| R015 | Omurice | Japanese | Vegetarian |

To add more recipes, edit `data/recipes.json` and POST to `/api/rebuild-index`.

---

## 🎯 Features

- ✅ **RAG-based retrieval** — semantic search over recipe knowledge base
- ✅ **IBM Granite AI** — natural language recipe generation
- ✅ **Ingredient matching** — shows exactly which ingredients you have vs. need
- ✅ **Substitution suggestions** — IBM Granite recommends ingredient swaps
- ✅ **Dietary adjustments** — adapt any recipe for vegan, GF, keto, etc.
- ✅ **Cooking tips** — technique-specific advice from IBM Granite
- ✅ **Preference filtering** — dietary, difficulty, servings, cook time
- ✅ **Persistent vector store** — ChromaDB persists embeddings across restarts
- ✅ **Responsive UI** — dark-themed SPA with no external dependencies

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `IBM_API_KEY not set` | Add your key to `.env` |
| `401 Unauthorized` | Check your IBM API key is valid and not expired |
| `403 Forbidden` | Ensure WatsonX.ai service is provisioned in your IBM Cloud Lite account |
| `Model not found` | Verify `GRANITE_MODEL_ID` in `.env` |
| Slow first startup | Normal — `sentence-transformers` model downloads on first run (~80MB) |
| ChromaDB errors | Delete `data/chroma_db/` and restart to rebuild the index |

---

## 📄 License

MIT License — AICTE Project, Problem Statement No. 16
