"""
RAG Vector Store — Recipe Retrieval Engine
Uses ChromaDB as the vector database and sentence-transformers for local embeddings.
On first run, it ingests all recipes from recipes.json into the vector store.
On subsequent runs, it loads the persisted database.
"""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports to avoid slow startup when modules are not needed
# ---------------------------------------------------------------------------

_chroma_client = None
_collection = None
_embedder = None

COLLECTION_NAME = "recipe_knowledge_base"
RECIPES_DATA_PATH = os.getenv("RECIPES_DATA_PATH", "./data/recipes.json")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))


# ---------------------------------------------------------------------------
# Embedding model (local sentence-transformers, no API key needed)
# ---------------------------------------------------------------------------

def _get_embedder():
    """Load the sentence-transformer embedding model (cached after first load)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer embedding model...")
        # all-MiniLM-L6-v2 is lightweight (~80MB) and works well for recipe matching
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _embedder


def _embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a text string."""
    embedder = _get_embedder()
    return embedder.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# ChromaDB client and collection management
# ---------------------------------------------------------------------------

def _get_chroma_client():
    """Get or create the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info(f"ChromaDB client initialized at: {CHROMA_PERSIST_DIR}")
    return _chroma_client


def _recipe_to_document(recipe: dict) -> str:
    """
    Convert a recipe dict into a rich searchable text document.
    This text is what gets embedded and stored in the vector store.
    """
    ingredients_str = ", ".join(recipe.get("ingredients", []))
    dietary_str = ", ".join(recipe.get("dietary", []))
    instructions_str = " ".join(recipe.get("instructions", []))
    tips = recipe.get("tips", "")

    doc = (
        f"Recipe: {recipe.get('name', '')}. "
        f"Category: {recipe.get('category', '')}. "
        f"Cuisine: {recipe.get('cuisine', '')}. "
        f"Dietary: {dietary_str}. "
        f"Ingredients needed: {ingredients_str}. "
        f"How to cook: {instructions_str} "
        f"Tips: {tips} "
        f"Difficulty: {recipe.get('difficulty', '')}. "
        f"Prep time: {recipe.get('prep_time', '')}. "
        f"Cook time: {recipe.get('cook_time', '')}."
    )
    return doc


def initialize_vector_store(force_rebuild: bool = False) -> None:
    """
    Initialize the ChromaDB collection.
    - If the collection already has data and force_rebuild=False, skip ingestion.
    - Otherwise, ingest all recipes from recipes.json.
    """
    global _collection
    client = _get_chroma_client()

    existing_collections = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing_collections and not force_rebuild:
        _collection = client.get_collection(COLLECTION_NAME)
        count = _collection.count()
        if count > 0:
            logger.info(f"Loaded existing recipe collection with {count} recipes.")
            return

    # Create or recreate the collection
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection for rebuild.")

    _collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Load recipes
    if not os.path.exists(RECIPES_DATA_PATH):
        logger.error(f"Recipes data file not found: {RECIPES_DATA_PATH}")
        return

    with open(RECIPES_DATA_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    logger.info(f"Ingesting {len(recipes)} recipes into the vector store...")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for recipe in recipes:
        doc_text = _recipe_to_document(recipe)
        embedding = _embed_text(doc_text)

        ids.append(recipe["id"])
        documents.append(doc_text)
        embeddings.append(embedding)
        metadatas.append({
            "name": recipe.get("name", ""),
            "category": recipe.get("category", ""),
            "cuisine": recipe.get("cuisine", ""),
            "dietary": json.dumps(recipe.get("dietary", [])),
            "difficulty": recipe.get("difficulty", ""),
            "prep_time": recipe.get("prep_time", ""),
            "cook_time": recipe.get("cook_time", ""),
            "servings": str(recipe.get("servings", "")),
            "ingredients": json.dumps(recipe.get("ingredients", [])),
        })

    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Successfully ingested {len(recipes)} recipes.")


def _get_collection():
    """Get or initialize the ChromaDB collection."""
    global _collection
    if _collection is None:
        initialize_vector_store()
    return _collection


# ---------------------------------------------------------------------------
# Query / retrieval functions
# ---------------------------------------------------------------------------

def retrieve_recipes(
    query: str,
    top_k: int = RAG_TOP_K,
    dietary_filter: Optional[list[str]] = None,
) -> list[dict]:
    """
    Retrieve the most relevant recipes for a user query using semantic similarity.

    Args:
        query: User's ingredient list or natural-language query.
        top_k: Number of top results to retrieve.
        dietary_filter: Optional list of dietary tags to filter (e.g. ["vegan", "gluten-free"]).

    Returns:
        List of recipe dicts with an added 'similarity_score' field.
    """
    collection = _get_collection()

    # Embed the user query
    query_embedding = _embed_text(query)

    # Build where clause for dietary filter
    where_clause = None
    # Note: ChromaDB metadata stores dietary as JSON string; direct filtering is limited.
    # We'll filter post-retrieval for dietary preferences.

    # Retrieve from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, collection.count()),  # over-fetch for post-filter
        include=["documents", "metadatas", "distances"],
    )

    # Load the original recipe JSON for full data
    with open(RECIPES_DATA_PATH, "r", encoding="utf-8") as f:
        all_recipes = {r["id"]: r for r in json.load(f)}

    retrieved = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for recipe_id, distance, metadata in zip(ids, distances, metadatas):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - distance/2
        similarity = 1.0 - (distance / 2.0)

        if similarity < RAG_SIMILARITY_THRESHOLD:
            continue

        recipe = all_recipes.get(recipe_id, {})
        if not recipe:
            continue

        # Apply dietary filter if provided
        if dietary_filter:
            recipe_dietary = recipe.get("dietary", [])
            if not any(d.lower() in [rd.lower() for rd in recipe_dietary] for d in dietary_filter):
                continue

        recipe["similarity_score"] = round(similarity, 4)
        retrieved.append(recipe)

    # Sort by similarity and return top_k
    retrieved.sort(key=lambda x: x["similarity_score"], reverse=True)
    return retrieved[:top_k]


def ingredient_match_score(user_ingredients: list[str], recipe: dict) -> dict:
    """
    Calculate how many of the user's ingredients match the recipe.

    Returns:
        dict with 'matched', 'missing', 'match_percentage', and 'match_score'.
    """
    recipe_ingredients = [i.lower().strip() for i in recipe.get("ingredients", [])]
    user_set = [i.lower().strip() for i in user_ingredients]

    matched = []
    missing = []

    for req_ing in recipe_ingredients:
        # Check if any user ingredient is a substring match
        found = any(
            req_ing in user_ing or user_ing in req_ing
            for user_ing in user_set
        )
        if found:
            matched.append(req_ing)
        else:
            missing.append(req_ing)

    total = len(recipe_ingredients)
    match_pct = round((len(matched) / total * 100) if total > 0 else 0, 1)

    return {
        "matched": matched,
        "missing": missing,
        "match_percentage": match_pct,
        "match_score": len(matched) / total if total > 0 else 0,
    }


def get_all_ingredients_from_db() -> list[str]:
    """Return a deduplicated sorted list of all ingredients across all recipes."""
    if not os.path.exists(RECIPES_DATA_PATH):
        return []
    with open(RECIPES_DATA_PATH, "r", encoding="utf-8") as f:
        recipes = json.load(f)
    all_ingredients = set()
    for recipe in recipes:
        for ing in recipe.get("ingredients", []):
            all_ingredients.add(ing.lower().strip())
    return sorted(all_ingredients)
