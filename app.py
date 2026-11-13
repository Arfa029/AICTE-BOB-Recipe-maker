"""
Flask Backend API — Recipe Preparation Agent
Exposes REST endpoints consumed by the frontend.
"""

import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "recipe-agent-secret")
CORS(app)

# ---------------------------------------------------------------------------
# Lazy agent initialization (avoids slow startup on import)
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        logger.info("Initializing Recipe Agent (first request)...")
        from agent import RecipeAgent
        _agent = RecipeAgent()
        logger.info("Recipe Agent initialized.")
    return _agent


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Recipe Preparation Agent",
        "powered_by": "IBM WatsonX Granite AI + RAG",
        "model": os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"),
    })


@app.route("/api/status", methods=["GET"])
def status():
    """Return configuration status (without exposing secrets)."""
    api_key = os.getenv("IBM_API_KEY", "")
    project_id = os.getenv("WATSONX_PROJECT_ID", "")
    return jsonify({
        "ibm_api_key_set": bool(api_key and api_key != "your_ibm_cloud_api_key_here"),
        "watsonx_project_id_set": bool(project_id and project_id != "your_watsonx_project_id_here"),
        "watsonx_url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        "granite_model": os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"),
        "rag_top_k": int(os.getenv("RAG_TOP_K", "5")),
    })


# ---------------------------------------------------------------------------
# Core Recipe Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/suggest", methods=["POST"])
def suggest_recipes():
    """
    POST /api/suggest
    Body: {
        "ingredients": ["egg", "spinach", "cheese"],
        "preferences": {
            "dietary": ["vegetarian"],
            "servings": 2,
            "difficulty": "Easy",
            "time": 30
        }
    }
    Returns recipe suggestions powered by IBM Granite + RAG.
    """
    data = request.get_json(silent=True) or {}
    ingredients = data.get("ingredients", [])
    preferences = data.get("preferences", {})

    if not ingredients or not isinstance(ingredients, list):
        return jsonify({"error": "Please provide a non-empty 'ingredients' list."}), 400

    # Sanitize inputs
    ingredients = [str(i).strip().lower() for i in ingredients if str(i).strip()]
    if not ingredients:
        return jsonify({"error": "All ingredient entries are empty."}), 400

    try:
        agent = get_agent()
        result = agent.suggest_recipes(
            ingredients=ingredients,
            preferences=preferences if preferences else None,
            top_k=3,
        )
        return jsonify({
            "success": True,
            "query_ingredients": result["query_ingredients"],
            "recipes": result["recipes"],
            "ai_response": result["ai_response"],
            "top_recipe": result["top_recipe"],
        })
    except Exception as e:
        logger.exception("Error in /api/suggest")
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


@app.route("/api/substitution", methods=["POST"])
def get_substitution():
    """
    POST /api/substitution
    Body: {
        "ingredient": "butter",
        "recipe_name": "Banana Oat Pancakes",
        "context": "making it vegan"
    }
    Returns IBM Granite substitution suggestions.
    """
    data = request.get_json(silent=True) or {}
    ingredient = data.get("ingredient", "").strip()
    recipe_name = data.get("recipe_name", "").strip()
    context = data.get("context", "").strip()

    if not ingredient:
        return jsonify({"error": "Please provide an 'ingredient' to substitute."}), 400
    if not recipe_name:
        return jsonify({"error": "Please provide a 'recipe_name'."}), 400

    try:
        agent = get_agent()
        response = agent.get_substitution(ingredient, recipe_name, context)
        return jsonify({
            "success": True,
            "ingredient": ingredient,
            "recipe_name": recipe_name,
            "substitutions": response,
        })
    except Exception as e:
        logger.exception("Error in /api/substitution")
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


@app.route("/api/tips", methods=["POST"])
def cooking_tips():
    """
    POST /api/tips
    Body: { "recipe_name": "Spaghetti Aglio e Olio", "technique": "garlic sautéing" }
    Returns cooking tips from IBM Granite.
    """
    data = request.get_json(silent=True) or {}
    recipe_name = data.get("recipe_name", "").strip()
    technique = data.get("technique", "").strip()

    if not recipe_name:
        return jsonify({"error": "Please provide a 'recipe_name'."}), 400

    try:
        agent = get_agent()
        tips = agent.get_cooking_tips(recipe_name, technique)
        return jsonify({
            "success": True,
            "recipe_name": recipe_name,
            "tips": tips,
        })
    except Exception as e:
        logger.exception("Error in /api/tips")
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


@app.route("/api/dietary-adjust", methods=["POST"])
def dietary_adjust():
    """
    POST /api/dietary-adjust
    Body: {
        "recipe_id": "R001",
        "dietary_requirement": "vegan"
    }
    Returns dietary adaptation advice from IBM Granite.
    """
    data = request.get_json(silent=True) or {}
    recipe_id = data.get("recipe_id", "").strip()
    dietary_req = data.get("dietary_requirement", "").strip()

    if not recipe_id or not dietary_req:
        return jsonify({"error": "Provide 'recipe_id' and 'dietary_requirement'."}), 400

    # Load recipe from JSON
    recipes_path = os.getenv("RECIPES_DATA_PATH", "./data/recipes.json")
    try:
        with open(recipes_path, "r", encoding="utf-8") as f:
            all_recipes = {r["id"]: r for r in json.load(f)}
    except FileNotFoundError:
        return jsonify({"error": "Recipe database not found."}), 500

    recipe = all_recipes.get(recipe_id)
    if not recipe:
        return jsonify({"error": f"Recipe '{recipe_id}' not found."}), 404

    try:
        agent = get_agent()
        adjustment = agent.adjust_for_dietary(recipe, dietary_req)
        return jsonify({
            "success": True,
            "recipe": recipe,
            "dietary_requirement": dietary_req,
            "adjustment": adjustment,
        })
    except Exception as e:
        logger.exception("Error in /api/dietary-adjust")
        return jsonify({"error": f"Agent error: {str(e)}"}), 500


@app.route("/api/recipes", methods=["GET"])
def list_recipes():
    """
    GET /api/recipes
    Returns the full recipe list from the knowledge base.
    Optional query params: ?category=Pasta&dietary=vegan&cuisine=Italian
    """
    recipes_path = os.getenv("RECIPES_DATA_PATH", "./data/recipes.json")
    try:
        with open(recipes_path, "r", encoding="utf-8") as f:
            recipes = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Recipe database not found."}), 500

    # Apply optional filters
    category = request.args.get("category", "").lower()
    dietary = request.args.get("dietary", "").lower()
    cuisine = request.args.get("cuisine", "").lower()

    if category:
        recipes = [r for r in recipes if r.get("category", "").lower() == category]
    if dietary:
        recipes = [r for r in recipes if any(dietary in d.lower() for d in r.get("dietary", []))]
    if cuisine:
        recipes = [r for r in recipes if r.get("cuisine", "").lower() == cuisine]

    return jsonify({
        "success": True,
        "count": len(recipes),
        "recipes": recipes,
    })


@app.route("/api/ingredients", methods=["GET"])
def list_ingredients():
    """GET /api/ingredients — returns all known ingredients in the knowledge base."""
    from rag_engine import get_all_ingredients_from_db
    ingredients = get_all_ingredients_from_db()
    return jsonify({"success": True, "ingredients": ingredients})


@app.route("/api/rebuild-index", methods=["POST"])
def rebuild_index():
    """
    POST /api/rebuild-index
    Forces a rebuild of the ChromaDB vector index from recipes.json.
    Useful after adding new recipes.
    """
    try:
        from rag_engine import initialize_vector_store
        initialize_vector_store(force_rebuild=True)
        return jsonify({"success": True, "message": "Vector index rebuilt successfully."})
    except Exception as e:
        logger.exception("Error rebuilding index")
        return jsonify({"error": f"Rebuild failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Serve the frontend SPA
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


@app.errorhandler(404)
def not_found(e):
    # For SPA routing, serve index.html for unknown routes
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    logger.info(f"Starting Recipe Preparation Agent on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
