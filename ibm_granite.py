"""
IBM WatsonX / Granite AI Client
Handles all interactions with IBM WatsonX.ai using the ibm-watsonx-ai SDK.
Model: IBM Granite (ibm/granite-13b-instruct-v2) — available on IBM Cloud Lite tier.
"""

import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# ---------------------------------------------------------------------------
# Credentials & client singleton
# ---------------------------------------------------------------------------

_credentials = None
_api_client = None


def _get_credentials() -> Credentials:
    """Build IBM WatsonX credentials from environment variables."""
    global _credentials
    if _credentials is None:
        api_key = os.getenv("IBM_API_KEY")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        if not api_key or api_key == "your_ibm_cloud_api_key_here":
            raise ValueError(
                "IBM_API_KEY is not set. Please update your .env file with a valid IBM Cloud API key."
            )
        _credentials = Credentials(url=url, api_key=api_key)
    return _credentials


def _get_api_client() -> APIClient:
    global _api_client
    if _api_client is None:
        _api_client = APIClient(_get_credentials())
    return _api_client


# ---------------------------------------------------------------------------
# Granite LLM inference
# ---------------------------------------------------------------------------

def get_granite_model() -> ModelInference:
    """
    Return a ModelInference instance pointing at IBM Granite.
    IBM Granite model (granite-13b-instruct-v2) is available on the IBM Cloud Lite tier.
    """
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")

    if not project_id or project_id == "your_watsonx_project_id_here":
        raise ValueError(
            "WATSONX_PROJECT_ID is not set. Please update your .env file with your WatsonX project ID."
        )

    model = ModelInference(
        model_id=model_id,
        credentials=_get_credentials(),
        project_id=project_id,
        params={
            GenParams.MAX_NEW_TOKENS: 1024,
            GenParams.MIN_NEW_TOKENS: 50,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.9,
            GenParams.TOP_K: 50,
            GenParams.REPETITION_PENALTY: 1.1,
            GenParams.STOP_SEQUENCES: ["Human:", "User:"],
        },
    )
    return model


def generate_recipe_response(prompt: str) -> str:
    """
    Call IBM Granite model and return the generated text response.

    Args:
        prompt: The full prompt string to send to the model.

    Returns:
        Generated text string from the Granite model.
    """
    try:
        model = get_granite_model()
        response = model.generate_text(prompt=prompt)
        return response.strip() if isinstance(response, str) else str(response)
    except ValueError as ve:
        # Configuration error — return a friendly message
        return f"[Configuration Error] {str(ve)}"
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "[Auth Error] Invalid IBM API key. Please check your .env configuration."
        elif "403" in error_msg or "Forbidden" in error_msg:
            return "[Access Error] Your IBM account does not have access to this model. Ensure WatsonX.ai is provisioned."
        elif "404" in error_msg:
            return "[Model Error] Model not found. Check GRANITE_MODEL_ID in your .env file."
        else:
            return f"[WatsonX Error] {error_msg}"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RECIPE_AGENT_SYSTEM_PROMPT = """You are a helpful Recipe Preparation Agent powered by IBM Granite AI.
Your job is to help users cook delicious meals using only the ingredients they have on hand.
You provide:
- Tailored recipe suggestions based on available ingredients
- Step-by-step cooking instructions adapted to ingredient limitations
- Ingredient substitutions when needed
- Cooking tips and techniques
- Dietary adjustments based on user preferences
Be concise, friendly, practical, and focused on reducing food waste."""


def build_rag_prompt(
    user_query: str,
    retrieved_recipes: list[dict],
    user_preferences: dict | None = None,
) -> str:
    """
    Build a structured prompt for RAG-based recipe generation.

    Args:
        user_query: The user's input (ingredients + any preferences).
        retrieved_recipes: List of recipe dicts retrieved from the vector store.
        user_preferences: Optional dict with dietary restrictions, servings, etc.

    Returns:
        A complete prompt string to send to the Granite model.
    """
    preferences_text = ""
    if user_preferences:
        prefs = []
        if user_preferences.get("dietary"):
            prefs.append(f"Dietary restrictions: {', '.join(user_preferences['dietary'])}")
        if user_preferences.get("servings"):
            prefs.append(f"Servings needed: {user_preferences['servings']}")
        if user_preferences.get("difficulty"):
            prefs.append(f"Preferred difficulty: {user_preferences['difficulty']}")
        if user_preferences.get("time"):
            prefs.append(f"Available cooking time: {user_preferences['time']} minutes")
        if prefs:
            preferences_text = "\nUser Preferences:\n" + "\n".join(f"- {p}" for p in prefs)

    recipes_context = ""
    for i, recipe in enumerate(retrieved_recipes[:3], 1):
        recipes_context += f"""
Recipe {i}: {recipe.get('name', 'Unknown')}
- Category: {recipe.get('category', 'N/A')} | Cuisine: {recipe.get('cuisine', 'N/A')}
- Difficulty: {recipe.get('difficulty', 'N/A')} | Prep: {recipe.get('prep_time', 'N/A')} | Cook: {recipe.get('cook_time', 'N/A')}
- Required ingredients: {', '.join(recipe.get('ingredients', []))}
- Dietary info: {', '.join(recipe.get('dietary', []))}
- Instructions: {' | '.join(recipe.get('instructions', []))}
- Tips: {recipe.get('tips', 'N/A')}
- Substitutions: {recipe.get('substitutions', {})}
"""

    prompt = f"""{RECIPE_AGENT_SYSTEM_PROMPT}

=== RETRIEVED RECIPES FROM KNOWLEDGE BASE ===
{recipes_context if recipes_context else "No exact recipes found. Suggest a creative recipe based on available ingredients."}

=== USER REQUEST ===
Available ingredients: {user_query}{preferences_text}

=== TASK ===
Based on the user's available ingredients and the retrieved recipes above:
1. Recommend the BEST matching recipe (or adapt one if ingredients are partial).
2. List which user ingredients are used and which are missing.
3. Suggest substitutions for any missing key ingredients.
4. Provide clear step-by-step cooking instructions adapted to what the user has.
5. Add 2-3 practical cooking tips.
6. Note any dietary information or adjustments.

Response:"""

    return prompt


def build_substitution_prompt(ingredient: str, recipe_name: str, context: str = "") -> str:
    """Build a prompt specifically for ingredient substitution queries."""
    return f"""{RECIPE_AGENT_SYSTEM_PROMPT}

A user is making "{recipe_name}" and needs a substitution for: {ingredient}
{f"Context: {context}" if context else ""}

Provide 3 practical substitution options with brief notes on how each affects the dish.
Keep your response concise and practical.

Response:"""
