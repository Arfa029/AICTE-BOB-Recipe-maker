"""
Recipe Preparation Agent — Core Orchestration Logic
Ties together the RAG retrieval engine and IBM Granite LLM to deliver:
  - Tailored recipe suggestions from available ingredients
  - Step-by-step instructions adapted to what the user has
  - Ingredient substitutions
  - Dietary adjustments
  - Cooking tips
"""

import logging
from typing import Optional

from rag_engine import retrieve_recipes, ingredient_match_score, initialize_vector_store
from ibm_granite import generate_recipe_response, build_rag_prompt, build_substitution_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class RecipeAgent:
    """
    Recipe Preparation Agent powered by IBM Granite (WatsonX.ai) + RAG (ChromaDB).

    Usage:
        agent = RecipeAgent()
        result = agent.suggest_recipes(
            ingredients=["eggs", "spinach", "cheese", "butter"],
            preferences={"dietary": ["vegetarian"], "servings": 2}
        )
    """

    def __init__(self):
        logger.info("Initializing Recipe Preparation Agent...")
        initialize_vector_store()
        logger.info("Recipe Agent ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_recipes(
        self,
        ingredients: list[str],
        preferences: Optional[dict] = None,
        top_k: int = 3,
    ) -> dict:
        """
        Main entry point: given a list of ingredients, return recipe suggestions.

        Args:
            ingredients: List of ingredient strings the user has available.
            preferences: Optional dict:
                {
                    "dietary": ["vegan", "gluten-free"],  # dietary filters
                    "servings": 2,
                    "difficulty": "Easy",
                    "time": 30  # max cook time in minutes
                }
            top_k: Max number of recipes to surface to the LLM.

        Returns:
            dict with:
                - "recipes": list of matched recipe dicts with match scores
                - "ai_response": IBM Granite's generated recommendation text
                - "top_recipe": the best-matched recipe dict
                - "query_ingredients": echoed back ingredient list
        """
        if not ingredients:
            return {
                "recipes": [],
                "ai_response": "Please provide a list of ingredients you have available.",
                "top_recipe": None,
                "query_ingredients": [],
            }

        # Build query from ingredients + preferences
        query = self._build_query_text(ingredients, preferences)

        # Retrieve relevant recipes via RAG
        dietary_filter = preferences.get("dietary") if preferences else None
        retrieved = retrieve_recipes(query, top_k=top_k * 2, dietary_filter=dietary_filter)

        # Rank retrieved recipes by ingredient match score
        for recipe in retrieved:
            match_info = ingredient_match_score(ingredients, recipe)
            recipe["ingredient_match"] = match_info

        # Sort by combined similarity + ingredient match
        retrieved.sort(
            key=lambda r: (
                r.get("similarity_score", 0) * 0.4 +
                r["ingredient_match"]["match_score"] * 0.6
            ),
            reverse=True,
        )
        retrieved = retrieved[:top_k]

        # Build prompt and call IBM Granite
        prompt = build_rag_prompt(
            user_query=", ".join(ingredients),
            retrieved_recipes=retrieved,
            user_preferences=preferences,
        )
        ai_response = generate_recipe_response(prompt)

        return {
            "recipes": retrieved,
            "ai_response": ai_response,
            "top_recipe": retrieved[0] if retrieved else None,
            "query_ingredients": ingredients,
        }

    def get_substitution(
        self,
        ingredient: str,
        recipe_name: str,
        context: str = "",
    ) -> str:
        """
        Ask IBM Granite for substitution suggestions for a specific ingredient.

        Args:
            ingredient: The ingredient to find a substitute for.
            recipe_name: The name of the recipe being prepared.
            context: Optional extra context.

        Returns:
            IBM Granite's substitution suggestions as a string.
        """
        prompt = build_substitution_prompt(ingredient, recipe_name, context)
        return generate_recipe_response(prompt)

    def get_cooking_tips(self, recipe_name: str, technique: str = "") -> str:
        """
        Ask IBM Granite for specific cooking tips for a recipe or technique.
        """
        from ibm_granite import RECIPE_AGENT_SYSTEM_PROMPT
        prompt = (
            f"{RECIPE_AGENT_SYSTEM_PROMPT}\n\n"
            f"The user is making \"{recipe_name}\""
            + (f" and wants tips on: {technique}" if technique else "")
            + ".\n\nProvide 5 practical, actionable cooking tips.\n\nResponse:"
        )
        return generate_recipe_response(prompt)

    def adjust_for_dietary(
        self,
        recipe: dict,
        dietary_requirement: str,
    ) -> str:
        """
        Ask IBM Granite how to adapt a specific recipe for a dietary requirement.
        """
        from ibm_granite import RECIPE_AGENT_SYSTEM_PROMPT
        ingredients = ", ".join(recipe.get("ingredients", []))
        prompt = (
            f"{RECIPE_AGENT_SYSTEM_PROMPT}\n\n"
            f"The user wants to make \"{recipe.get('name', 'this recipe')}\" "
            f"but needs it adapted for: {dietary_requirement}.\n"
            f"Original ingredients: {ingredients}\n\n"
            f"Provide specific ingredient swaps and any instruction changes needed.\n\nResponse:"
        )
        return generate_recipe_response(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_text(
        self,
        ingredients: list[str],
        preferences: Optional[dict],
    ) -> str:
        """Build the RAG query string from ingredients and preferences."""
        parts = ["I have: " + ", ".join(ingredients)]
        if preferences:
            if preferences.get("dietary"):
                parts.append("dietary: " + ", ".join(preferences["dietary"]))
            if preferences.get("difficulty"):
                parts.append("difficulty: " + preferences["difficulty"])
            if preferences.get("time"):
                parts.append(f"ready in {preferences['time']} minutes")
            if preferences.get("cuisine"):
                parts.append("cuisine: " + preferences["cuisine"])
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialized)
# ---------------------------------------------------------------------------

_agent_instance: Optional[RecipeAgent] = None


def get_agent() -> RecipeAgent:
    """Return the module-level RecipeAgent singleton."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RecipeAgent()
    return _agent_instance
