"""LiteLLM router: Vertex AI Gemini as primary, Groq as automatic fallback."""

from hr_rag.config import Settings

PRIMARY = "hr-primary"
FALLBACK = "hr-fallback"


def build_router(settings: Settings):
    from litellm import Router

    model_list = [
        {
            "model_name": PRIMARY,
            "litellm_params": {
                "model": settings.primary_model,
                "vertex_project": settings.gcp_project or None,
                "vertex_location": settings.gcp_location,
                "temperature": settings.llm_temperature,
                "timeout": settings.llm_timeout_seconds,
            },
        }
    ]
    fallbacks = []
    if settings.groq_api_key:
        model_list.append(
            {
                "model_name": FALLBACK,
                "litellm_params": {
                    "model": settings.fallback_model,
                    "api_key": settings.groq_api_key,
                    "temperature": settings.llm_temperature,
                    "timeout": settings.llm_timeout_seconds,
                },
            }
        )
        fallbacks = [{PRIMARY: [FALLBACK]}]

    return Router(model_list=model_list, fallbacks=fallbacks, num_retries=2)
