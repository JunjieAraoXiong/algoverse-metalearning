"""Google Gemini provider."""

from .base import LLMProvider, LLMResponse


class GoogleProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def _create_client(self):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Gemini combines system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self.client.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        content = ""
        if response and response.text:
            content = response.text

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }

        return LLMResponse(
            content=content,
            model=self.model_name,
            provider="google",
            usage=usage,
        )

    @property
    def provider_name(self) -> str:
        return "google"
