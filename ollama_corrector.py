import requests
import logging
import re
import os

logger = logging.getLogger(__name__)

class OllamaCorrector:
    def __init__(self, api_url: str, model: str, default_correction_prompt: str, ollama_options: dict, request_timeout: int = 300):
        """
        Initializes the OllamaCorrector.

        Args:
            api_url (str): The URL of the Ollama API.
            model (str): The Ollama model to use for correction.
            default_correction_prompt (str): The default system prompt for correction.
            ollama_options (dict): Default options for the Ollama API request.
            request_timeout (int): Timeout for the API request in seconds.
        """
        if not api_url:
            raise ValueError("Ollama API URL must be provided.")
        if not model:
            raise ValueError("Ollama model name must be provided.")
        
        self.api_url = api_url
        self.model = model
        self.default_correction_prompt = default_correction_prompt
        self.ollama_options = ollama_options if ollama_options is not None else {}
        self.request_timeout = request_timeout
        logger.info(f"OllamaCorrector initialized for model '{self.model}' at {self.api_url}")

    def correct_text(self, text_to_correct: str, original_filename: str, correction_prompt_override: str = None) -> dict:
        """
        Corrects the given text using the Ollama API.

        Args:
            text_to_correct (str): The text to be corrected.
            original_filename (str): The base name of the original file, for context in the prompt.
            correction_prompt_override (str, optional): A specific correction prompt to use instead of the default.

        Returns:
            dict: A dictionary containing:
                  'corrected_text': The corrected text (str) or None if an error occurred or correction is empty.
                  'error': An error message (str) or None if successful.
                  'original_text': The original text_to_correct.
        """
        if not text_to_correct.strip():
            logger.warning(f"Skipping correction for {original_filename}: input text is empty.")
            return {"corrected_text": None, "error": "Input text is empty", "original_text": text_to_correct}

        current_correction_prompt = correction_prompt_override or self.default_correction_prompt
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that corrects transcriptions." 
                               # This system message could also be part of the main prompt if preferred.
                },
                {
                    "role": "user",
                    "content": f"File: {original_filename}\n\n{current_correction_prompt}\n\n---\n\n{text_to_correct}"
                }
            ],
            "options": self.ollama_options,
            "stream": False 
        }

        logger.info(f"Sending correction request for {original_filename} to Ollama model {self.model} at {self.api_url}")
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

            response_data = response.json()
            corrected_text_content = response_data.get("message", {}).get("content", "")

            # Remove <thinking>...</thinking> blocks if present
            # Ensure case-insensitivity for <think> and </think> tags
            corrected_text_content = re.sub(r'<think>.*?</think>', '', corrected_text_content, flags=re.DOTALL | re.IGNORECASE)
            corrected_text_content = corrected_text_content.strip()


            if not corrected_text_content:
                logger.warning(f"Ollama returned empty correction for {original_filename}. Original text (first 100 chars): '{text_to_correct[:100]}...'")
                return {"corrected_text": None, "error": "Ollama returned empty correction", "original_text": text_to_correct}

            logger.info(f"Successfully received correction for {original_filename}.")
            return {"corrected_text": corrected_text_content, "error": None, "original_text": text_to_correct}

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama API connection error for {original_filename}: {e}. Ensure Ollama is running and reachable at {self.api_url}.")
            return {"corrected_text": None, "error": f"API Connection Error: {e}", "original_text": text_to_correct}
        except requests.exceptions.Timeout:
            logger.error(f"Ollama API request timed out for {original_filename} after {self.request_timeout}s.")
            return {"corrected_text": None, "error": "API Request Timeout", "original_text": text_to_correct}
        except requests.exceptions.RequestException as e: # Catches HTTPError too
            logger.error(f"Ollama API request failed for {original_filename}: {e}")
            if e.response is not None:
                logger.error(f"Ollama API Response Status: {e.response.status_code}, Body: {e.response.text[:500]}...") # Log first 500 chars
            return {"corrected_text": None, "error": f"API Request Failed: {e}", "original_text": text_to_correct}
        except KeyError as e:
            # This implies the response structure was not as expected
            logger.error(f"Unexpected response structure from Ollama for {original_filename}. Missing key: {e}. Full response: {response_data if 'response_data' in locals() else 'Response data not available'}")
            return {"corrected_text": None, "error": f"Unexpected API response structure (missing key: {e})", "original_text": text_to_correct}
        except Exception as e:
            logger.error(f"Unexpected error during Ollama correction of {original_filename}: {e}", exc_info=True)
            return {"corrected_text": None, "error": f"Unexpected error: {e}", "original_text": text_to_correct}

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Mock configuration (replace with your actual config loading)
    MOCK_OLLAMA_API_URL = "http://localhost:11434/api/chat" # Replace if your Ollama runs elsewhere
    MOCK_OLLAMA_MODEL = "llama2:latest" # Replace with your desired model
    MOCK_CORRECTION_PROMPT = "Please correct any spelling or grammatical errors in the following transcription. Preserve the original meaning and speaker intent. Ensure the output is clean and well-formatted."
    MOCK_OLLAMA_OPTIONS = {"temperature": 0.5}

    if not MOCK_OLLAMA_API_URL or not MOCK_OLLAMA_MODEL:
        logger.error("Mock Ollama API URL and Model must be set for example usage.")
    else:
        try:
            # Test connection (optional, but good for quick check)
            requests.get(MOCK_OLLAMA_API_URL.replace("/api/chat", "/api/tags"), timeout=5) # Assuming /api/tags is a valid endpoint to check connection
            logger.info(f"Successfully connected to Ollama base URL.")
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {MOCK_OLLAMA_API_URL}. Please ensure Ollama is running and accessible.")
            exit()


        corrector = OllamaCorrector(
            api_url=MOCK_OLLAMA_API_URL,
            model=MOCK_OLLAMA_MODEL,
            default_correction_prompt=MOCK_CORRECTION_PROMPT,
            ollama_options=MOCK_OLLAMA_OPTIONS
        )

        sample_text = "this is a test transription with sme speling erors and gramatical mistakes. i hope ollama can fix it."
        filename = "sample_test_file.txt"
        
        logger.info(f"Sending sample text for correction: '{sample_text}'")
        result = corrector.correct_text(sample_text, filename)

        if result["error"]:
            logger.error(f"Correction failed: {result['error']}")
        elif result["corrected_text"]:
            logger.info(f"Original:    '{result['original_text']}'")
            logger.info(f"Corrected:   '{result['corrected_text']}'")
        else:
            logger.warning("Correction resulted in empty text but no explicit error.") 