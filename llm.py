from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
import random

GLOBAL_LLM = None

class LLM:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        lang: str = "English",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        retry_jitter: float = 0.5,
        max_consecutive_failures: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_jitter = retry_jitter
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            if self._consecutive_failures >= self.max_consecutive_failures:
                logger.warning("Skipping LLM request due to consecutive failures.")
                return ""
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.chat.completions.create(
                        messages=messages,
                        temperature=0,
                        model=self.model,
                        timeout=self.timeout,
                    )
                    self._consecutive_failures = 0
                    return response.choices[0].message.content
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    if attempt == self.max_retries - 1:
                        self._consecutive_failures += 1
                        logger.error("LLM request failed after all retries.")
                        return ""
                    delay = self.retry_backoff * (2 ** attempt) + random.uniform(0, self.retry_jitter)
                    sleep(delay)
            return ""
        else:
            try:
                response = self.llm.create_chat_completion(messages=messages, temperature=0)
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Local LLM request failed: {e}")
                return ""

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
