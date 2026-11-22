"""Check the OpenAI configuration used by POST /explain (real network call, costs ~nothing).

Run from backend/ (reads backend/.env like the API does):
    python -m app.llm_check              # key present? model visible to this key?
    python -m app.llm_check --generate   # + one grounded answer on SYNTHETIC evidence

Prints no part of the API key. Exit code 0 = ready, 1 = not ready.
"""

from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from rag.llm import LLMError, get_client
from rag.pipeline import explain

# Synthetic record (not real recall data) used only to exercise the generation path.
SAMPLE_HIT = {
    "rank": 1, "recall_id": 1, "title": "Example brand Peanut Butter recalled due to Salmonella",
    "url": "https://recalls-rappels.canada.ca/en/example", "product": "Peanut butter, 500 g",
    "issue": "Microbiological - Salmonella", "category": "Nuts and seeds",
    "recall_class": "Class 1", "last_updated": "2024-01-15",
    "what_you_should_do": "Do not consume the recalled product. Throw it out or return it to "
                          "the store where it was purchased.",
    "identifier_verdict": "UNKNOWN",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generate", action="store_true",
                        help="also run one grounded generation on synthetic evidence")
    args = parser.parse_args(argv)
    config = get_settings().llm_config
    print(f"provider: {config.provider}")
    print(f"model:    {config.model}")
    print(f"api key:  {'set' if config.api_key else 'NOT SET'}")
    print(f"timeout:  {config.timeout_s}s, retries: {config.max_retries}, "
          f"max output tokens: {config.max_output_tokens}")
    if config.provider != "openai":
        print("RECALLIQ_LLM_PROVIDER is not 'openai': /explain uses the deterministic template.")
        return 1
    if not config.api_key:
        print("Set OPENAI_API_KEY in backend/.env to enable AI answers.")
        return 1
    try:
        model = get_client(config).models.retrieve(config.model)
    except LLMError as error:
        print(f"model check failed: {error}")
        return 1
    except Exception as error:  # class + status only: SDK messages can echo a masked key
        status = getattr(error, "status_code", None)
        print(f"model check failed: {type(error).__name__}" + (f" (HTTP {status})" if status else ""))
        return 1
    print(f"model available: {model.id}")
    if args.generate:
        answer = explain("is my peanut butter recalled?", "bm25", [SAMPLE_HIT],
                         llm_config=config).explanation
        stats = answer.generation
        print(f"generator: {answer.generator}  outcome: {stats.outcome if stats else '-'}")
        if stats:
            print(f"latency: {stats.latency_ms} ms  tokens: prompt={stats.prompt_tokens} "
                  f"completion={stats.completion_tokens} total={stats.total_tokens}")
        if answer.fallback_reason:
            print(f"fallback: {answer.fallback_reason}")
        print("---\n" + answer.text)
        return 0 if answer.generator.startswith("llm:") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
