# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
make dev              # Run with auto-reload on port 8000
make install          # Install all deps + spaCy model (use this first)
make setup-dev        # Install deps + pre-commit hooks

# Testing
make test             # Run all tests (80% coverage required)
make test-cov         # Run tests with HTML/XML/terminal coverage report
make test-unit        # Run unit tests only (-m unit)
make test-integration # Run integration tests only (-m integration)
make test-verbose     # Verbose output

# To run a single test file or function:
pytest tests/test_code.py
pytest tests/test_code.py::test_function_name -v

# Code quality
make lint             # flake8 + mypy
make format           # black + isort
make check            # lint + test (pre-commit)

# Docker
make docker-build && make docker-run
```

## Architecture

Single-file FastAPI app ([main.py](main.py)) — no separate router or service modules. All logic lives in one file (~930 lines).

**Request flow:**
1. `POST /anonymize` receives `AnonymizeRequest` (text, language, optional `AnonymizationConfig`)
2. Request logging middleware captures timing
3. Validation: text length (default max 10,000 chars), language against `Config.ENABLED_LANGUAGES` (`SUPPORTED_LANGUAGES` ∩ languages of `SPACY_MODELS`)
4. `AnalyzerEngine.analyze()` — Presidio detection with one spaCy model per language (optional `score_threshold`)
5. Filter to requested entity types (if specified); otherwise keep every detected entity
6. `remove_overlapping_results()` — one result per text region; the same list feeds both the anonymizer and `detected_entities`
7. Build per-entity operator config based on `AnonymizationStrategy` (REPLACE, REDACT, HASH, MASK, ENCRYPT)
8. `AnonymizerEngine.anonymize()` — apply operators to text
9. Return `AnonymizeResponse` with anonymized text, `DetectedEntity` list, and timing metrics

**Key classes in main.py:**
- `Config` — reads env vars (`DEFAULT_LANGUAGE`, `LOG_LEVEL`, `MAX_TEXT_LENGTH`, `SUPPORTED_LANGUAGES`, `SPACY_MODELS`, `DETECT_ORGANIZATIONS`, `CORS_ORIGINS`); `parse_spacy_models()` / `resolve_enabled_languages()` compute `SPACY_MODELS` and `ENABLED_LANGUAGES`
- `AnonymizationStrategy` enum — maps to Presidio operator names
- `EntityType` enum — 19 PII types (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, ..., IT_FISCAL_CODE, IT_VAT_CODE, IT_IDENTITY_CARD, IT_DRIVER_LICENSE, IT_PASSPORT, IT_POSTAL_CODE)
- `AnonymizeRequest` / `AnonymizeResponse` / `AnonymizationConfig` / `DetectedEntity` — Pydantic v2 models
- `create_analyzer_engine()` — explicit NLP config (`NER_MODEL_CONFIGURATION`, copied from Presidio's `default.yaml`; no private Presidio APIs), `RecognizerRegistry` with the predefined recognizers of the configured languages (activates Presidio's `It*Recognizer`s), `PhoneRecognizer` with the `IT` region added, and for `it` the custom recognizers from `create_italian_recognizers()`: street address (`LOCATION`, score 0.9), CAP (`IT_POSTAL_CODE`, only after a "CAP" label or before a city name), fiscal code after a label
- `remove_overlapping_results()` — pattern recognizers beat NER (`NER_RECOGNIZER_NAMES`) regardless of score; then higher score, longer span, closest `TIE_BREAK_CONTEXT_WORDS` word, `TIE_BREAK_PREFERENCE` (IT_PASSPORT before IT_IDENTITY_CARD)
- Lifespan context manager builds the analyzer for `ENABLED_LANGUAGES` only and fails fast if no language, or not `DEFAULT_LANGUAGE`, is enabled

**Other endpoints:** `GET /health`, `GET /metrics` (psutil), `GET /info` (effective languages, `supported_entities_by_language`), `GET /test/error/{error_type}` (dev only)

## Environment

Copy `env.example` to `.env`. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `DEFAULT_LANGUAGE` | `en` | |
| `MAX_TEXT_LENGTH` | `10000` | |
| `SUPPORTED_LANGUAGES` | `en,es,fr,de,it` | Only those with a model in `SPACY_MODELS` are enabled |
| `SPACY_MODELS` | `en:en_core_web_lg` | `lang:model` pairs; install extra models in Docker with build arg `SPACY_EXTRA_MODELS` |
| `DETECT_ORGANIZATIONS` | `false` | `true` stops ignoring spaCy `ORG` entities |
| `CORS_ORIGINS` | `*` | |
| `LOG_LEVEL` | `INFO` | |

## Tests

- `tests/conftest.py` — shared fixtures (test client, sample text, mocks)
- `tests/test_code.py` — unit tests for anonymization logic
- `tests/test_integration.py` — end-to-end API tests
- `tests/test_config.py` — config/validation tests
- `tests/test_performance.py` — load/perf benchmarks (`-m performance`)
- `tests/test_simple.py` — smoke tests (no heavy presidio imports)
- `tests/test_italian.py` — Italian support; `-m italian_model` tests load `it_core_news_lg` (skipped if missing)

Set `ENVIRONMENT=testing` when running tests locally (CI does): the `/test/error/*` endpoints used by `test_simple.py` only exist then.

Coverage minimum is meant to be 80%, but `pytest.ini` uses a `[tool:pytest]` header, which pytest ignores in a `pytest.ini` file: its `addopts` (coverage, `--strict-markers`) and markers are not applied. That is why the `italian_model` marker is also registered in `tests/conftest.py`.

## CI

GitHub Actions runs on push/PR: multi-Python (3.9–3.12), multi-platform (Ubuntu, macOS, Windows). See [.github/workflows/ci.yml](.github/workflows/ci.yml).
