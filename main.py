import asyncio
import copy
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from enum import Enum

import psutil

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import PhoneRecognizer
from presidio_anonymizer import AnonymizerEngine

try:
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    # Fallback for newer versions of presidio-anonymizer
    try:
        from presidio_anonymizer import OperatorConfig
    except ImportError:

        class _OperatorConfigFallback:
            def __init__(self, operator_name, params=None):
                self.operator_name = operator_name
                self.params = params or {}

        OperatorConfig = _OperatorConfigFallback


# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_spacy_models(value: str) -> Dict[str, str]:
    """Parse SPACY_MODELS ("lang:model,lang:model") into {lang: model}."""
    models: Dict[str, str] = {}
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        lang, separator, model = entry.partition(":")
        lang, model = lang.strip(), model.strip()
        if not separator or not lang or not model:
            raise ValueError(
                f"Invalid SPACY_MODELS entry '{entry}': expected comma-separated "
                "'language:model' pairs, e.g. 'en:en_core_web_lg,it:it_core_news_lg'"
            )
        models[lang] = model
    if not models:
        raise ValueError(
            "SPACY_MODELS must define at least one 'language:model' pair, "
            "e.g. 'en:en_core_web_lg'"
        )
    return models


def resolve_enabled_languages(
    supported_languages: List[str], spacy_models: Dict[str, str]
) -> List[str]:
    """Keep only the supported languages that have a spaCy model configured."""
    missing = [lang for lang in supported_languages if lang not in spacy_models]
    if missing:
        logger.warning(
            f"Languages without a spaCy model in SPACY_MODELS are disabled: "
            f"{', '.join(missing)}"
        )
    return [lang for lang in supported_languages if lang in spacy_models]


# Configuration
class Config:
    """Application configuration"""

    DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "en,es,fr,de,it").split(",")
    # spaCy model per language, as "lang:model" pairs
    SPACY_MODELS = parse_spacy_models(os.getenv("SPACY_MODELS", "en:en_core_web_lg"))
    # Languages actually accepted by the API: SUPPORTED_LANGUAGES ∩ SPACY_MODELS
    ENABLED_LANGUAGES = resolve_enabled_languages(SUPPORTED_LANGUAGES, SPACY_MODELS)
    # Presidio ignores spaCy ORG entities by default (many false positives)
    DETECT_ORGANIZATIONS = os.getenv("DETECT_ORGANIZATIONS", "false").lower() == "true"


# Enums
class AnonymizationStrategy(str, Enum):
    """Available anonymization strategies"""

    REPLACE = "replace"
    REDACT = "redact"
    HASH = "hash"
    MASK = "mask"
    ENCRYPT = "encrypt"

    def __str__(self) -> str:
        return self.value


class EntityType(str, Enum):
    """Supported PII entity types"""

    def __str__(self) -> str:
        return self.value

    PERSON = "PERSON"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"
    URL = "URL"
    US_SSN = "US_SSN"
    US_PASSPORT = "US_PASSPORT"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    IT_FISCAL_CODE = "IT_FISCAL_CODE"
    IT_VAT_CODE = "IT_VAT_CODE"
    IT_IDENTITY_CARD = "IT_IDENTITY_CARD"
    IT_DRIVER_LICENSE = "IT_DRIVER_LICENSE"
    IT_PASSPORT = "IT_PASSPORT"
    IT_POSTAL_CODE = "IT_POSTAL_CODE"


# Pydantic Models
class AnonymizationConfig(BaseModel):
    """Configuration for anonymization process"""

    strategy: AnonymizationStrategy = Field(
        default=AnonymizationStrategy.REPLACE,
        description="Anonymization strategy to use",
    )
    entities_to_anonymize: Optional[List[EntityType]] = Field(
        default=None,
        description="Specific entities to anonymize. If None, all detected entities will be anonymized",
    )
    replacement_text: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Custom replacement text for REPLACE strategy",
    )
    mask_char: str = Field(
        default="*",
        min_length=1,
        max_length=1,
        description="Character to use for MASK strategy",
    )
    chars_to_mask: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Characters to mask per entity for MASK strategy. "
            "Masks the entire entity when omitted"
        ),
    )
    mask_from_end: bool = Field(
        default=False,
        description=(
            "Mask from the end of each entity instead of the start "
            "(MASK strategy)"
        ),
    )
    hash_type: str = Field(
        default="sha256", description="Hash algorithm for HASH strategy"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Minimum confidence score for a detected entity to be anonymized. "
            "No filtering when omitted"
        ),
    )

    @field_validator("hash_type")
    @classmethod
    def validate_hash_type(cls, v: str) -> str:
        # presidio's hash operator implements sha256 and sha512 only.
        allowed = {"sha256", "sha512"}
        if v not in allowed:
            raise ValueError(
                f"hash_type must be one of: {', '.join(sorted(allowed))}"
            )
        return v


class AnonymizeRequest(BaseModel):
    """Request model for anonymization endpoint"""

    text: str = Field(
        ...,
        min_length=1,
        max_length=Config.MAX_TEXT_LENGTH,
        description="Text to anonymize",
    )
    language: str = Field(
        default=Config.DEFAULT_LANGUAGE, description="Language of the text"
    )
    config: Optional[AnonymizationConfig] = Field(
        default=None, description="Anonymization configuration"
    )

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v):
        if v not in Config.ENABLED_LANGUAGES:
            raise ValueError(
                f"Language '{v}' not supported. Supported languages: {', '.join(Config.ENABLED_LANGUAGES)}"
            )
        return v


class DetectedEntity(BaseModel):
    """Model for detected PII entities"""

    entity_type: str = Field(..., description="Type of detected entity")
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")
    score: float = Field(..., description="Confidence score")


class AnonymizeResponse(BaseModel):
    """Response model for anonymization endpoint"""

    anonymized_text: str = Field(..., description="The anonymized text")
    detected_entities: List[DetectedEntity] = Field(
        ..., description="List of detected PII entities"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    original_length: int = Field(..., description="Length of original text")
    anonymized_length: int = Field(..., description="Length of anonymized text")


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    dependencies: Dict[str, str] = Field(..., description="Status of dependencies")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


# NER configuration for the spaCy engine. Copied from presidio_analyzer's
# conf/default.yaml (presidio 2.2.364) rather than read through
# NlpEngineProvider's private helpers, which may change between releases.
NER_MODEL_CONFIGURATION: Dict[str, Any] = {
    "model_to_presidio_entity_mapping": {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "NORP": "NRP",
        "FAC": "LOCATION",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "LOCATION": "LOCATION",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
    },
    "low_confidence_score_multiplier": 0.4,
    "low_score_entity_names": [],
    "labels_to_ignore": [
        "ORGANIZATION",  # Has many false positives
        "CARDINAL",
        "EVENT",
        "LANGUAGE",
        "LAW",
        "MONEY",
        "ORDINAL",
        "PERCENT",
        "PRODUCT",
        "QUANTITY",
        "WORK_OF_ART",
        "MISC",  # Catch-all label of the it/es/fr/de spaCy models
    ],
}

# Recognizers backed by an NER model. Everything else (regex, checksum,
# phonenumbers) is treated as a pattern recognizer by remove_overlapping_results.
NER_RECOGNIZER_NAMES = frozenset(
    {"SpacyRecognizer", "StanzaRecognizer", "TransformersRecognizer"}
)

# Distinctive context words used to break ties between results with the same
# span and score, e.g. "YA1234567" matches both an Italian passport and an
# identity card number. The closest word preceding the span wins.
TIE_BREAK_CONTEXT_WORDS: Dict[str, List[str]] = {
    "IT_PASSPORT": ["passaporto"],
    "IT_IDENTITY_CARD": ["carta d'identità", "carta di identità", "identità", "cie"],
    "IT_DRIVER_LICENSE": ["patente"],
}
TIE_BREAK_CONTEXT_WINDOW = 50
# Fixed preference when no context word decides (earlier entries win)
TIE_BREAK_PREFERENCE = ["IT_PASSPORT", "IT_IDENTITY_CARD", "IT_DRIVER_LICENSE"]

PHONE_SUPPORTED_REGIONS = PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS + ("IT",)
PHONE_CONTEXT_IT = [
    "telefono",
    "tel",
    "cellulare",
    "cell",
    "numero",
    "fax",
    "chiamare",
    "recapito",
]

# Italian street address: toponym prefix (case-insensitive), capitalised street
# name and optional house number ("12", "12/A", "12 bis", "n. 12", "snc").
_IT_ADDRESS_PREFIX = (
    r"\b(?:via|viale|v\.le|piazzale|piazza|p\.zza|p\.za|corso|c\.so|largo"
    r"|vicolo|strada|contrada|località|loc\.|lungomare|borgo)"
)
_IT_NAME_WORD = r"[A-ZÀ-ÖØ-Þ](?:[\w'’-]*\w)?"
_IT_NAME_CONNECTOR = (
    r"(?:(?:di|de|da|del|dei|degli|della|dello|delle|dal|dalla|dai|in|e|al"
    r"|alla|ai|sul|sulla)[ \t]+|(?:dell|dall|all|sull|nell|d)['’])"
)
_IT_STREET_NAME = (
    rf"(?-i:(?:\d{{1,2}}[ \t]+)?{_IT_NAME_CONNECTOR}?{_IT_NAME_WORD}"
    rf"(?:[ \t]+{_IT_NAME_CONNECTOR}?{_IT_NAME_WORD}){{0,5}})"
)
_IT_HOUSE_NUMBER = (
    r"(?:,?[ \t]*(?:(?:n|nr|num)\.?|n°)?[ \t]*"
    r"(?:\d{1,4}(?!\d)(?:[ \t]*/[ \t]*[A-Z0-9]{1,3}\b|[ \t]+(?:bis|ter|quater)\b"
    r"|[A-Z]\b)?|snc\b))?"
)
# "via" also means "by means of": skip "via PEC", "via WhatsApp", ...
_IT_NOT_STREET_NAME = (
    r"(?!(?:pec|e-?mail|mail|posta|fax|sms|mms|web|internet|telefono|cellulare"
    r"|app|chat|whatsapp|telegram|skype|teams|zoom|meet|messenger|facebook"
    r"|linkedin|instagram|raccomandata|corriere|bonifico|paypal|ftp|api)\b)"
)
IT_ADDRESS_REGEX = rf"{_IT_ADDRESS_PREFIX}[ \t]+{_IT_NOT_STREET_NAME}{_IT_STREET_NAME}{_IT_HOUSE_NUMBER}"

# Italian postal code (CAP). A bare 5-digit number is too ambiguous, so it is
# only reported after a "CAP" label or right before a capitalised city name,
# and never when it is part of a phone number, VAT code or IBAN.
IT_POSTAL_CODE_LABEL_REGEX = r"(?<=\bc\.?a\.?p\.?[ \t]*[:.]?[ \t]*)\d{5}(?!\d)"
IT_POSTAL_CODE_CITY_REGEX = (
    r"(?<![\w.+/-])(?<!\d[ \t.-])\d{5}"
    r"(?=[ \t]*[-–]?[ \t]*(?-i:[A-ZÀ-ÖØ-Þ](?:[a-zß-öø-ÿ']|[A-ZÀ-ÖØ-Þ']{2})))"
)

# Fiscal code shape (including omocodia letters) after an explicit label, so
# that codes with a wrong check character are still reported confidently.
IT_FISCAL_CODE_LABEL_REGEX = (
    r"(?<=\b(?:codice[ \t]+fiscale|cod\.?[ \t]*fisc\.?|c\.[ \t]?f\.|cf)[ \t]*[:.]?[ \t]*)"
    r"[A-Z]{6}[\dLMNP-V]{2}[A-EHLMPR-T][\dLMNP-V]{2}[A-Z][\dLMNP-V]{3}[A-Z]\b"
)


def create_italian_recognizers() -> List[PatternRecognizer]:
    """Italian recognizers that complement Presidio's predefined It* ones."""
    return [
        PatternRecognizer(
            supported_entity="LOCATION",
            name="ItAddressRecognizer",
            supported_language="it",
            # Higher than the spaCy NER score (0.85) so the full address wins
            patterns=[Pattern("Italian street address", IT_ADDRESS_REGEX, 0.9)],
        ),
        PatternRecognizer(
            supported_entity="IT_POSTAL_CODE",
            name="ItPostalCodeRecognizer",
            supported_language="it",
            patterns=[
                Pattern("CAP after label", IT_POSTAL_CODE_LABEL_REGEX, 0.6),
                Pattern("CAP before city", IT_POSTAL_CODE_CITY_REGEX, 0.5),
            ],
            context=["cap", "c.a.p."],
        ),
        PatternRecognizer(
            supported_entity="IT_FISCAL_CODE",
            name="ItFiscalCodeLabelRecognizer",
            supported_language="it",
            patterns=[
                Pattern("Fiscal code after label", IT_FISCAL_CODE_LABEL_REGEX, 0.8)
            ],
        ),
    ]


def create_phone_recognizer(language: str) -> PhoneRecognizer:
    """Phone recognizer that also validates Italian numbers."""
    return PhoneRecognizer(
        supported_language=language,
        supported_regions=PHONE_SUPPORTED_REGIONS,
        context=PHONE_CONTEXT_IT if language == "it" else None,
    )


def build_nlp_configuration(
    spacy_models: Dict[str, str], detect_organizations: bool = False
) -> Dict[str, Any]:
    """Build the Presidio NLP engine configuration for the given models."""
    ner_configuration = copy.deepcopy(NER_MODEL_CONFIGURATION)
    if detect_organizations:
        ner_configuration["labels_to_ignore"].remove("ORGANIZATION")
    return {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": lang, "model_name": model}
            for lang, model in spacy_models.items()
        ],
        "ner_model_configuration": ner_configuration,
    }


def create_analyzer_engine(
    spacy_models: Dict[str, str], detect_organizations: bool = False
) -> AnalyzerEngine:
    """Create an analyzer with one spaCy model and the predefined recognizers per language."""
    languages = list(spacy_models)
    nlp_engine = NlpEngineProvider(
        nlp_configuration=build_nlp_configuration(spacy_models, detect_organizations)
    ).create_engine()

    registry = RecognizerRegistry(supported_languages=languages)
    registry.load_predefined_recognizers(languages=languages, nlp_engine=nlp_engine)

    registry.remove_recognizer("PhoneRecognizer")
    for language in languages:
        registry.add_recognizer(create_phone_recognizer(language))

    if "it" in languages:
        for recognizer in create_italian_recognizers():
            registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        nlp_engine=nlp_engine, registry=registry, supported_languages=languages
    )


def _is_ner_result(result: RecognizerResult) -> bool:
    metadata = result.recognition_metadata or {}
    return metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY) in NER_RECOGNIZER_NAMES


def _context_distance(result: RecognizerResult, text: Optional[str]) -> int:
    """Characters between the result and its closest preceding tie-break context word."""
    words = TIE_BREAK_CONTEXT_WORDS.get(result.entity_type)
    if not text or not words:
        return TIE_BREAK_CONTEXT_WINDOW + 1
    window = text[max(0, result.start - TIE_BREAK_CONTEXT_WINDOW) : result.start]
    distance = TIE_BREAK_CONTEXT_WINDOW + 1
    for word in words:
        for match in re.finditer(rf"\b{re.escape(word)}\b", window, re.IGNORECASE):
            distance = min(distance, len(window) - match.end())
    return distance


def _result_priority(result: RecognizerResult, text: Optional[str]) -> tuple:
    """Sort key for overlap resolution: lower sorts first and wins."""
    preference = (
        TIE_BREAK_PREFERENCE.index(result.entity_type)
        if result.entity_type in TIE_BREAK_PREFERENCE
        else len(TIE_BREAK_PREFERENCE)
    )
    return (
        _is_ner_result(result),  # pattern recognizers win over NER
        -result.score,
        -(result.end - result.start),
        _context_distance(result, text),
        preference,
        result.start,
        result.entity_type,
    )


def remove_overlapping_results(
    results: List[RecognizerResult], text: Optional[str] = None
) -> List[RecognizerResult]:
    """
    Resolve overlapping analyzer results, keeping one result per text region.

    A pattern-based result beats an overlapping NER result regardless of score
    (e.g. an IT_FISCAL_CODE with a wrong checksum over a spaCy LOCATION).
    Otherwise the higher score wins, then the longer span, then the closest
    tie-break context word, then TIE_BREAK_PREFERENCE. Scores are not changed.
    The returned results are sorted by start position.
    """
    kept: List[RecognizerResult] = []
    for candidate in sorted(results, key=lambda r: _result_priority(r, text)):
        if all(candidate.end <= r.start or candidate.start >= r.end for r in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda r: (r.start, r.end))


# Global variables for engines
analyzer_engine: Optional[AnalyzerEngine] = None
anonymizer_engine: Optional[AnonymizerEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analyzer_engine, anonymizer_engine

    logger.info("Starting PII Anonymizer API...")

    try:
        # Initialize engines
        logger.info("Initializing Presidio engines...")
        if not Config.ENABLED_LANGUAGES:
            raise RuntimeError(
                "No language enabled: SUPPORTED_LANGUAGES and SPACY_MODELS "
                "have no language in common"
            )
        if Config.DEFAULT_LANGUAGE not in Config.ENABLED_LANGUAGES:
            raise RuntimeError(
                f"DEFAULT_LANGUAGE '{Config.DEFAULT_LANGUAGE}' is not enabled. "
                f"Enabled languages: {', '.join(Config.ENABLED_LANGUAGES)}"
            )
        spacy_models = {
            lang: Config.SPACY_MODELS[lang] for lang in Config.ENABLED_LANGUAGES
        }
        logger.info(f"Loading spaCy models: {spacy_models}")
        analyzer_engine = create_analyzer_engine(
            spacy_models, detect_organizations=Config.DETECT_ORGANIZATIONS
        )
        anonymizer_engine = AnonymizerEngine()
        logger.info("Presidio engines initialized successfully")
        app.state.start_time = time.time()

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        logger.info("Shutting down PII Anonymizer API...")


# FastAPI app initialization
app = FastAPI(
    title="PII Anonymizer API",
    description="A FastAPI service for anonymizing Personally Identifiable Information (PII) in text data using Microsoft Presidio",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(error="ValidationError", message=str(exc)).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"RuntimeError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="RuntimeError", message=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError", message="An unexpected error occurred"
        ).model_dump(),
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint that returns the status of the service and its dependencies.
    """
    try:
        # Check if engines are initialized
        analyzer_status = "healthy" if analyzer_engine is not None else "unhealthy"
        anonymizer_status = "healthy" if anonymizer_engine is not None else "unhealthy"

        return HealthResponse(
            status=(
                "healthy"
                if analyzer_status == "healthy" and anonymizer_status == "healthy"
                else "unhealthy"
            ),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            version="2.0.0",
            dependencies={
                "presidio_analyzer": analyzer_status,
                "presidio_anonymizer": anonymizer_status,
            },
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy"
        )


@app.post("/anonymize", response_model=AnonymizeResponse, tags=["Anonymization"])
async def anonymize_text(request: AnonymizeRequest) -> AnonymizeResponse:
    """
    Anonymize PII in the provided text data.

    This endpoint analyzes the input text for personally identifiable information (PII)
    and anonymizes it according to the specified configuration.
    """
    start_time = time.time()

    try:
        if not analyzer_engine or not anonymizer_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Anonymization engines not initialized",
            )

        logger.info(
            f"Processing anonymization request for text of length {len(request.text)}"
        )

        # Analyze text for PII entities (CPU-bound — offload to thread pool)
        loop = asyncio.get_running_loop()
        score_threshold = request.config.score_threshold if request.config else None
        analyzer_results = await loop.run_in_executor(
            None,
            lambda: analyzer_engine.analyze(
                text=request.text,
                language=request.language,
                score_threshold=score_threshold,
            ),
        )

        # Filter entities if specific types are requested
        if request.config and request.config.entities_to_anonymize is not None:
            allowed = {e.value for e in request.config.entities_to_anonymize}
            analyzer_results = [
                result
                for result in analyzer_results
                if result.entity_type in allowed
            ]

        # Resolve overlaps once, so the anonymized text and detected_entities
        # are built from the same results
        analyzer_results = remove_overlapping_results(analyzer_results, request.text)

        # Configure anonymization operators
        operators = {}
        if request.config:
            if request.config.strategy == AnonymizationStrategy.REPLACE:
                operators = {
                    "DEFAULT": OperatorConfig(
                        "replace",
                        {
                            "new_value": request.config.replacement_text
                            or "<ANONYMIZED>"
                        },
                    )
                }
            elif request.config.strategy == AnonymizationStrategy.REDACT:
                operators = {"DEFAULT": OperatorConfig("redact")}
            elif request.config.strategy == AnonymizationStrategy.MASK:
                # presidio's mask operator requires all three parameters and
                # clamps chars_to_mask to the entity length, so defaulting it
                # to the text length masks every entity in full.
                chars_to_mask = request.config.chars_to_mask
                if chars_to_mask is None:
                    chars_to_mask = len(request.text)
                operators = {
                    "DEFAULT": OperatorConfig(
                        "mask",
                        {
                            "masking_char": request.config.mask_char,
                            "chars_to_mask": chars_to_mask,
                            "from_end": request.config.mask_from_end,
                        },
                    )
                }
            elif request.config.strategy == AnonymizationStrategy.HASH:
                # presidio-anonymizer >= 2.2.361 salts each entity with a
                # random 32-byte salt, so hashes differ between requests. Set
                # ANONYMIZER_HASH_SALT to keep them stable across requests.
                hash_params: Dict[str, Any] = {
                    "hash_type": request.config.hash_type
                }
                hash_salt = os.getenv("ANONYMIZER_HASH_SALT")
                if hash_salt:
                    if len(hash_salt.encode()) < 16:
                        raise ValueError(
                            "ANONYMIZER_HASH_SALT must be at least 16 bytes"
                        )
                    hash_params["salt"] = hash_salt
                operators = {"DEFAULT": OperatorConfig("hash", hash_params)}
            elif request.config.strategy == AnonymizationStrategy.ENCRYPT:
                encrypt_key = os.getenv("ANONYMIZER_ENCRYPT_KEY")
                if not encrypt_key:
                    raise ValueError(
                        "ENCRYPT strategy requires ANONYMIZER_ENCRYPT_KEY environment variable to be set"
                    )
                operators = {
                    "DEFAULT": OperatorConfig("encrypt", {"key": encrypt_key})
                }

        # Anonymize text (CPU-bound — offload to thread pool)
        _ops = operators if operators else None
        anonymized_result = await loop.run_in_executor(
            None,
            lambda: anonymizer_engine.anonymize(
                text=request.text,
                analyzer_results=analyzer_results,  # type: ignore[arg-type]
                operators=_ops,
            ),
        )

        # Prepare detected entities for response
        detected_entities = [
            DetectedEntity(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
            )
            for result in analyzer_results
        ]

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Anonymization completed in {processing_time:.2f}ms, detected {len(detected_entities)} entities"
        )

        return AnonymizeResponse(
            anonymized_text=anonymized_result.text,
            detected_entities=detected_entities,
            processing_time_ms=processing_time,
            original_length=len(request.text),
            anonymized_length=len(anonymized_result.text),
        )

    except HTTPException:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error during anonymization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anonymization failed: {str(e)}",
        )


# Additional endpoints for monitoring and metrics
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get application metrics for monitoring.
    """
    try:
        process = psutil.Process(os.getpid())

        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            },
            "process": {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            },
            "application": {
                "analyzer_status": (
                    "healthy" if analyzer_engine is not None else "unhealthy"
                ),
                "anonymizer_status": (
                    "healthy" if anonymizer_engine is not None else "unhealthy"
                ),
                "uptime_seconds": (
                    time.time() - app.state.start_time
                    if hasattr(app.state, "start_time")
                    else 0
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics",
        )


@app.get("/info", tags=["Information"])
async def get_info():
    """
    Get application information and configuration.
    """
    supported_entities_by_language = {}
    if analyzer_engine is not None:
        supported_entities_by_language = {
            lang: sorted(analyzer_engine.get_supported_entities(language=lang))
            for lang in Config.ENABLED_LANGUAGES
        }

    return {
        "name": "PII Anonymizer API",
        "version": "2.0.0",
        "description": "A FastAPI service for anonymizing Personally Identifiable Information (PII) in text data",
        "configuration": {
            "max_text_length": Config.MAX_TEXT_LENGTH,
            "supported_languages": Config.ENABLED_LANGUAGES,
            "default_language": Config.DEFAULT_LANGUAGE,
            "spacy_models": Config.SPACY_MODELS,
            "detect_organizations": Config.DETECT_ORGANIZATIONS,
        },
        "supported_entities": [entity.value for entity in EntityType],
        "supported_entities_by_language": supported_entities_by_language,
        "supported_strategies": [strategy.value for strategy in AnonymizationStrategy],
        "endpoints": {
            "health": "/health",
            "anonymize": "/anonymize",
            "metrics": "/metrics",
            "info": "/info",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


# Test endpoints (only available in development/testing)
if os.getenv("ENVIRONMENT", "production") in ["development", "testing"]:

    @app.get("/test/error/{error_type}", tags=["Testing"], include_in_schema=False)
    async def test_error_handler(error_type: str):
        """Test endpoint for error handling (development/testing only)."""
        try:
            if error_type == "value":
                raise ValueError("Test ValueError")
            elif error_type == "runtime":
                raise RuntimeError("Test RuntimeError")
            elif error_type == "http":
                raise HTTPException(status_code=418, detail="I'm a teapot")
            else:
                return {"message": "No error raised"}
        except HTTPException:
            # Re-raise HTTPExceptions so they're handled properly
            raise
        except Exception as e:
            # For testing purposes, we'll manually call our exception handlers
            if isinstance(e, ValueError):
                return JSONResponse(
                    status_code=400,
                    content=ErrorResponse(
                        error="ValidationError", message=str(e)
                    ).model_dump(),
                )
            elif isinstance(e, RuntimeError):
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        error="RuntimeError", message=str(e)
                    ).model_dump(),
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        error="InternalServerError",
                        message="An unexpected error occurred",
                    ).model_dump(),
                )
