"""
Tests for Italian language support: overlap resolution, SPACY_MODELS parsing,
Italian recognizers and the related API behaviour.

Unit tests run without spaCy models. Tests marked ``italian_model`` load the
real ``it_core_news_lg`` model and are skipped when it is not installed.
"""

import importlib
import logging
import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

import main
from main import (
    NER_MODEL_CONFIGURATION,
    PHONE_CONTEXT_IT,
    AnonymizationConfig,
    EntityType,
    PhoneRecognizer,
    app,
    build_nlp_configuration,
    create_analyzer_engine,
    create_italian_recognizers,
    create_phone_recognizer,
    parse_spacy_models,
    remove_overlapping_results,
    resolve_enabled_languages,
)

client = TestClient(app)

ITALIAN_ENTITIES = [
    "IT_FISCAL_CODE",
    "IT_VAT_CODE",
    "IT_IDENTITY_CARD",
    "IT_DRIVER_LICENSE",
    "IT_PASSPORT",
    "IT_POSTAL_CODE",
]


def make_result(entity_type, start, end, score, recognizer="PatternRecognizer"):
    return RecognizerResult(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
        recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: recognizer},
    )


def spans(results):
    return [(r.entity_type, r.start, r.end) for r in results]


def recognizer_by_name(name):
    return next(r for r in create_italian_recognizers() if r.name == name)


def matched_texts(recognizer, text):
    results = recognizer.analyze(text, recognizer.supported_entities, None)
    return [text[r.start : r.end] for r in results]


@pytest.fixture
def reload_main():
    """Reload main with the patched environment, then restore the defaults."""
    yield lambda: importlib.reload(main)
    importlib.reload(main)


@pytest.mark.unit
class TestRemoveOverlappingResults:
    def test_pattern_result_beats_ner_with_higher_score(self):
        results = [
            make_result("LOCATION", 15, 31, 0.85, "SpacyRecognizer"),
            make_result("IT_FISCAL_CODE", 15, 31, 0.3, "ItFiscalCodeRecognizer"),
        ]

        resolved = remove_overlapping_results(results)

        assert spans(resolved) == [("IT_FISCAL_CODE", 15, 31)]
        assert resolved[0].score == 0.3

    def test_pattern_result_inside_longer_ner_span(self):
        results = [
            make_result("LOCATION", 0, 17, 0.85, "SpacyRecognizer"),
            make_result("IT_VAT_CODE", 6, 17, 1.0, "ItVatCodeRecognizer"),
            make_result("PHONE_NUMBER", 6, 17, 0.4, "PhoneRecognizer"),
        ]

        assert spans(remove_overlapping_results(results)) == [("IT_VAT_CODE", 6, 17)]

    def test_higher_score_wins_between_pattern_results(self):
        results = [
            make_result("URL", 0, 8, 0.5, "UrlRecognizer"),
            make_result("EMAIL_ADDRESS", 0, 22, 1.0, "EmailRecognizer"),
            make_result("URL", 12, 22, 0.5, "UrlRecognizer"),
        ]

        assert spans(remove_overlapping_results(results)) == [("EMAIL_ADDRESS", 0, 22)]

    def test_higher_score_wins_between_ner_results(self):
        results = [
            make_result("LOCATION", 0, 10, 0.6, "SpacyRecognizer"),
            make_result("PERSON", 5, 15, 0.85, "SpacyRecognizer"),
        ]

        assert spans(remove_overlapping_results(results)) == [("PERSON", 5, 15)]

    def test_longer_span_wins_on_equal_score(self):
        results = [
            make_result("LOCATION", 13, 26, 0.9),
            make_result("LOCATION", 13, 29, 0.9),
        ]

        assert spans(remove_overlapping_results(results)) == [("LOCATION", 13, 29)]

    def test_non_overlapping_results_are_kept_and_sorted(self):
        results = [
            make_result("EMAIL_ADDRESS", 40, 60, 1.0),
            make_result("PERSON", 0, 10, 0.85, "SpacyRecognizer"),
            make_result("PHONE_NUMBER", 10, 20, 0.4),
        ]

        assert spans(remove_overlapping_results(results)) == [
            ("PERSON", 0, 10),
            ("PHONE_NUMBER", 10, 20),
            ("EMAIL_ADDRESS", 40, 60),
        ]

    def test_empty_results(self):
        assert remove_overlapping_results([]) == []

    def test_results_without_metadata_are_pattern_results(self):
        results = [
            make_result("LOCATION", 0, 10, 0.85, "SpacyRecognizer"),
            RecognizerResult("IT_PASSPORT", 0, 10, 0.1),
        ]

        assert spans(remove_overlapping_results(results)) == [("IT_PASSPORT", 0, 10)]

    def test_tie_break_uses_closest_context_word(self):
        text = "Carta identità CA12345AB, patente MI1234567A, passaporto YA1234567"
        start = text.index("YA1234567")
        results = [
            make_result("IT_IDENTITY_CARD", start, start + 9, 0.4),
            make_result("IT_PASSPORT", start, start + 9, 0.4),
        ]

        resolved = remove_overlapping_results(results, text)

        assert spans(resolved) == [("IT_PASSPORT", start, start + 9)]

    def test_tie_break_prefers_identity_card_with_its_context(self):
        text = "Numero carta di identità YA1234567"
        start = text.index("YA1234567")
        results = [
            make_result("IT_PASSPORT", start, start + 9, 0.4),
            make_result("IT_IDENTITY_CARD", start, start + 9, 0.4),
        ]

        resolved = remove_overlapping_results(results, text)

        assert spans(resolved) == [("IT_IDENTITY_CARD", start, start + 9)]

    def test_tie_break_falls_back_to_fixed_preference(self):
        results = [
            make_result("IT_IDENTITY_CARD", 0, 9, 0.4),
            make_result("IT_PASSPORT", 0, 9, 0.4),
        ]

        assert spans(remove_overlapping_results(results, "YA1234567")) == [
            ("IT_PASSPORT", 0, 9)
        ]


@pytest.mark.unit
class TestSpacyModelsConfiguration:
    def test_parse_valid_models(self):
        assert parse_spacy_models("en:en_core_web_lg,it:it_core_news_lg") == {
            "en": "en_core_web_lg",
            "it": "it_core_news_lg",
        }

    def test_parse_ignores_whitespace_and_empty_entries(self):
        assert parse_spacy_models(" en : en_core_web_lg , ,it:it_core_news_lg,") == {
            "en": "en_core_web_lg",
            "it": "it_core_news_lg",
        }

    @pytest.mark.parametrize("value", ["en_core_web_lg", "en:", ":en_core_web_lg"])
    def test_parse_malformed_entry(self, value):
        with pytest.raises(
            ValueError, match="expected comma-separated 'language:model'"
        ):
            parse_spacy_models(value)

    def test_parse_empty_value(self):
        with pytest.raises(ValueError, match="at least one"):
            parse_spacy_models(" , ")

    def test_enabled_languages_intersection(self, caplog):
        models = {"en": "en_core_web_lg", "it": "it_core_news_lg"}

        with caplog.at_level(logging.WARNING, logger="main"):
            enabled = resolve_enabled_languages(["en", "es", "it", "fr"], models)

        assert enabled == ["en", "it"]
        assert "es, fr" in caplog.text

    def test_enabled_languages_without_missing_models(self, caplog):
        with caplog.at_level(logging.WARNING, logger="main"):
            enabled = resolve_enabled_languages(["it"], {"it": "it_core_news_lg"})

        assert enabled == ["it"]
        assert caplog.text == ""

    def test_config_from_environment(self, reload_main):
        env = {
            "SUPPORTED_LANGUAGES": "en,it,fr",
            "SPACY_MODELS": "en:en_core_web_lg,it:it_core_news_lg",
            "DETECT_ORGANIZATIONS": "true",
        }
        with patch.dict(os.environ, env):
            reload_main()

            assert main.Config.SPACY_MODELS == {
                "en": "en_core_web_lg",
                "it": "it_core_news_lg",
            }
            assert main.Config.ENABLED_LANGUAGES == ["en", "it"]
            assert main.Config.DETECT_ORGANIZATIONS is True

    def test_config_defaults(self):
        assert main.Config.SPACY_MODELS == {"en": "en_core_web_lg"}
        assert main.Config.ENABLED_LANGUAGES == ["en"]
        assert main.Config.DETECT_ORGANIZATIONS is False


@pytest.mark.unit
class TestNlpConfiguration:
    def test_organizations_ignored_by_default(self):
        configuration = build_nlp_configuration({"it": "it_core_news_lg"})

        assert configuration["nlp_engine_name"] == "spacy"
        assert configuration["models"] == [
            {"lang_code": "it", "model_name": "it_core_news_lg"}
        ]
        labels = configuration["ner_model_configuration"]["labels_to_ignore"]
        assert "ORGANIZATION" in labels
        assert "MISC" in labels

    def test_detect_organizations(self):
        configuration = build_nlp_configuration(
            {"it": "it_core_news_lg"}, detect_organizations=True
        )

        labels = configuration["ner_model_configuration"]["labels_to_ignore"]
        assert "ORGANIZATION" not in labels
        assert "ORGANIZATION" in NER_MODEL_CONFIGURATION["labels_to_ignore"]

    def test_phone_recognizer_regions(self):
        recognizer = create_phone_recognizer("it")

        assert "IT" in recognizer.supported_regions
        for region in PhoneRecognizer.DEFAULT_SUPPORTED_REGIONS:
            assert region in recognizer.supported_regions
        assert recognizer.context == PHONE_CONTEXT_IT
        assert create_phone_recognizer("en").context == PhoneRecognizer.CONTEXT

    @pytest.mark.parametrize(
        "number", ["347 1234567", "06 68401", "+39 347 123 4567", "0331 123456"]
    )
    def test_phone_recognizer_detects_italian_numbers(self, number):
        text = f"Chiamami al {number}"

        assert matched_texts(create_phone_recognizer("it"), text) == [number]

    @pytest.mark.parametrize(
        "models, has_italian",
        [
            ({"en": "en_core_web_lg", "it": "it_core_news_lg"}, True),
            ({"en": "x"}, False),
        ],
    )
    def test_create_analyzer_engine_registry(self, models, has_italian):
        nlp_engine = Mock()
        nlp_engine.get_supported_languages.return_value = list(models)
        nlp_engine.get_supported_entities.return_value = ["PERSON", "LOCATION"]

        with patch("main.NlpEngineProvider") as provider, patch(
            "main.RecognizerRegistry.get_nlp_recognizer",
            return_value=main.PatternRecognizer,
        ), patch("main.RecognizerRegistry._create_nlp_recognizer"):
            provider.return_value.create_engine.return_value = nlp_engine
            engine = create_analyzer_engine(models, detect_organizations=True)

        configuration = provider.call_args.kwargs["nlp_configuration"]
        assert (
            "ORGANIZATION"
            not in configuration["ner_model_configuration"]["labels_to_ignore"]
        )
        recognizers = engine.registry.recognizers
        phones = [r for r in recognizers if r.name == "PhoneRecognizer"]
        assert sorted(r.supported_language for r in phones) == sorted(models)
        assert all("IT" in r.supported_regions for r in phones)
        names = {r.name for r in recognizers}
        for name in ("ItAddressRecognizer", "ItPostalCodeRecognizer"):
            assert (name in names) is has_italian
        assert engine.supported_languages == list(models)


@pytest.mark.unit
class TestItalianRecognizers:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Residente in Via Garibaldi 12, 20121 Milano", "Via Garibaldi 12"),
            ("abita in viale dei Mille 3/B, Firenze", "viale dei Mille 3/B"),
            ("ufficio in P.zza San Marco n. 1", "P.zza San Marco n. 1"),
            (
                "Corso Vittorio Emanuele II 45 bis, Roma",
                "Corso Vittorio Emanuele II 45 bis",
            ),
            ("in Via XX Settembre, 5", "Via XX Settembre, 5"),
            ("Loc. Pian di Sco snc", "Loc. Pian di Sco snc"),
            ("Largo dell'Indipendenza 7A", "Largo dell'Indipendenza 7A"),
            ("C.so Buenos Aires 12", "C.so Buenos Aires 12"),
            ("VIA GARIBALDI 12 - MILANO", "VIA GARIBALDI 12"),
            ("Piazza Duomo", "Piazza Duomo"),
        ],
    )
    def test_address_positive(self, text, expected):
        recognizer = recognizer_by_name("ItAddressRecognizer")

        assert matched_texts(recognizer, text) == [expected]

    def test_address_score_beats_ner(self):
        recognizer = recognizer_by_name("ItAddressRecognizer")
        results = recognizer.analyze("Via Roma 1", ["LOCATION"], None)

        assert results[0].entity_type == "LOCATION"
        assert results[0].score == 0.9

    @pytest.mark.parametrize(
        "text",
        [
            "Ho inviato il documento via email",
            "Te lo mando via PEC oppure via WhatsApp",
            "per via di un problema",
            "strada facendo abbiamo parlato",
            "Invio via 3 canali",
        ],
    )
    def test_address_negative(self, text):
        recognizer = recognizer_by_name("ItAddressRecognizer")

        assert matched_texts(recognizer, text) == []

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Via Garibaldi 12, 20121 Milano (MI)", ["20121"]),
            ("CAP 50131", ["50131"]),
            ("c.a.p.: 00186", ["00186"]),
            ("00186 ROMA", ["00186"]),
            ("Via Roma 1, 10121 - Torino", ["10121"]),
        ],
    )
    def test_postal_code_positive(self, text, expected):
        recognizer = recognizer_by_name("ItPostalCodeRecognizer")

        assert matched_texts(recognizer, text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "P.IVA 07643520567",
            "IBAN IT60X0542811101000000123456",
            "tel 06 68401",
            "tel. 06 68401 Roma",
            "cell. 347 1234567",
            "Ordine 12345 del 2024",
            "Totale 12345 euro",
            "+39 12345 Mario",
            "Protocollo 2024/12345 Milano",
        ],
    )
    def test_postal_code_negative(self, text):
        recognizer = recognizer_by_name("ItPostalCodeRecognizer")

        assert matched_texts(recognizer, text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "codice fiscale RSSMRA80C12F205X",
            "Cod. Fisc.: RSSMRA80C12F205X",
            "C.F. RSSMRA80C12F205X",
            "CF: rssmra80c12f205x",
        ],
    )
    def test_fiscal_code_after_label(self, text):
        recognizer = recognizer_by_name("ItFiscalCodeLabelRecognizer")

        assert [m.upper() for m in matched_texts(recognizer, text)] == [
            "RSSMRA80C12F205X"
        ]

    def test_fiscal_code_without_label(self):
        recognizer = recognizer_by_name("ItFiscalCodeLabelRecognizer")

        assert matched_texts(recognizer, "Il codice è RSSMRA80C12F205X") == []


@pytest.mark.unit
class TestItalianApi:
    def test_entity_enum_contains_italian_entities(self):
        values = {entity.value for entity in EntityType}

        for entity in ITALIAN_ENTITIES:
            assert entity in values

    def test_config_accepts_italian_entities(self):
        config = AnonymizationConfig(entities_to_anonymize=["IT_FISCAL_CODE"])

        assert config.entities_to_anonymize == [EntityType.IT_FISCAL_CODE]

    def test_entities_to_anonymize_italian_entity(self):
        text = "codice fiscale RSSMRA80C12F205X di Mario Rossi"
        with patch("main.analyzer_engine") as analyzer, patch(
            "main.anonymizer_engine"
        ) as anonymizer, patch("main.Config.ENABLED_LANGUAGES", ["en", "it"]):
            analyzer.analyze.return_value = [
                make_result("LOCATION", 15, 31, 0.85, "SpacyRecognizer"),
                make_result("IT_FISCAL_CODE", 15, 31, 0.3),
                make_result("PERSON", 35, 46, 0.85, "SpacyRecognizer"),
            ]
            anonymizer.anonymize.return_value = Mock(
                text="codice fiscale <IT_FISCAL_CODE> di Mario Rossi"
            )

            response = client.post(
                "/anonymize",
                json={
                    "text": text,
                    "language": "it",
                    "config": {"entities_to_anonymize": ["IT_FISCAL_CODE"]},
                },
            )

        assert response.status_code == 200
        assert [
            (e["entity_type"], e["start"], e["end"], e["score"])
            for e in response.json()["detected_entities"]
        ] == [("IT_FISCAL_CODE", 15, 31, 0.3)]

    def test_detected_entities_match_anonymized_results(self):
        with patch("main.analyzer_engine") as analyzer, patch(
            "main.anonymizer_engine"
        ) as anonymizer:
            analyzer.analyze.return_value = [
                make_result("EMAIL_ADDRESS", 20, 42, 1.0, "EmailRecognizer"),
                make_result("URL", 20, 28, 0.5, "UrlRecognizer"),
                make_result("URL", 32, 42, 0.5, "UrlRecognizer"),
                make_result("PERSON", 0, 8, 0.85, "SpacyRecognizer"),
            ]
            anonymizer.anonymize.return_value = Mock(text="anonymized")

            response = client.post(
                "/anonymize",
                json={"text": "John Doe's email is john.doe@example.com"},
            )

        assert response.status_code == 200
        entities = response.json()["detected_entities"]
        assert [(e["entity_type"], e["start"], e["end"]) for e in entities] == [
            ("PERSON", 0, 8),
            ("EMAIL_ADDRESS", 20, 42),
        ]
        anonymized = anonymizer.anonymize.call_args.kwargs["analyzer_results"]
        assert spans(anonymized) == [("PERSON", 0, 8), ("EMAIL_ADDRESS", 20, 42)]

    def test_score_threshold_is_passed_to_analyzer(self):
        with patch("main.analyzer_engine") as analyzer, patch(
            "main.anonymizer_engine"
        ) as anonymizer:
            analyzer.analyze.return_value = []
            anonymizer.anonymize.return_value = Mock(text="Test text")

            response = client.post(
                "/anonymize",
                json={"text": "Test text", "config": {"score_threshold": 0.5}},
            )

        assert response.status_code == 200
        assert analyzer.analyze.call_args.kwargs["score_threshold"] == 0.5

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_score_threshold_out_of_range(self, threshold):
        response = client.post(
            "/anonymize",
            json={"text": "Test text", "config": {"score_threshold": threshold}},
        )

        assert response.status_code == 422

    def test_language_without_model_is_rejected(self):
        with patch("main.Config.ENABLED_LANGUAGES", ["en"]):
            response = client.post(
                "/anonymize", json={"text": "Ciao Mario", "language": "it"}
            )

        assert response.status_code == 422
        assert "Supported languages: en" in response.text

    def test_info_lists_italian_entities_and_effective_languages(self):
        analyzer = Mock()
        analyzer.get_supported_entities.side_effect = lambda language: (
            ["PERSON", "IT_FISCAL_CODE"] if language == "it" else ["PERSON"]
        )
        with patch("main.analyzer_engine", analyzer), patch(
            "main.Config.ENABLED_LANGUAGES", ["en", "it"]
        ):
            data = client.get("/info").json()

        for entity in ITALIAN_ENTITIES:
            assert entity in data["supported_entities"]
        assert data["configuration"]["supported_languages"] == ["en", "it"]
        assert data["supported_entities_by_language"] == {
            "en": ["PERSON"],
            "it": ["IT_FISCAL_CODE", "PERSON"],
        }

    def test_info_without_engine(self):
        with patch("main.analyzer_engine", None):
            data = client.get("/info").json()

        assert data["supported_entities_by_language"] == {}


@pytest.mark.unit
class TestLifespan:
    def test_startup_loads_models_for_enabled_languages(self):
        models = {"en": "en_core_web_lg", "it": "it_core_news_lg", "de": "unused"}
        with patch("main.analyzer_engine", None), patch(
            "main.anonymizer_engine", None
        ), patch("main.Config.SPACY_MODELS", models), patch(
            "main.Config.ENABLED_LANGUAGES", ["en", "it"]
        ), patch(
            "main.create_analyzer_engine"
        ) as create:
            with TestClient(app) as test_client:
                assert test_client.get("/health").json()["status"] == "healthy"

        create.assert_called_once_with(
            {"en": "en_core_web_lg", "it": "it_core_news_lg"},
            detect_organizations=main.Config.DETECT_ORGANIZATIONS,
        )

    @pytest.mark.parametrize(
        "enabled, message",
        [([], "No language enabled"), (["it"], "DEFAULT_LANGUAGE 'en' is not enabled")],
    )
    def test_startup_fails_without_usable_languages(self, enabled, message):
        with patch("main.analyzer_engine", None), patch(
            "main.anonymizer_engine", None
        ), patch("main.Config.ENABLED_LANGUAGES", enabled), patch(
            "main.Config.DEFAULT_LANGUAGE", "en"
        ):
            with pytest.raises(RuntimeError, match=message):
                with TestClient(app):
                    pass


@pytest.fixture(scope="module")
def italian_engine():
    pytest.importorskip("it_core_news_lg")
    return create_analyzer_engine({"it": "it_core_news_lg"})


@pytest.fixture
def italian_client(italian_engine):
    with patch("main.analyzer_engine", italian_engine), patch(
        "main.anonymizer_engine", AnonymizerEngine()
    ), patch("main.Config.ENABLED_LANGUAGES", ["en", "it"]):
        yield lambda text, **config: client.post(
            "/anonymize",
            json={"text": text, "language": "it", "config": config or None},
        ).json()


def assert_no_overlaps(entities):
    ordered = sorted(entities, key=lambda e: e["start"])
    for previous, current in zip(ordered, ordered[1:]):
        assert previous["end"] <= current["start"], (previous, current)


@pytest.mark.italian_model
@pytest.mark.integration
class TestItalianModel:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "Il paziente Giuseppe Verdi è ricoverato.",
                "Il paziente <PERSON> è ricoverato.",
            ),
            ("Codice fiscale RSSMRA80C12F205H", "Codice fiscale <IT_FISCAL_CODE>"),
            ("codice fiscale RSSMRA80C12F205X", "codice fiscale <IT_FISCAL_CODE>"),
            ("P.IVA 07643520567", "P.IVA <IT_VAT_CODE>"),
            (
                "Carta identità CA12345AB, patente MI1234567A, passaporto YA1234567",
                "Carta identità <IT_IDENTITY_CARD>, patente <IT_DRIVER_LICENSE>, "
                "passaporto <IT_PASSPORT>",
            ),
            (
                "Residente in Via Garibaldi 12, 20121 Milano (MI).",
                "Residente in <LOCATION>, <IT_POSTAL_CODE> <LOCATION> (<LOCATION>).",
            ),
            ("Lavora per Heltyca S.r.l.", "Lavora per Heltyca S.r.l."),
            (
                "cell. 347 1234567, tel 06 68401, +39 347 123 4567",
                "cell. <PHONE_NUMBER>, tel <PHONE_NUMBER>, <PHONE_NUMBER>",
            ),
            (
                "IBAN IT60X0542811101000000123456 email mario.rossi@example.it",
                "IBAN <IBAN_CODE> email <EMAIL_ADDRESS>",
            ),
        ],
    )
    def test_examples(self, italian_client, text, expected):
        data = italian_client(text)

        assert data["anonymized_text"] == expected
        assert len(data["detected_entities"]) == expected.count("<")
        assert_no_overlaps(data["detected_entities"])

    def test_full_record(self, italian_client):
        text = (
            "Il sig. Mario Rossi, codice fiscale RSSMRA80C12F205X, residente in "
            "Via Garibaldi 12, 20121 Milano. Email mario.rossi@example.it, cell. "
            "+39 347 123 4567. IBAN IT60X0542811101000000123456. P.IVA 07643520567. "
            "Passaporto YA1234567."
        )

        data = italian_client(text)

        for pii in [
            "Mario",
            "Rossi",
            "RSSMRA80C12F205X",
            "Garibaldi",
            "12,",
            "20121",
            "Milano",
            "mario.rossi@example.it",
            "347",
            "IT60X0542811101000000123456",
            "07643520567",
            "YA1234567",
        ]:
            assert pii not in data["anonymized_text"]
        assert_no_overlaps(data["detected_entities"])

    def test_entities_to_anonymize(self, italian_client):
        data = italian_client(
            "Mario Rossi, codice fiscale RSSMRA80C12F205X",
            entities_to_anonymize=["IT_FISCAL_CODE"],
        )

        # An explicit config uses the REPLACE strategy's "<ANONYMIZED>" default
        assert data["anonymized_text"] == "Mario Rossi, codice fiscale <ANONYMIZED>"
        assert [e["entity_type"] for e in data["detected_entities"]] == [
            "IT_FISCAL_CODE"
        ]

    def test_organizations_detection(self):
        pytest.importorskip("it_core_news_lg")
        engine = create_analyzer_engine(
            {"it": "it_core_news_lg"}, detect_organizations=True
        )

        results = engine.analyze("Lavora per Heltyca S.r.l.", language="it")

        assert "ORGANIZATION" in {r.entity_type for r in results}
