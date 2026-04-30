from __future__ import annotations


FINAL_RESEARCH_QUESTION = (
    "To what extent do different volcanic activity metrics, including named hotspot "
    "occurrence, estimated thermal intensity, and metadata-normalized observation "
    "activity, produce different spatial interpretations of Io's volcanism on a "
    "common 1 deg grid?"
)


def test_public_faq_has_twelve_current_entries():
    from dashboard.i18n import TRANSLATIONS

    faq = TRANSLATIONS["en"]
    questions = [faq[f"faq.public.q{i}"] for i in range(1, 13)]
    answers = [faq[f"faq.public.a{i}"] for i in range(1, 13)]

    assert len(questions) == 12
    assert len(answers) == 12
    assert questions[0] == "What is this project about?"
    assert "No single map should be treated as the complete truth" in answers[7]
    assert "does not forecast future eruptions" in answers[9]


def test_researcher_faq_has_twenty_current_entries():
    from dashboard.i18n import TRANSLATIONS

    faq = TRANSLATIONS["en"]
    questions = [faq[f"faq.research.q{i}"] for i in range(1, 21)]
    answers = [faq[f"faq.research.a{i}"] for i in range(1, 21)]

    assert len(questions) == 20
    assert len(answers) == 20
    assert FINAL_RESEARCH_QUESTION in answers[0]
    assert "metadata-normalized observation activity" in answers[0]
    assert "not true footprint/sensitivity correction" in answers[5]
    assert "Spearman `-0.424`" in answers[11]
    assert "top 50 contribute `73.3%`" in answers[12]


def test_faq_categories_match_researcher_filter_plan():
    from dashboard.i18n import TRANSLATIONS

    faq = TRANSLATIONS["en"]
    expected = {
        "overview",
        "grid",
        "metrics",
        "intensity",
        "coverage",
        "comparison",
        "time",
        "tidal",
        "limits",
        "publish",
    }

    assert {key.removeprefix("faq.category.") for key in faq if key.startswith("faq.category.")} >= expected


def test_faq_copy_avoids_unsupported_affirmative_claims():
    from dashboard.i18n import TRANSLATIONS

    faq = TRANSLATIONS["en"]
    active_answers = "\n".join(
        [faq[f"faq.public.a{i}"] for i in range(1, 13)]
        + [faq[f"faq.research.a{i}"] for i in range(1, 21)]
    ).lower()

    assert "true coverage-corrected" not in active_answers
    assert "complete energy budget" not in active_answers
    assert "solved energy budget" not in active_answers
    assert "proves" not in active_answers


def test_faq_renderer_uses_all_public_and_researcher_entries():
    import inspect

    from dashboard.app import page_faq

    source = inspect.getsource(page_faq)

    assert "range(1, 13)" in source
    assert '"faq.research.q20"' in source
    assert '"publish"' in source
