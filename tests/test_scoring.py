from app import Lead, baseline_proba_closed_deal, baseline_score_and_priority


def test_baseline_prior_rule_true() -> None:
    lead = Lead(
        company_name="TestCo",
        industry="Fintech",
        emp_count=201,
        engagement_score=61,
    )
    assert baseline_proba_closed_deal(lead) == 0.85


def test_baseline_prior_rule_false() -> None:
    lead = Lead(
        company_name="TestCo",
        industry="SaaS",
        emp_count=5000,
        engagement_score=100,
    )
    assert baseline_proba_closed_deal(lead) == 0.15


def test_baseline_score_blends_with_industry_rate() -> None:
    lead = Lead(
        company_name="TestCo",
        industry="Fintech",
        emp_count=500,
        engagement_score=70,
    )
    win_rates = {"Fintech": 0.5}
    score, priority, rationale = baseline_score_and_priority(lead, win_rates)
    assert score == round(100 * (0.75 * 0.85 + 0.25 * 0.5))
    assert priority in {"High", "Medium", "Low"}
    assert "Исторический Win Rate" in rationale

