import json
import os
from dataclasses import dataclass
from typing import Any

import altair as alt
import pandas as pd
import requests
from dotenv import load_dotenv

PROXYAPI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Lead:
    company_name: str
    industry: str
    emp_count: int
    engagement_score: int


def baseline_proba_closed_deal(lead: Lead) -> float:
    """
    Deterministic prior based on the hidden rule embedded in the synthetic dataset.
    """
    if lead.industry == "Fintech" and lead.emp_count > 200 and lead.engagement_score > 60:
        return 0.85
    return 0.15


def baseline_score_and_priority(
    lead: Lead, win_rates: dict[str, float]
) -> tuple[int, str, str]:
    """
    Produce a reproducible baseline assessment.

    - Start from hidden-rule probability (85% vs 15%)
    - Blend slightly with historical win rate for the industry (if available)
    - Convert to score 0..100 and a priority label
    """
    p_rule = baseline_proba_closed_deal(lead)
    p_industry = win_rates.get(lead.industry)

    if p_industry is None:
        p = p_rule
        industry_note = "Для этой индустрии нет исторического Win Rate — использован только prior по правилу."
    else:
        # Keep the hidden rule dominant so the pattern remains learnable.
        p = 0.75 * p_rule + 0.25 * float(p_industry)
        industry_note = (
            f"Исторический Win Rate по индустрии: {round(p_industry * 100, 2)}% (учтён с весом 25%)."
        )

    score = int(round(max(0.0, min(1.0, p)) * 100))

    if score >= 70:
        priority = "High"
    elif score >= 40:
        priority = "Medium"
    else:
        priority = "Low"

    reasons = [
        f"Prior по правилу (Fintech & emp_count>200 & engagement_score>60): {round(p_rule * 100)}%.",
        industry_note,
        f"Итоговая baseline-оценка: {score}/100 → Priority={priority}.",
    ]
    return score, priority, "\n".join(reasons)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "company_name",
        "industry",
        "emp_count",
        "revenue_mln",
        "engagement_score",
        "lead_source",
        "closed_deal",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df


def win_rate_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    wr = (
        df.groupby("industry", as_index=False)["closed_deal"]
        .mean()
        .rename(columns={"closed_deal": "win_rate"})
    )
    wr["win_rate_pct"] = (wr["win_rate"] * 100).round(2)
    return wr.sort_values("win_rate_pct", ascending=False)


def _bin_emp_count(series: pd.Series) -> pd.Series:
    bins = [-float("inf"), 200, 500, 2000, 5000, float("inf")]
    labels = ["<=200", "201-500", "501-2000", "2001-5000", "5000+"]
    return pd.cut(series.astype(float), bins=bins, labels=labels, right=True, include_lowest=True)


def _bin_engagement(series: pd.Series) -> pd.Series:
    bins = [-float("inf"), 40, 60, 80, 100]
    labels = ["1-40", "41-60", "61-80", "81-100"]
    s = series.astype(float).clip(lower=1, upper=100)
    return pd.cut(s, bins=bins, labels=labels, right=True, include_lowest=True)


def segment_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    seg = df.copy()
    seg["emp_bin"] = _bin_emp_count(seg["emp_count"])
    seg["engagement_bin"] = _bin_engagement(seg["engagement_score"])

    base_rate = float(seg["closed_deal"].mean())
    grouped = (
        seg.groupby(["industry", "emp_bin", "engagement_bin"], dropna=False)["closed_deal"]
        .agg(n="count", win_rate="mean")
        .reset_index()
    )
    grouped["win_rate_pct"] = (grouped["win_rate"] * 100).round(2)
    grouped["lift_vs_overall_pct_points"] = ((grouped["win_rate"] - base_rate) * 100).round(2)
    return grouped.sort_values(["win_rate", "n"], ascending=[False, False])


def mine_simple_rules(
    df: pd.DataFrame, min_support: int = 20, top_k: int = 8
) -> pd.DataFrame:
    """
    Brute-force a small set of interpretable rules from the history.
    We search combinations of:
      - industry == X (optional)
      - emp_count > T (optional)
      - engagement_score > T (optional)
    """
    base_rate = float(df["closed_deal"].mean())

    industries = sorted(df["industry"].dropna().unique().tolist())
    emp_thresholds = [200, 500, 1000, 2000]
    eng_thresholds = [40, 60, 80]

    candidates: list[dict[str, Any]] = []

    def add_rule(name: str, mask: pd.Series) -> None:
        n = int(mask.sum())
        if n < min_support:
            return
        win_rate = float(df.loc[mask, "closed_deal"].mean())
        lift = win_rate - base_rate
        candidates.append(
            {
                "rule": name,
                "support_n": n,
                "win_rate": win_rate,
                "win_rate_pct": round(win_rate * 100, 2),
                "lift_pct_points": round(lift * 100, 2),
            }
        )

    # 1-condition rules
    for ind in industries:
        add_rule(f"industry == {ind}", df["industry"] == ind)
    for t in emp_thresholds:
        add_rule(f"emp_count > {t}", df["emp_count"].astype(float) > t)
    for t in eng_thresholds:
        add_rule(f"engagement_score > {t}", df["engagement_score"].astype(float) > t)

    # 2-condition rules
    for ind in industries:
        ind_mask = df["industry"] == ind
        for t in emp_thresholds:
            add_rule(f"industry == {ind} AND emp_count > {t}", ind_mask & (df["emp_count"].astype(float) > t))
        for t in eng_thresholds:
            add_rule(
                f"industry == {ind} AND engagement_score > {t}",
                ind_mask & (df["engagement_score"].astype(float) > t),
            )

    for t_emp in emp_thresholds:
        emp_mask = df["emp_count"].astype(float) > t_emp
        for t_eng in eng_thresholds:
            add_rule(
                f"emp_count > {t_emp} AND engagement_score > {t_eng}",
                emp_mask & (df["engagement_score"].astype(float) > t_eng),
            )

    # 3-condition rules
    for ind in industries:
        ind_mask = df["industry"] == ind
        for t_emp in emp_thresholds:
            emp_mask = df["emp_count"].astype(float) > t_emp
            for t_eng in eng_thresholds:
                add_rule(
                    f"industry == {ind} AND emp_count > {t_emp} AND engagement_score > {t_eng}",
                    ind_mask & emp_mask & (df["engagement_score"].astype(float) > t_eng),
                )

    rules_df = pd.DataFrame(candidates)
    if rules_df.empty:
        return rules_df
    return rules_df.sort_values(["lift_pct_points", "support_n"], ascending=[False, False]).head(top_k)


def train_simple_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Train a lightweight interpretable model: Logistic Regression.
    Returns metrics and a compact feature importance table.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"scikit-learn not available: {e}"}

    data = df[["industry", "emp_count", "engagement_score", "closed_deal"]].dropna().copy()
    data["emp_count"] = data["emp_count"].astype(float)
    data["engagement_score"] = data["engagement_score"].astype(float)
    y = data["closed_deal"].astype(int)
    X = data.drop(columns=["closed_deal"])

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["industry"]),
            ("num", "passthrough", ["emp_count", "engagement_score"]),
        ]
    )
    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "ok": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else None,
    }

    # Feature importance (coefficients)
    ohe: OneHotEncoder = pipe.named_steps["pre"].named_transformers_["cat"]  # type: ignore[assignment]
    cat_names = [f"industry={c}" for c in ohe.get_feature_names_out(["industry"]).tolist()]
    feat_names = cat_names + ["emp_count", "engagement_score"]
    coefs = pipe.named_steps["clf"].coef_[0].tolist()  # type: ignore[assignment]
    imp = pd.DataFrame({"feature": feat_names, "coef": coefs})
    imp["abs_coef"] = imp["coef"].abs()
    imp = imp.sort_values("abs_coef", ascending=False).head(12).reset_index(drop=True)

    metrics["top_coefficients"] = imp.to_dict(orient="records")
    return metrics


def get_api_key() -> str:
    load_dotenv(override=False)
    key = os.getenv("PROXYAPI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "API key not found. Add PROXYAPI_API_KEY to your .env file (recommended)."
        )
    return key


def _safe_json_extract(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON object in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_proxyapi(lead: Lead, win_rates: dict[str, float]) -> dict[str, Any]:
    api_key = get_api_key()

    system = (
        "You are a sales intelligence assistant. "
        "Given a lead profile and historical win-rate statistics by industry, "
        "and a baseline deterministic assessment computed from business rules, "
        "produce a numeric score (0-100) estimating likelihood/quality, a priority label, "
        "and a detailed justification. "
        "Return ONLY valid JSON."
    )

    baseline_score, baseline_priority, baseline_rationale = baseline_score_and_priority(
        lead, win_rates
    )

    user = {
        "lead": {
            "company_name": lead.company_name,
            "industry": lead.industry,
            "emp_count": lead.emp_count,
            "engagement_score": lead.engagement_score,
        },
        "historical_win_rate_by_industry_pct": {
            k: round(v * 100, 2) for k, v in win_rates.items()
        },
        "baseline_assessment": {
            "score": baseline_score,
            "priority": baseline_priority,
            "rationale": baseline_rationale,
        },
        "required_output_schema": {
            "score": "integer 0..100",
            "priority": "one of: High, Medium, Low",
            "justification": "string, detailed reasoning",
        },
    }

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        f"{PROXYAPI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    content = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")  # type: ignore[call-arg]
    )
    parsed = _safe_json_extract(content)
    if not parsed:
        raise RuntimeError(f"Failed to parse model JSON. Raw content:\n{content}")
    return parsed


def call_proxyapi_history_summary(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Send aggregated historical analysis to the LLM for a compact business-friendly summary.
    """
    api_key = get_api_key()
    system = (
        "You are a sales analytics assistant. "
        "You will receive aggregated historical sales outcomes (wins/losses), segmented win rates, "
        "mined interpretable rules, and (optionally) simple model metrics. "
        "Return ONLY valid JSON with concise insights and recommendations."
    )
    user = {
        "analysis": analysis_payload,
        "required_output_schema": {
            "overall_summary": "string (2-4 sentences)",
            "key_drivers": "array of 3-6 bullets (strings)",
            "top_segments_to_prioritize": "array of objects {segment, why}",
            "segments_to_deprioritize": "array of objects {segment, why}",
            "recommended_next_steps": "array of 3-6 bullets (strings)",
        },
    }

    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        f"{PROXYAPI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _safe_json_extract(content)
    if not parsed:
        raise RuntimeError(f"Failed to parse model JSON. Raw content:\n{content}")
    return parsed


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AI Sales Predictor", layout="wide")
    st.title("AI Sales Predictor")

    csv_path = "historical_sales_data.csv"
    try:
        df = load_data(csv_path)
    except Exception as e:
        st.error(f"Не удалось загрузить `{csv_path}`: {e}")
        st.stop()

    wr_df = win_rate_by_industry(df)
    win_rates = {
        row["industry"]: float(row["win_rate"]) for _, row in wr_df.iterrows()
    }

    with st.sidebar:
        st.subheader("Win Rate по индустриям")
        st.dataframe(
            wr_df[["industry", "win_rate_pct"]].rename(
                columns={"industry": "Industry", "win_rate_pct": "Win Rate (%)"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Всего лидов: **{len(df)}**")

    tab_predictor, tab_history = st.tabs(["Lead Scoring", "Historical Analysis"])

    with tab_predictor:
        st.subheader("Новый лид")
        with st.form("lead_form", border=True):
            company_name = st.text_input("Название", value="")
            industry = st.selectbox(
                "Отрасль",
                options=sorted(df["industry"].dropna().unique().tolist()),
            )
            emp_count = st.number_input(
                "Кол-во сотрудников",
                min_value=1,
                max_value=200_000,
                value=250,
                step=1,
            )
            engagement_score = st.slider(
                "Вовлеченность", min_value=1, max_value=100, value=50
            )
            analyze = st.form_submit_button("Анализировать", type="primary")

        if analyze:
            if not company_name.strip():
                st.error("Введите название компании.")
                st.stop()

            lead = Lead(
                company_name=company_name.strip(),
                industry=str(industry),
                emp_count=int(emp_count),
                engagement_score=int(engagement_score),
            )

            base_score, base_priority, base_rationale = baseline_score_and_priority(
                lead, win_rates
            )

            st.subheader("Baseline (правила + статистика)")
            b1, b2 = st.columns([1, 1])
            with b1:
                st.metric("Baseline Score", base_score)
            with b2:
                st.metric("Baseline Priority", base_priority)
            st.text(base_rationale)

            with st.spinner("Отправляю данные в ИИ…"):
                try:
                    result = call_proxyapi(lead, win_rates)
                except Exception as e:
                    st.error(f"Ошибка при обращении к ProxyAPI: {e}")
                    st.stop()

            score = result.get("score")
            priority = result.get("priority")
            justification = result.get("justification")

            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Score", score)
            with c2:
                st.metric("Priority", priority)

            st.subheader("Обоснование")
            st.write(justification)

            with st.expander("Сырой ответ (JSON)", expanded=False):
                st.json(result)

    with tab_history:
        st.subheader("Краткий срез по истории сделок")

        overall_win_rate = float(df["closed_deal"].mean())
        h1, h2, h3 = st.columns([1, 1, 1])
        with h1:
            st.metric("Всего наблюдений", int(len(df)))
        with h2:
            st.metric("Wins", int(df["closed_deal"].sum()))
        with h3:
            st.metric("Overall Win Rate", f"{overall_win_rate*100:.2f}%")

        st.markdown("**Win Rate по индустриям**")
        chart = (
            alt.Chart(wr_df)
            .mark_bar()
            .encode(
                x=alt.X("win_rate_pct:Q", title="Win Rate (%)"),
                y=alt.Y("industry:N", sort="-x", title="Industry"),
                tooltip=[
                    alt.Tooltip("industry:N", title="Industry"),
                    alt.Tooltip("win_rate_pct:Q", title="Win Rate (%)", format=".2f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(
            wr_df[["industry", "win_rate_pct"]].rename(
                columns={"industry": "Industry", "win_rate_pct": "Win Rate (%)"}
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("**Сегменты (industry × emp_bin × engagement_bin)**")
        seg_df = segment_win_rates(df)
        st.dataframe(
            seg_df.head(20).rename(
                columns={
                    "industry": "Industry",
                    "emp_bin": "Emp Bin",
                    "engagement_bin": "Engagement Bin",
                    "n": "N",
                    "win_rate_pct": "Win Rate (%)",
                    "lift_vs_overall_pct_points": "Lift (pp)",
                }
            ),
            hide_index=True,
            width="stretch",
        )

        st.markdown("**Найденные простые правила (интерпретируемые)**")
        rules_df = mine_simple_rules(df, min_support=20, top_k=8)
        if rules_df.empty:
            st.info("Не удалось найти правила с достаточной поддержкой (min_support).")
        else:
            st.dataframe(
                rules_df.rename(
                    columns={
                        "rule": "Rule",
                        "support_n": "Support (N)",
                        "win_rate_pct": "Win Rate (%)",
                        "lift_pct_points": "Lift (pp)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.markdown("**Простая модель (Logistic Regression)**")
        model_result = train_simple_model(df)
        if not model_result.get("ok"):
            st.warning(model_result.get("error", "Model training failed."))
        else:
            m1, m2 = st.columns([1, 1])
            with m1:
                st.metric("Accuracy", f"{model_result['accuracy']:.3f}")
            with m2:
                roc = model_result.get("roc_auc")
                st.metric("ROC AUC", f"{roc:.3f}" if roc is not None else "n/a")

            top_coefs = pd.DataFrame(model_result["top_coefficients"])
            st.dataframe(top_coefs, hide_index=True, width="stretch")

        st.divider()
        st.subheader("AI Summary (по агрегатам)")
        st.caption(
            "Отправляем в ИИ только агрегированные результаты анализа (без всей таблицы строк)."
        )

        analysis_payload = {
            "overall": {
                "n": int(len(df)),
                "wins": int(df["closed_deal"].sum()),
                "overall_win_rate_pct": round(overall_win_rate * 100, 2),
            },
            "win_rate_by_industry_pct": {
                row["industry"]: float(row["win_rate_pct"]) for _, row in wr_df.iterrows()
            },
            "top_segments": seg_df.head(12)[
                ["industry", "emp_bin", "engagement_bin", "n", "win_rate_pct", "lift_vs_overall_pct_points"]
            ].to_dict(orient="records"),
            "top_rules": rules_df.to_dict(orient="records") if not rules_df.empty else [],
            "model": model_result,
        }

        if st.button("Сформировать AI Summary", type="primary"):
            with st.spinner("Формирую summary в ИИ…"):
                try:
                    summary = call_proxyapi_history_summary(analysis_payload)
                except Exception as e:
                    st.error(f"Ошибка при обращении к ProxyAPI: {e}")
                    st.stop()
            st.json(summary)


if __name__ == "__main__":
    main()

