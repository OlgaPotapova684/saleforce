## AI Sales Predictor (Streamlit)

Demo project “AI Sales Predictor”: a Streamlit app that shows how to:
- score a new lead (lead prioritization),
- analyze historical wins/losses,
- use an LLM as an *interpreter* of quantitative results (not as the only source of truth).

## What’s inside

### 1) Lead Scoring (new lead)
The form takes 4 inputs:
- `company_name` (display only),
- `industry`,
- `emp_count`,
- `engagement_score`.

The app produces a **two-layer assessment**:

- **Baseline (deterministic scoring + statistics)**  
  Uses the “hidden” pattern embedded in the synthetic dataset:
  - if `industry == Fintech` and `emp_count > 200` and `engagement_score > 60`, then \(P(\text{win}) \approx 0.85\)
  - otherwise \(P(\text{win}) \approx 0.15\)

  Then it lightly blends in the empirical industry Win Rate (if available):
  \[
  p = 0.75 \cdot p_{rule} + 0.25 \cdot p_{industry}
  \]
  \[
  score = round(100 \cdot p)
  \]
  Priority:
  - `High`: score ≥ 70
  - `Medium`: 40–69
  - `Low`: < 40

- **LLM verdict (ProxyAPI → gpt-4o-mini)**  
  The LLM receives the lead profile + win-rate-by-industry + the baseline assessment.  
  It returns JSON: `score (0..100)`, `priority (High/Medium/Low)`, and a detailed justification.

The idea: baseline provides **reproducible math**, while the LLM provides **human-friendly rationale**.

### 2) Historical Analysis (wins vs losses)
Goal: identify patterns in the dataset — “why some deals close and others don’t”.

The UI shows a compact report:

- **Overall Win Rate**: share of `closed_deal=1`
- **Win Rate by industry**: table + bar chart
- **Segments**: `industry × emp_bin × engagement_bin`  
  Where:
  - `emp_bin`: `<=200`, `201-500`, `501-2000`, `2001-5000`, `5000+`
  - `engagement_bin`: `1-40`, `41-60`, `61-80`, `81-100`

  Per segment we compute:
  - `N` (support)
  - `win_rate` / `win_rate_pct`
  - `lift` vs overall Win Rate (percentage points)

- **Simple rule mining (interpretable)**  
  Brute-force rules of the form:
  - `industry == X`
  - `emp_count > T`
  - `engagement_score > T`
  - up to 3 combined conditions  
  We keep the rules with sufficient support (`min_support`) and the highest lift.

- **Simple model (Logistic Regression)**  
  A logistic regression is trained on:
  - `industry` (one-hot),
  - `emp_count`,
  - `engagement_score`.

  The report shows:
  - Accuracy and ROC AUC on a hold-out test split,
  - top coefficients (drivers increasing/decreasing win probability).

### 3) AI Summary (from aggregates)
The **AI Summary** button sends only **aggregated results** to ProxyAPI (not all raw CSV rows):
- overall metrics,
- win rate by industry,
- top segments,
- top rules,
- model metrics/coefficients.

The LLM returns a compact JSON with insights and recommendations (prioritize/deprioritize segments, next steps).

## Project structure
- `app.py` — Streamlit UI + scoring + historical analysis + ProxyAPI integration
- `historical_sales_data.csv` — synthetic sales history (500 rows)
- `generate_historical_sales_data.py` — synthetic data generator (with the hidden pattern)
- `requirements.txt` — dependencies
- `.env.example` — environment variable example (no secrets)

---

Проект-демо “AI Sales Predictor”: Streamlit-приложение, которое показывает, как можно:
- оценивать перспективность нового лида (скоринг),
- анализировать историю сделок/отказов (wins/losses),
- использовать ИИ как “интерпретатор” результатов математического анализа, а не как единственный источник истины.

## Что внутри

### 1) Lead Scoring (оценка нового лида)
Форма принимает 4 параметра:
- `company_name` — для удобного отображения,
- `industry`,
- `emp_count`,
- `engagement_score`.

Дальше приложение делает **двухуровневую оценку**:

- **Baseline (детерминированный скоринг + статистика)**  
  Используется “скрытая” закономерность, заложенная в синтетических данных:
  - если `industry == Fintech` и `emp_count > 200` и `engagement_score > 60`, то \(P(\text{win}) \approx 0.85\)
  - иначе \(P(\text{win}) \approx 0.15\)

  Затем baseline слегка “подмешивает” эмпирический Win Rate индустрии (если доступен), чтобы учитывать историю:
  \[
  p = 0.75 \cdot p_{rule} + 0.25 \cdot p_{industry}
  \]
  \[
  score = round(100 \cdot p)
  \]
  Приоритет:
  - `High`: score ≥ 70
  - `Medium`: 40–69
  - `Low`: < 40

- **LLM-вердикт (ProxyAPI → gpt-4o-mini)**  
  В ИИ отправляется профиль лида + Win Rate по индустриям + baseline-оценка.  
  Модель возвращает JSON: `score (0..100)`, `priority (High/Medium/Low)` и **подробное обоснование**.

Идея: baseline даёт **воспроизводимую математику**, а ИИ — **человеческое объяснение** и “sales-нарратив”.

### 2) Historical Analysis (анализ истории сделок/НЕсделок)
Цель — найти закономерности в базе: “почему одни дошли до сделки, а другие — нет”.

В интерфейсе выводится краткий отчёт:

- **Overall Win Rate**: доля `closed_deal=1` по всем наблюдениям
- **Win Rate по индустриям**: таблица и бар-чарт
- **Сегменты**: `industry × emp_bin × engagement_bin`  
  Где:
  - `emp_bin`: `<=200`, `201-500`, `501-2000`, `2001-5000`, `5000+`
  - `engagement_bin`: `1-40`, `41-60`, `61-80`, `81-100`

  Для каждого сегмента считаются:
  - `N` (support)
  - `win_rate` и `win_rate_pct`
  - `lift` относительно общего Win Rate (в процентных пунктах)

- **Майнинг простых правил (интерпретируемых)**  
  Перебор правил вида:
  - `industry == X`
  - `emp_count > T`
  - `engagement_score > T`
  - комбинации до 3 условий  
  Отбираются правила с достаточной поддержкой (`min_support`) и наибольшим lift.

- **Простая модель (Logistic Regression)**  
  Обучается логистическая регрессия на признаках:
  - `industry` (one-hot),
  - `emp_count`,
  - `engagement_score`.

  В отчёте показываются:
  - Accuracy и ROC AUC на тестовой выборке (train/test split),
  - топ коэффициентов (какие факторы сильнее всего “толкают” вероятность вверх/вниз).

### 3) AI Summary по агрегатам
Кнопка **AI Summary** отправляет в ProxyAPI **только агрегированные результаты** анализа (без всех строк CSV):
- overall-метрики,
- win rate по индустриям,
- топ сегментов,
- топ правил,
- метрики/коэффициенты модели.

ИИ возвращает компактный JSON с инсайтами и рекомендациями (что приоритезировать, что деприоритезировать, какие next steps).

## Структура проекта
- `app.py` — Streamlit UI + скоринг + исторический анализ + ProxyAPI интеграция
- `historical_sales_data.csv` — синтетическая история продаж (500 строк)
- `generate_historical_sales_data.py` — генератор синтетических данных (со “скрытой” закономерностью)
- `requirements.txt` — зависимости
- `.env.example` — пример переменных окружения (без секретов)

### Запуск локально

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

### Настройка ключа ProxyAPI

Создайте `.env` (или используйте `.env.example`) и добавьте:

```bash
PROXYAPI_API_KEY=...
```

### Деплой (рекомендуется)

- Streamlit Community Cloud: укажите репозиторий GitHub и файл `app.py`, добавьте secret `PROXYAPI_API_KEY`.
- Альтернативы: Render / Railway / Fly.io.

