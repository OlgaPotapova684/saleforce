## AI Sales Predictor (Streamlit)

Streamlit-приложение для:
- скоринга нового лида (baseline + ProxyAPI),
- анализа истории сделок/отказов (сегменты, правила, простая модель),
- AI summary по агрегированным результатам анализа.

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

