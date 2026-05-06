## Contributing

Thanks for your interest in contributing!

### Local setup

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

### Run checks

```bash
./venv/bin/ruff check .
./venv/bin/pytest -q
```

### Pull request guidelines
- Keep PRs small and focused (one feature/fix per PR).
- Include a short test plan in the PR description.
- If you change the scoring logic, update the README section explaining it.

