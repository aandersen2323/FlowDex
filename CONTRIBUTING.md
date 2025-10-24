# Contributing to FlowDex

Thank you for investing time in improving FlowDex! This guide summarizes the expectations for local development and the recommended debugging workflow used by the project.

## Local Development Setup

1. **Create a virtual environment** (Python 3.10+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the application dependencies**:
   ```bash
   pip install -r server/requirements.txt
   ```
3. **Install developer tooling** (tests + UI automation):
   ```bash
   pip install pytest pytest-asyncio httpx==0.25.2 playwright
   playwright install  # downloads browser binaries
   ```
4. (Optional) **Run the API locally** for manual testing:
   ```bash
   uvicorn server.app:app --reload --port 8787
   ```
5. (Optional) **Start the full docker-compose stack** when you need Redis + the MCP bridge:
   ```bash
   docker compose up --build
   ```

## Recommended Debugging Workflow

Follow this workflow whenever you reproduce or validate a bug fix. It keeps feedback loops fast and avoids jumping straight to manual reproduction.

1. **Start with automated tests and logs.**
   - Run the full suite with `pytest server/tests`.
   - When the docker-compose stack is running, stream API logs with `docker compose logs -f api` and Redis logs with `docker compose logs -f redis` to surface server-side errors quickly.
   - For CLI regressions, run targeted scripts from the `examples/` directory to capture console output.
2. **Use Playwright for UI confirmation.**
   - Add or update Playwright tests alongside the feature you are touching.
   - Execute them with `pytest tests/ui` (or any other directory you create for UI checks) to confirm behaviour automatically.
3. **Capture targeted Playwright scripts when results are ambiguous.**
   - Write a focused script that reproduces the user flow in question.
   - Save a screenshot or trace artifact (e.g., `--screenshot on`, `--video on`) so the investigation is auditable.
4. **Escalate to manual verification only as a last resort.**
   - Manual testing is great for sanity checks but should be the exception, not the norm.
   - Always record the exact steps and environment if you do perform a manual run.

Keeping investigations scripted ensures regressions are caught by CI and reduces the effort required to share findings with the rest of the team.

## Pull Request Checklist

Before opening a pull request:

- [ ] Ensure linting and tests pass locally.
- [ ] Update documentation, examples, and configuration defaults when behaviour changes.
- [ ] Include any new or updated Playwright tests/snapshots required to prove UI changes.
- [ ] Provide clear reproduction steps and links to failing tests/logs in the PR description.

We appreciate your contributions! If anything in this guide is unclear, open an issue so we can improve it.
