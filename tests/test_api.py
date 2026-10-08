from __future__ import annotations

from fastapi.testclient import TestClient

from environment.api import app


client = TestClient(app)


SUCCESSFUL_EMAIL = {
    "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
    "format": "json",
}


def test_run_endpoint_executes_allowlisted_mock_agent() -> None:
    response = client.post(
        "/runs",
        json={
            "environment": "openenv",
            "task": "email_classification",
            "agent": "mock",
            "max_steps": 3,
            "actions": [SUCCESSFUL_EMAIL],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["run_id"]
    assert result["environment"] == "email_classification"
    assert result["completed"] is True
    assert result["steps"] == 1
    assert result["grading"]["score"] == 1.0


def test_run_endpoint_rejects_unknown_environment() -> None:
    response = client.post(
        "/runs",
        json={"environment": "arbitrary.module.Environment", "agent": "mock", "actions": [SUCCESSFUL_EMAIL]},
    )

    assert response.status_code == 400
    assert "Unknown registered environment" in response.json()["detail"]


def test_run_endpoint_rejects_unknown_agent() -> None:
    response = client.post(
        "/runs",
        json={"agent": "arbitrary.module.Agent", "actions": [SUCCESSFUL_EMAIL]},
    )

    assert response.status_code == 400
    assert "Unknown registered agent" in response.json()["detail"]


def test_result_retrieval_endpoints_return_persisted_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENENV_RESULTS_DB", str(tmp_path / "api-results.db"))
    from environment.api import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/runs",
        json={
            "agent": "mock",
            "actions": [
                {
                    "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
                    "format": "json",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()

    stored = client.get(f"/results/runs/{payload['run_id']}")
    assert stored.status_code == 200
    assert stored.json()["trajectory"]
    by_agent = client.get("/results/agents/mock")
    assert by_agent.status_code == 200
    assert any(item["run_id"] == payload["run_id"] for item in by_agent.json())


def test_run_endpoint_requires_actions_for_mock_agent() -> None:
    response = client.post("/runs", json={"agent": "mock"})

    assert response.status_code == 400
    assert "actions are required" in response.json()["detail"]


def test_existing_reset_endpoint_is_unchanged() -> None:
    response = client.post("/reset", json={"task_name": "email_classification"})

    assert response.status_code == 200
    assert response.json()["task_name"] == "email_classification"


def test_run_endpoint_executes_registered_coding_environment() -> None:
    response = client.post(
        "/runs",
        json={
            "environment": "coding",
            "task": "fix_addition",
            "agent": "mock",
            "max_steps": 4,
            "actions": [
                {
                    "content": '{"tool":"edit_file","path":"solution.py","content":"def add_numbers(first: int, second: int) -> int:\\n    return first + second\\n"}',
                    "format": "json",
                },
                {"content": '{"tool":"submit"}', "format": "json"},
            ],
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["completed"] is True
    assert result["task_id"] == "fix_addition"
    assert result["grading"]["score"] == 1.0
