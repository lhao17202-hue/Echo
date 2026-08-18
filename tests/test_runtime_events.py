from echo.runtime.events import RuntimeEvent, render_runtime_events
from echo.scheduler.cron_scheduler import CronJob


def test_render_runtime_events_groups_events_for_model_context():
    events = [
        RuntimeEvent(source="teammate", event_type="task_completed", content="README title is Echo", metadata={"from": "researcher"}),
        RuntimeEvent(source="cron", event_type="fired", content="check status", metadata={"job_id": "cron_1"}),
    ]

    rendered = render_runtime_events(events)

    assert "## Runtime Events" in rendered
    assert "[teammate/task_completed]" in rendered
    assert "from=researcher" in rendered
    assert "README title is Echo" in rendered
    assert "[cron/fired]" in rendered
    assert "job_id=cron_1" in rendered
    assert "check status" in rendered




def test_runtime_event_from_cron_job_preserves_job_metadata():
    job = CronJob(job_id="cron_123", cron_expr="* * * * *", prompt="check status", recurring=False)

    event = RuntimeEvent.from_cron_job(job)

    assert event.source == "cron"
    assert event.event_type == "fired"
    assert event.content == "check status"
    assert event.metadata["job_id"] == "cron_123"
    assert event.metadata["cron_expr"] == "* * * * *"
    assert event.metadata["recurring"] is False
