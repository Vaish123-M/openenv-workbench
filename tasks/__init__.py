from .base import BaseTask
from .customer_support import CustomerSupportReplyTask
from .data_cleaning import DataCleaningTask
from .email_classification import EmailClassificationTask

TASK_REGISTRY = {
    "email_classification": EmailClassificationTask(),
    "data_cleaning": DataCleaningTask(),
    "customer_support_reply": CustomerSupportReplyTask(),
}


def get_task(name: str) -> BaseTask:
    try:
        return TASK_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {name}") from exc


def list_tasks() -> list[str]:
    return list(TASK_REGISTRY.keys())
