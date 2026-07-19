"""OpenHealth working package — task specs, facade API, CLI (ehr-ai).

Public marketing rebrand deferred; this is the internal framework namespace.
"""

from openhealth.task_spec import TaskSpec, list_tasks, load_task

__version__ = "1.0.0"

__all__ = ["TaskSpec", "list_tasks", "load_task", "__version__"]
