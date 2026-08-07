import uuid
import time
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

@dataclass
class BackgroundJob:
    job_id: str
    job_type: str
    status: str = "queued" # queued, running, completed, failed
    progress: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class JobsService:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, BackgroundJob] = {}

    def submit_job(self, job_type: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = str(uuid.uuid4())
        job = BackgroundJob(job_id=job_id, job_type=job_type)
        self.jobs[job_id] = job
        
        def runner():
            job.status = "running"
            job.started_at = time.time()
            try:
                res = fn(*args, **kwargs)
                job.result = res
                job.status = "completed"
                job.progress = 1.0
            except Exception as e:
                job.error = str(e)
                job.status = "failed"
            finally:
                job.finished_at = time.time()

        self.executor.submit(runner)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "type": job.job_type,
            "status": job.status,
            "progress": job.progress,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result": job.result,
            "error": job.error
        }
