"""Power Interpreter - Async Job Manager

Manages long-running code execution jobs with:
- Submit/poll/result pattern (prevents timeouts)
- Concurrent job execution with limits
- Job status tracking in PostgreSQL
- Automatic cleanup of old jobs

Pattern:
  1. Client submits job -> gets job_id immediately
  2. Client polls job status -> gets progress
  3. Client gets result when complete -> gets full output

This ensures NO MCP call ever times out, even for 5-minute jobs.
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Job, JobStatus
from app.engine.executor import executor, ExecutionResult
from app.database import get_session_factory

logger = logging.getLogger(__name__)


class JobManager:
    """Manages async code execution jobs"""
    
    def __init__(self):
        self._running_jobs: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)
    
    async def submit_job(
        self,
        code: str,
        session_id: str = None,
        timeout: int = None,
        context: Dict = None,
        metadata: Dict = None
    ) -> str:
        """Submit a job for async execution
        
        Returns job_id immediately. Use get_job_status() to check progress.
        """
        job_id = str(uuid.uuid4())
        timeout = timeout or settings.JOB_TIMEOUT
        
        # Create job record in database
        factory = get_session_factory()
        async with factory() as session:
            job = Job(
                id=uuid.UUID(job_id),
                session_id=uuid.UUID(session_id) if session_id else None,
                code=code,
                status=JobStatus.PENDING,
                submitted_at=datetime.utcnow(),
                metadata_=metadata or {}
            )
            session.add(job)
            await session.commit()
        
        # Start execution in background
        task = asyncio.create_task(
            self._execute_job(job_id, code, session_id or "default", timeout, context)
        )
        self._running_jobs[job_id] = task
        
        # Cleanup task reference when done
        task.add_done_callback(lambda t: self._running_jobs.pop(job_id, None))
        
        logger.info(f"Job {job_id} submitted (session: {session_id})")
        return job_id
    
    async def _execute_job(
        self,
        job_id: str,
        code: str,
        session_id: str,
        timeout: int,
        context: Dict = None
    ):
        """Execute a job with semaphore limiting"""
        async with self._semaphore:
            factory = get_session_factory()
            
            # Update status to RUNNING
            async with factory() as session:
                await session.execute(
                    update(Job)
                    .where(Job.id == uuid.UUID(job_id))
                    .values(
                        status=JobStatus.RUNNING,
                        started_at=datetime.utcnow()
                    )
                )
                await session.commit()
            
            logger.info(f"Job {job_id} started execution")
            
            # Execute the code
            try:
                exec_result = await executor.execute(
                    code=code,
                    session_id=session_id,
                    timeout=timeout,
                    context=context
                )
                
                # Update job with results
                async with factory() as session:
                    status = JobStatus.COMPLETED if exec_result.success else JobStatus.FAILED
                    
                    await session.execute(
                        update(Job)
                        .where(Job.id == uuid.UUID(job_id))
                        .values(
                            status=status,
                            completed_at=datetime.utcnow(),
                            execution_time_ms=exec_result.execution_time_ms,
                            stdout=exec_result.stdout[:100000],  # Limit stored output
                            stderr=exec_result.stderr[:100000],
                            result=exec_result.to_dict(),
                            error_message=exec_result.error_message,
                            error_traceback=exec_result.error_traceback,
                            memory_used_mb=exec_result.memory_used_mb,
                            files_created=exec_result.files_created
                        )
                    )
                    await session.commit()
                
                logger.info(
                    f"Job {job_id} completed: {status.value} "
                    f"({exec_result.execution_time_ms}ms)"
                )
            
            except Exception as e:
                logger.error(f"Job {job_id} failed with unexpected error: {e}")
                async with factory() as session:
                    await session.execute(
                        update(Job)
                        .where(Job.id == uuid.UUID(job_id))
                        .values(
                            status=JobStatus.FAILED,
                            completed_at=datetime.utcnow(),
                            error_message=str(e)
                        )
                    )
                    await session.commit()
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a job"""
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Job).where(Job.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            elapsed = None
            if job.started_at:
                end = job.completed_at or datetime.utcnow()
                elapsed = int((end - job.started_at).total_seconds() * 1000)
            
            return {
                'job_id': str(job.id),
                'status': job.status.value,
                'submitted_at': job.submitted_at.isoformat() if job.submitted_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'elapsed_ms': elapsed,
                'execution_time_ms': job.execution_time_ms,
                'has_result': job.result is not None,
                'has_error': job.error_message is not None,
            }
    
    async def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Get full result of a completed job"""
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(Job).where(Job.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()
            
            if not job:
                return None
            
            return {
                'job_id': str(job.id),
                'status': job.status.value,
                'code': job.code,
                'stdout': job.stdout,
                'stderr': job.stderr,
                'result': job.result,
                'error_message': job.error_message,
                'error_traceback': job.error_traceback,
                'execution_time_ms': job.execution_time_ms,
                'memory_used_mb': job.memory_used_mb,
                'files_created': job.files_created,
                'submitted_at': job.submitted_at.isoformat() if job.submitted_at else None,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        task = self._running_jobs.get(job_id)
        if task and not task.done():
            task.cancel()
            
            factory = get_session_factory()
            async with factory() as session:
                await session.execute(
                    update(Job)
                    .where(Job.id == uuid.UUID(job_id))
                    .values(
                        status=JobStatus.CANCELLED,
                        completed_at=datetime.utcnow()
                    )
                )
                await session.commit()
            
            logger.info(f"Job {job_id} cancelled")
            return True
        return False
    
    async def list_jobs(
        self, 
        session_id: str = None, 
        status: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """List jobs with optional filters"""
        factory = get_session_factory()
        async with factory() as session:
            query = select(Job).order_by(Job.submitted_at.desc()).limit(limit)
            
            if session_id:
                query = query.where(Job.session_id == uuid.UUID(session_id))
            if status:
                query = query.where(Job.status == JobStatus(status))
            
            result = await session.execute(query)
            jobs = result.scalars().all()
            
            return [{
                'job_id': str(j.id),
                'status': j.status.value,
                'submitted_at': j.submitted_at.isoformat() if j.submitted_at else None,
                'execution_time_ms': j.execution_time_ms,
                'has_error': j.error_message is not None,
            } for j in jobs]
    
    async def cleanup_old_jobs(self, hours: int = None):
        """Delete jobs older than specified hours"""
        hours = hours or settings.JOB_CLEANUP_HOURS
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                delete(Job).where(Job.submitted_at < cutoff)
            )
            await session.commit()
            
            count = result.rowcount
            if count > 0:
                logger.info(f"Cleaned up {count} old jobs (older than {hours}h)")
            return count
    
    @property
    def active_job_count(self) -> int:
        """Number of currently running jobs"""
        return len([t for t in self._running_jobs.values() if not t.done()])


# Singleton
job_manager = JobManager()
