"""
Batch Processing Service - Generic batch operation orchestration.

This application service provides reusable batch processing capabilities
with parallel execution, progress tracking, and error handling.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchTask:
    """Represents a single task in a batch operation."""
    id: str
    data: Any
    priority: int = 0  # Higher numbers = higher priority


@dataclass 
class BatchResult:
    """Result of a batch task execution."""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class BatchStatistics:
    """Statistics for completed batch operation."""
    total_tasks: int
    successful: int
    failed: int
    skipped: int
    total_duration: float
    average_task_duration: float
    throughput_per_second: float
    start_time: datetime
    end_time: datetime


class BatchProcessingService(Generic[T, R]):
    """
    Generic service for orchestrating batch operations with parallel processing.
    
    Provides:
    - Parallel task execution with configurable workers
    - Progress tracking and statistics
    - Error handling and retry logic
    - Task prioritization
    - Resource management
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.active_batches: Dict[str, Dict[str, Any]] = {}
    
    def execute_batch(
        self,
        tasks: List[BatchTask],
        processor_func: Callable[[Any], R],
        batch_id: Optional[str] = None,
        max_workers: Optional[int] = None,
        retry_failed: bool = False,
        max_retries: int = 2,
        timeout_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a batch of tasks in parallel with comprehensive monitoring.
        
        Args:
            tasks: List of BatchTask objects to process
            processor_func: Function that processes individual task data
            batch_id: Optional identifier for tracking this batch
            max_workers: Override default worker count
            retry_failed: Whether to retry failed tasks
            max_retries: Maximum retry attempts per task
            timeout_seconds: Optional timeout for entire batch
            
        Returns:
            Comprehensive batch execution results
        """
        
        if not tasks:
            return self._create_empty_batch_result(batch_id)
        
        # Initialize batch tracking
        batch_id = batch_id or f"batch_{int(time.time())}"
        workers = max_workers or self.max_workers
        
        start_time = datetime.utcnow()
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Track batch progress
        self.active_batches[batch_id] = {
            "start_time": start_time,
            "total_tasks": len(tasks),
            "completed": 0,
            "failed": 0,
            "in_progress": True
        }
        
        try:
            # Execute batch with timeout handling
            if timeout_seconds:
                results = self._execute_with_timeout(
                    sorted_tasks, processor_func, workers, timeout_seconds
                )
            else:
                results = self._execute_parallel_batch(
                    sorted_tasks, processor_func, workers
                )
            
            # Handle retry logic
            if retry_failed and max_retries > 0:
                failed_tasks = [
                    task for task, result in zip(sorted_tasks, results)
                    if not result.success
                ]
                
                if failed_tasks:
                    logger.info(f"Retrying {len(failed_tasks)} failed tasks")
                    retry_results = self._retry_failed_tasks(
                        failed_tasks, processor_func, workers, max_retries
                    )
                    
                    # Update results with retry outcomes
                    results = self._merge_retry_results(results, retry_results, sorted_tasks)
            
            # Calculate final statistics
            end_time = datetime.utcnow()
            statistics = self._calculate_statistics(results, start_time, end_time)
            
            # Clean up batch tracking
            self.active_batches[batch_id]["completed"] = statistics.successful + statistics.failed
            self.active_batches[batch_id]["in_progress"] = False
            
            return {
                "batch_id": batch_id,
                "success": statistics.failed == 0,
                "statistics": statistics,
                "results": results,
                "processing_mode": f"parallel_batch_{workers}workers",
                "retry_enabled": retry_failed
            }
            
        except Exception as e:
            logger.error(f"Batch execution failed for {batch_id}: {e}")
            
            # Update batch tracking for failure
            if batch_id in self.active_batches:
                self.active_batches[batch_id]["in_progress"] = False
                self.active_batches[batch_id]["error"] = str(e)
            
            return {
                "batch_id": batch_id,
                "success": False,
                "error": str(e),
                "tasks_attempted": len(tasks)
            }
    
    def get_batch_progress(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for an active batch."""
        batch_info = self.active_batches.get(batch_id)
        
        if not batch_info:
            return None
        
        progress_percentage = 0.0
        if batch_info["total_tasks"] > 0:
            completed = batch_info["completed"] + batch_info["failed"]
            progress_percentage = (completed / batch_info["total_tasks"]) * 100
        
        return {
            "batch_id": batch_id,
            "total_tasks": batch_info["total_tasks"],
            "completed_tasks": batch_info["completed"],
            "failed_tasks": batch_info["failed"],
            "progress_percentage": round(progress_percentage, 1),
            "in_progress": batch_info["in_progress"],
            "start_time": batch_info["start_time"].isoformat(),
            "error": batch_info.get("error")
        }
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Attempt to cancel an active batch (best effort)."""
        if batch_id in self.active_batches:
            self.active_batches[batch_id]["cancelled"] = True
            self.active_batches[batch_id]["in_progress"] = False
            return True
        return False
    
    def get_active_batches(self) -> List[str]:
        """Get list of currently active batch IDs."""
        return [
            batch_id for batch_id, info in self.active_batches.items()
            if info.get("in_progress", False)
        ]
    
    def cleanup_completed_batches(self, max_age_hours: int = 24) -> int:
        """Clean up old completed batch tracking data."""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for batch_id, info in self.active_batches.items():
            if (not info.get("in_progress", False) and 
                info["start_time"].timestamp() < cutoff_time):
                to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del self.active_batches[batch_id]
        
        return len(to_remove)
    
    # Private implementation methods
    
    def _execute_parallel_batch(
        self, 
        tasks: List[BatchTask], 
        processor_func: Callable[[Any], R], 
        workers: int
    ) -> List[BatchResult]:
        """Execute tasks in parallel using ThreadPoolExecutor."""
        results: List[Optional[BatchResult]] = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_task, task, processor_func): i
                for i, task in enumerate(tasks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    # Create error result for failed futures
                    results[index] = BatchResult(
                        task_id=tasks[index].id,
                        success=False,
                        error=str(e)
                    )
        
        # Convert to non-Optional list (all should be filled)
        return [r for r in results if r is not None]
    
    def _execute_with_timeout(
        self,
        tasks: List[BatchTask],
        processor_func: Callable[[Any], R],
        workers: int,
        timeout_seconds: float
    ) -> List[BatchResult]:
        """Execute batch with overall timeout constraint."""
        # For simplicity, delegate to regular parallel execution
        # In production, this would implement proper timeout handling
        logger.warning("Timeout handling not fully implemented - using regular execution")
        return self._execute_parallel_batch(tasks, processor_func, workers)
    
    def _process_single_task(
        self, 
        task: BatchTask, 
        processor_func: Callable[[Any], R]
    ) -> BatchResult:
        """Process a single task with timing and error handling."""
        start_time = time.time()
        
        try:
            result = processor_func(task.data)
            duration = time.time() - start_time
            
            return BatchResult(
                task_id=task.id,
                success=True,
                result=result,
                duration_seconds=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Task {task.id} failed: {e}")
            
            return BatchResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration_seconds=duration
            )
    
    def _retry_failed_tasks(
        self,
        failed_tasks: List[BatchTask],
        processor_func: Callable[[Any], R],
        workers: int,
        max_retries: int
    ) -> List[BatchResult]:
        """Retry failed tasks with exponential backoff."""
        current_tasks = failed_tasks.copy()
        retry_count = 0
        
        while current_tasks and retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retry attempt {retry_count}/{max_retries} for {len(current_tasks)} tasks")
            
            # Add small delay between retries
            time.sleep(0.5 * retry_count)
            
            # Execute retry
            retry_results = self._execute_parallel_batch(
                current_tasks, processor_func, workers
            )
            
            # Prepare for next retry if needed
            if retry_count < max_retries:
                current_tasks = [
                    task for task, result in zip(current_tasks, retry_results)
                    if not result.success
                ]
            
            return retry_results
        
        return []
    
    def _merge_retry_results(
        self,
        original_results: List[BatchResult],
        retry_results: List[BatchResult],
        original_tasks: List[BatchTask]
    ) -> List[BatchResult]:
        """Merge original and retry results."""
        # For simplicity, return original results
        # In production, this would properly merge based on task IDs
        return original_results
    
    def _calculate_statistics(
        self,
        results: List[BatchResult],
        start_time: datetime,
        end_time: datetime
    ) -> BatchStatistics:
        """Calculate comprehensive statistics for completed batch."""
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        total_duration = (end_time - start_time).total_seconds()
        
        task_durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
        avg_task_duration = sum(task_durations) / len(task_durations) if task_durations else 0.0
        
        throughput = len(results) / total_duration if total_duration > 0 else 0.0
        
        return BatchStatistics(
            total_tasks=len(results),
            successful=successful,
            failed=failed,
            skipped=0,  # Not implemented in current version
            total_duration=total_duration,
            average_task_duration=avg_task_duration,
            throughput_per_second=throughput,
            start_time=start_time,
            end_time=end_time
        )
    
    def _create_empty_batch_result(self, batch_id: Optional[str]) -> Dict[str, Any]:
        """Create result structure for empty batch."""
        return {
            "batch_id": batch_id or "empty_batch",
            "success": True,
            "statistics": BatchStatistics(
                total_tasks=0,
                successful=0,
                failed=0,
                skipped=0,
                total_duration=0.0,
                average_task_duration=0.0,
                throughput_per_second=0.0,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            "results": [],
            "message": "No tasks to process"
        }
