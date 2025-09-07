"""
Azure Service Bus Client - Infrastructure service for message queue operations.

This service provides clean abstractions for Azure Service Bus operations,
including message publishing, subscription management, and error handling.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import json
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Optional Azure Service Bus imports (graceful degradation if not available)
try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage
    from azure.servicebus.exceptions import ServiceBusError
    AZURE_AVAILABLE = True
except ImportError:
    logger.warning("Azure Service Bus SDK not available - message queue features disabled")
    AZURE_AVAILABLE = False
    ServiceBusClient = None
    ServiceBusMessage = None
    ServiceBusError = Exception


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class QueueMessage:
    """Structured message for queue operations."""
    id: Optional[str]
    body: Dict[str, Any]
    priority: MessagePriority
    correlation_id: Optional[str] = None
    time_to_live: Optional[timedelta] = None
    scheduled_enqueue_time: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class QueueStats:
    """Queue statistics and metrics."""
    name: str
    active_message_count: int
    dead_letter_count: int
    scheduled_message_count: int
    size_in_bytes: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AzureServiceBusClient:
    """
    Infrastructure service for Azure Service Bus integration.
    
    Provides:
    - Message publishing with priorities and scheduling
    - Queue monitoring and statistics
    - Error handling and retry logic
    - Connection management and pooling
    - Message serialization/deserialization
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        default_queue: Optional[str] = None
    ):
        self.connection_string = connection_string or os.getenv("AZURE_SERVICE_BUS_CONNECTION_STRING")
        self.default_queue = default_queue or os.getenv("AZURE_SERVICE_BUS_QUEUE_NAME")
        self._client: Optional[ServiceBusClient] = None
        self._connected = False
        
        if not AZURE_AVAILABLE:
            logger.warning("Azure Service Bus not available - client will operate in mock mode")
        
        if not self.connection_string:
            logger.warning("Azure Service Bus connection string not configured")
    
    def connect(self) -> bool:
        """
        Establish connection to Azure Service Bus.
        
        Returns:
            True if connection successful
        """
        
        if not AZURE_AVAILABLE:
            logger.info("Azure Service Bus not available - using mock connection")
            self._connected = True
            return True
        
        if not self.connection_string:
            logger.error("Cannot connect: Azure Service Bus connection string not configured")
            return False
        
        try:
            self._client = ServiceBusClient.from_connection_string(self.connection_string)
            self._connected = True
            logger.info("Connected to Azure Service Bus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure Service Bus: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Close Azure Service Bus connection."""
        
        if self._client:
            try:
                self._client.close()
                logger.info("Disconnected from Azure Service Bus")
            except Exception as e:
                logger.warning(f"Error during Service Bus disconnect: {e}")
        
        self._client = None
        self._connected = False
    
    def send_message(
        self,
        message: QueueMessage,
        queue_name: Optional[str] = None
    ) -> bool:
        """
        Send a single message to the queue.
        
        Args:
            message: Message to send
            queue_name: Target queue (uses default if not specified)
            
        Returns:
            True if message sent successfully
        """
        
        target_queue = queue_name or self.default_queue
        
        if not target_queue:
            logger.error("Cannot send message: no queue specified")
            return False
        
        if not self._ensure_connected():
            return False
        
        try:
            # Handle mock mode
            if not AZURE_AVAILABLE or not self._client:
                logger.info(f"Mock: Would send message to queue '{target_queue}'")
                return True
            
            # Create Azure Service Bus message
            sb_message = ServiceBusMessage(
                body=json.dumps(message.body),
                correlation_id=message.correlation_id,
                time_to_live=message.time_to_live
            )
            
            # Add priority as custom property
            sb_message.application_properties = {
                "priority": message.priority.value,
                **(message.metadata or {})
            }
            
            # Schedule message if specified
            if message.scheduled_enqueue_time:
                sb_message.scheduled_enqueue_time_utc = message.scheduled_enqueue_time
            
            # Send message
            with self._client.get_queue_sender(queue_name=target_queue) as sender:
                sender.send_messages(sb_message)
            
            logger.debug(f"Sent message to queue '{target_queue}' with correlation_id: {message.correlation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to queue '{target_queue}': {e}")
            return False
    
    def send_batch_messages(
        self,
        messages: List[QueueMessage],
        queue_name: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Send multiple messages in batches for better performance.
        
        Args:
            messages: List of messages to send
            queue_name: Target queue
            batch_size: Messages per batch
            
        Returns:
            Batch sending results
        """
        
        if not messages:
            return {"success": True, "total_messages": 0, "batches_sent": 0}
        
        target_queue = queue_name or self.default_queue
        
        if not target_queue:
            logger.error("Cannot send batch: no queue specified")
            return {"success": False, "error": "No queue specified", "total_messages": len(messages)}
        
        if not self._ensure_connected():
            return {"success": False, "error": "Connection failed", "total_messages": len(messages)}
        
        successful_messages = 0
        batches_sent = 0
        
        try:
            # Process messages in batches
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i + batch_size]
                
                if not AZURE_AVAILABLE or not self._client:
                    # Mock mode
                    logger.info(f"Mock: Would send batch of {len(batch)} messages to '{target_queue}'")
                    successful_messages += len(batch)
                    batches_sent += 1
                    continue
                
                # Create Service Bus messages
                sb_messages = []
                for msg in batch:
                    sb_message = ServiceBusMessage(
                        body=json.dumps(msg.body),
                        correlation_id=msg.correlation_id,
                        time_to_live=msg.time_to_live
                    )
                    
                    sb_message.application_properties = {
                        "priority": msg.priority.value,
                        **(msg.metadata or {})
                    }
                    
                    if msg.scheduled_enqueue_time:
                        sb_message.scheduled_enqueue_time_utc = msg.scheduled_enqueue_time
                    
                    sb_messages.append(sb_message)
                
                # Send batch
                with self._client.get_queue_sender(queue_name=target_queue) as sender:
                    sender.send_messages(sb_messages)
                
                successful_messages += len(batch)
                batches_sent += 1
                
                logger.debug(f"Sent batch {batches_sent} with {len(batch)} messages to '{target_queue}'")
            
            return {
                "success": True,
                "total_messages": len(messages),
                "successful_messages": successful_messages,
                "batches_sent": batches_sent,
                "queue": target_queue
            }
            
        except Exception as e:
            logger.error(f"Batch send failed for queue '{target_queue}': {e}")
            return {
                "success": False,
                "error": str(e),
                "total_messages": len(messages),
                "successful_messages": successful_messages,
                "batches_sent": batches_sent
            }
    
    def get_queue_stats(self, queue_name: Optional[str] = None) -> Optional[QueueStats]:
        """
        Get statistics for a queue.
        
        Args:
            queue_name: Queue to inspect (uses default if not specified)
            
        Returns:
            Queue statistics or None if unavailable
        """
        
        target_queue = queue_name or self.default_queue
        
        if not target_queue:
            logger.error("Cannot get stats: no queue specified")
            return None
        
        if not self._ensure_connected():
            return None
        
        try:
            # Mock mode
            if not AZURE_AVAILABLE or not self._client:
                return QueueStats(
                    name=target_queue,
                    active_message_count=0,
                    dead_letter_count=0,
                    scheduled_message_count=0,
                    size_in_bytes=0,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            
            # Get queue properties
            with self._client.get_queue_receiver(queue_name=target_queue) as receiver:
                queue_properties = receiver.get_queue_properties()
                
                return QueueStats(
                    name=target_queue,
                    active_message_count=queue_properties.active_message_count,
                    dead_letter_count=queue_properties.dead_letter_message_count,
                    scheduled_message_count=queue_properties.scheduled_message_count,
                    size_in_bytes=queue_properties.size_in_bytes,
                    created_at=queue_properties.created_at_utc,
                    updated_at=queue_properties.updated_at_utc
                )
                
        except Exception as e:
            logger.error(f"Failed to get stats for queue '{target_queue}': {e}")
            return None
    
    def create_cvar_calculation_message(
        self,
        symbol: str,
        alpha_level: int,
        force_recalculate: bool = True,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None
    ) -> QueueMessage:
        """
        Create a CVaR calculation message for queue processing.
        
        Args:
            symbol: Financial symbol to process
            alpha_level: Alpha level (50, 95, 99)
            force_recalculate: Force recalculation flag
            priority: Message priority
            correlation_id: Optional correlation identifier
            
        Returns:
            Structured queue message
        """
        
        import time
        
        message_body = {
            "type": "cvar_calculation",
            "symbol": symbol,
            "alpha_level": alpha_level,
            "force_recalculate": force_recalculate,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "orchestrated_batch"
        }
        
        return QueueMessage(
            id=None,  # Will be assigned by Azure
            body=message_body,
            priority=priority,
            correlation_id=correlation_id or f"{symbol}-{int(time.time())}",
            time_to_live=timedelta(hours=24),  # Message expires after 24 hours
            metadata={
                "symbol": symbol,
                "alpha": str(alpha_level)
            }
        )
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status and configuration.
        
        Returns:
            Connection status information
        """
        
        return {
            "connected": self._connected,
            "azure_sdk_available": AZURE_AVAILABLE,
            "connection_string_configured": bool(self.connection_string),
            "default_queue_configured": bool(self.default_queue),
            "client_initialized": bool(self._client),
            "mode": "production" if AZURE_AVAILABLE and self.connection_string else "mock"
        }
    
    # Private helper methods
    
    def _ensure_connected(self) -> bool:
        """Ensure connection is established, attempting to connect if needed."""
        
        if self._connected:
            return True
        
        return self.connect()
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connect():
            raise Exception("Failed to connect to Azure Service Bus")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
