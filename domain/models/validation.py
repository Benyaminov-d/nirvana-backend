"""
Validation domain models.

Models for data validation results and rules
with business logic independent of implementation details.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from uuid import UUID, uuid4
from enum import Enum

from domain.value_objects.symbol import Symbol
from domain.value_objects.percentage import Percentage
from shared.exceptions import DataValidationError


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation categories."""
    DATA_QUALITY = "data_quality"
    HISTORY_SUFFICIENCY = "history_sufficiency"
    LIQUIDITY = "liquidity"
    STRUCTURAL = "structural"
    ANOMALY = "anomaly"


@dataclass
class ValidationRule:
    """Business rule for data validation."""
    
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    description: str
    threshold: Optional[float] = None
    
    def __post_init__(self):
        """Validate rule definition."""
        if not self.name.strip():
            raise DataValidationError("Validation rule name cannot be empty")


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    
    rule: ValidationRule
    message: str
    value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_blocking(self) -> bool:
        """Check if issue blocks further processing."""
        return self.rule.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL}


@dataclass
class ValidationResult:
    """Complete validation result for an instrument."""
    
    id: UUID
    symbol: Symbol
    validation_date: date
    is_valid: bool
    
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate result."""
        if self.id is None:
            object.__setattr__(self, 'id', uuid4())
    
    @property
    def has_errors(self) -> bool:
        """Check if result has error-level issues."""
        return any(issue.is_blocking for issue in self.issues)
    
    @property
    def severity_counts(self) -> Dict[ValidationSeverity, int]:
        """Get count of issues by severity."""
        counts = {severity: 0 for severity in ValidationSeverity}
        for issue in self.issues:
            counts[issue.rule.severity] += 1
        return counts
    
    def add_issue(self, rule: ValidationRule, message: str, value: float = None, **context) -> None:
        """Add validation issue."""
        issue = ValidationIssue(
            rule=rule,
            message=message,
            value=value,
            context=context
        )
        self.issues.append(issue)
        
        # Update overall validity
        if issue.is_blocking:
            self.is_valid = False
