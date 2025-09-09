"""Facade re-export for ORM models.

All real model definitions live under backend/core/db_models/.
"""

# flake8: noqa

from core.db import Base

from core.db_models.symbols import Symbols, InstrumentAlias
from core.db_models.exchange import Exchange
from core.db_models.snapshot import (
    PriceLast,
    CvarSnapshot,
    AnnualCvarViolation,
    AnomalyReport,
    InsufficientDataEvent,
)
from core.db_models.catalogue import (
    ParameterBucket,
    CatalogueSnapshot,
    CatalogueSnapshotEntry,
    CompassAnchor,
)
from core.db_models.sur import (
    SurModel,
    SurModelConstituent,
)
from core.db_models.session import (
    Session,
    SessionQuota,
    SessionPreviewCache,
)
from core.db_models.portfolio import (
    PortfolioRequest,
    PortfolioPosition,
    PortfolioMetrics,
)
from core.db_models.user_digest import (
    User,
    AuthAttempt,
    DigestSubscription,
    DigestIssue,
    DigestDelivery,
)
from core.db_models.logs import TickerLookupLog, ProximitySearchLog
from core.db_models.validation import ValidationFlags
from core.db_models.compass_anchor_versions import CompassAnchorVersions
from core.db_models.compass_inputs import CompassInputs
from core.db_models.compass_metadata import RiskModels, MuPolicies

__all__ = [
    # Base
    "Base",
    # Instruments
    "Symbols", # Backward compatibility
    "InstrumentAlias",
    # Exchanges
    "Exchange",
    # Snapshots
    "PriceLast",
    "CvarSnapshot",
    "AnnualCvarViolation",
    "AnomalyReport",
    "InsufficientDataEvent",
    # Catalogue
    "ParameterBucket",
    "CatalogueSnapshot",
    "CatalogueSnapshotEntry",
    "CompassAnchor",
    # Sur
    "SurModel",
    "SurModelConstituent",
    # Sessions
    "Session",
    "SessionQuota",
    "SessionPreviewCache",
    # Portfolio
    "PortfolioRequest",
    "PortfolioPosition",
    "PortfolioMetrics",
    # Digest
    "User",
    "AuthAttempt",
    "DigestSubscription",
    "DigestIssue",
    "DigestDelivery",
    # Logs
    "TickerLookupLog",
    "ProximitySearchLog",
    # Validation  
    "ValidationFlags",
    # Compass Score v2.0 - Parameters Only Architecture
    "CompassAnchorVersions",
    "CompassInputs", 
    "RiskModels",
    "MuPolicies",
]


