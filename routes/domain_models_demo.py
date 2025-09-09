"""
Domain Models Demo Routes.

Demonstrates the usage of domain models and value objects
with type-safe business logic independent of database schemas.
"""

from fastapi import APIRouter, HTTPException, Depends, Query  # type: ignore
from typing import Dict, List, Any, Optional
import logging
from datetime import date, datetime
from decimal import Decimal

from utils.auth import require_pub_or_basic as _require_pub_or_basic

# Import domain models
from domain.models.financial_instrument import FinancialInstrument
from domain.models.risk_assessment import RiskAssessment, CVaRCalculation
from domain.models.market_data import PriceHistory, PricePoint, MarketQuote
from domain.models.portfolio import Portfolio, Position
from domain.models.user import User
from domain.models.validation import ValidationResult, ValidationRule, ValidationIssue

# Import value objects
from domain.value_objects.symbol import Symbol, ISIN
from domain.value_objects.money import Money, Currency
from domain.value_objects.percentage import Percentage
from domain.value_objects.risk_metrics import CVaRValue, CVaRTriple, AlphaLevel
from domain.value_objects.date_range import DateRange
from domain.value_objects.instrument import InstrumentClassification, InstrumentType, Country, Exchange

# Import shared utilities
from shared.utilities import create_success_response, Timer
from shared.exceptions import DataValidationError, BusinessLogicError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/domain-models", tags=["domain-models"])


@router.get("/value-objects-demo")
def demo_value_objects(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate value objects with type safety and validation.
    
    Shows how value objects encapsulate domain concepts with proper validation.
    """
    
    with Timer("Value Objects Demo") as timer:
        examples = {}
        
        # Money examples
        try:
            usd_money = Money.usd(1250.75)
            eur_money = Money.eur(1000.50)
            
            examples["money"] = {
                "usd_amount": {
                    "creation": "Money.usd(1250.75)",
                    "value": str(usd_money),
                    "formatted": usd_money.format_currency(),
                    "operations": {
                        "addition": str(usd_money + Money.usd(250)),
                        "multiplication": str(usd_money * 1.1),
                        "negative": str(-usd_money)
                    }
                },
                "currency_safety": {
                    "description": "Different currencies cannot be mixed",
                    "example": "Money.usd(100) + Money.eur(100) raises DataValidationError"
                }
            }
        except Exception as e:
            examples["money"] = {"error": str(e)}
        
        # Percentage examples
        try:
            pct1 = Percentage.from_percent(15.67)  # 15.67%
            pct2 = Percentage.from_decimal(0.25)   # 25%
            
            examples["percentage"] = {
                "creation_methods": {
                    "from_percent": f"{pct1} (from 15.67)",
                    "from_decimal": f"{pct2} (from 0.25)",
                    "from_basis_points": str(Percentage.from_basis_points(150))  # 1.5%
                },
                "formatting": {
                    "default": str(pct1),
                    "precision_4": pct1.format(4),
                    "basis_points": pct1.format_basis_points()
                },
                "calculations": {
                    "addition": str(pct1 + pct2),
                    "multiplication": str(pct1 * 2),
                    "apply_to_value": f"{pct1.of(1000)} (15.67% of 1000)"
                }
            }
        except Exception as e:
            examples["percentage"] = {"error": str(e)}
        
        # Symbol examples
        try:
            symbol1 = Symbol("AAPL")
            symbol2 = Symbol("AAPL.US")
            
            examples["symbol"] = {
                "basic_symbol": {
                    "value": symbol1.value,
                    "base_symbol": symbol1.base_symbol,
                    "has_suffix": symbol1.has_exchange_suffix
                },
                "symbol_with_suffix": {
                    "value": symbol2.value,
                    "base_symbol": symbol2.base_symbol,
                    "exchange_suffix": symbol2.exchange_suffix,
                    "has_suffix": symbol2.has_exchange_suffix
                },
                "validation": {
                    "description": "Symbols are validated and normalized",
                    "example": "Symbol('  aapl  ') becomes 'AAPL'"
                },
                "operations": {
                    "with_suffix": symbol1.with_suffix("NASDAQ").value,
                    "without_suffix": symbol2.without_suffix().value,
                    "similarity": symbol1.is_similar_to(symbol2)
                }
            }
        except Exception as e:
            examples["symbol"] = {"error": str(e)}
        
        # CVaR Value examples
        try:
            cvar_99 = CVaRValue.from_percent(-25.5, 99, "nig")
            cvar_95 = CVaRValue.from_decimal(-0.18, 95, "ghst")
            
            examples["cvar_value"] = {
                "cvar_99_percent": {
                    "value": cvar_99.format(show_method=True),
                    "is_loss": cvar_99.is_loss,
                    "loss_magnitude": str(cvar_99.loss_magnitude),
                    "alpha_level": cvar_99.alpha.value
                },
                "cvar_95_percent": {
                    "value": cvar_95.format(show_method=True),
                    "decimal_form": float(cvar_95.to_decimal()),
                    "percentage_form": float(cvar_95.to_percent())
                },
                "comparison": {
                    "worse_cvar": cvar_99.is_worse_than(cvar_95),
                    "description": "More negative CVaR is worse (higher loss)"
                }
            }
        except Exception as e:
            examples["cvar_value"] = {"error": str(e)}
        
        # Date Range examples
        try:
            range_5y = DateRange.last_n_years(5)
            range_custom = DateRange("2020-01-01", "2023-12-31")
            
            examples["date_range"] = {
                "last_5_years": {
                    "range": str(range_5y),
                    "days": range_5y.days,
                    "years": round(range_5y.years, 2),
                    "trading_days": range_5y.trading_days
                },
                "custom_range": {
                    "range": str(range_custom),
                    "contains_date": range_custom.contains("2022-06-15"),
                    "overlaps_with_5y": range_custom.overlaps(range_5y)
                }
            }
        except Exception as e:
            examples["date_range"] = {"error": str(e)}
        
        # Instrument Classification examples
        try:
            us_stock = InstrumentClassification.from_strings("Common Stock", "US", "NYSE")
            eur_etf = InstrumentClassification.from_strings("ETF", "DE", "XETRA")
            
            examples["instrument_classification"] = {
                "us_stock": {
                    "description": str(us_stock),
                    "supports_cvar": us_stock.supports_cvar,
                    "is_equity_like": us_stock.is_equity_like,
                    "currency": us_stock.currency,
                    "eodhd_suffix": us_stock.get_eodhd_suffix()
                },
                "european_etf": {
                    "description": str(eur_etf),
                    "market_tier": eur_etf.market_tier,
                    "is_international": eur_etf.is_international
                }
            }
        except Exception as e:
            examples["instrument_classification"] = {"error": str(e)}
    
    return create_success_response({
        "examples": examples,
        "execution_time_ms": round(timer.elapsed * 1000, 2),
        "value_objects_benefits": [
            "Type safety with automatic validation",
            "Immutable objects prevent accidental modification", 
            "Rich behavior methods (formatting, calculations)",
            "Clear separation of domain concepts",
            "Consistent handling across application",
            "Better IDE support and IntelliSense"
        ],
        "validation_examples": [
            "Money prevents mixing currencies",
            "Symbol normalizes and validates format",
            "Percentage handles decimal/percent conversion safely",
            "CVaRValue ensures proper alpha level validation",
            "DateRange validates date consistency"
        ]
    }, message="Value objects demo - type-safe domain primitives")


@router.get("/financial-instrument-demo")
def demo_financial_instrument(
    symbol: str = Query("AAPL", description="Symbol to create instrument for"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate FinancialInstrument domain model.
    
    Shows how domain models encapsulate business logic and rules.
    """
    
    with Timer("Financial Instrument Demo") as timer:
        try:
            # Create financial instrument
            instrument = FinancialInstrument.create(
                symbol=symbol,
                name="Apple Inc.",
                instrument_type="Common Stock",
                country_code="US",
                exchange_code="NASDAQ",
                currency="USD"
            )
            
            # Add alternative symbol
            instrument.add_alternative_symbol(f"{symbol}.US")
            
            # Set market cap
            instrument.set_market_cap(2_800_000_000_000)  # $2.8T
            
            # Update data quality
            instrument.update_data_quality(0.95, has_sufficient_history=True)
            
            # Business logic examples
            business_logic = {
                "risk_analysis_eligibility": {
                    "supports_cvar": instrument.supports_cvar_analysis,
                    "eligible_for_portfolio": instrument.is_eligible_for_analysis("portfolio"),
                    "eligible_for_comparison": instrument.is_eligible_for_analysis("comparison")
                },
                "categorization": {
                    "is_equity": instrument.is_equity,
                    "is_fund": instrument.is_fund,
                    "is_derivative": instrument.is_derivative,
                    "risk_category": instrument.get_risk_category()
                },
                "market_info": {
                    "primary_market": instrument.primary_market,
                    "full_symbol": instrument.full_symbol,
                    "display_name": instrument.display_name
                }
            }
            
            # Demonstrate validation
            validation_demo = {}
            try:
                # This should fail - empty name
                invalid_instrument = FinancialInstrument.create(
                    symbol="TEST",
                    name="",  # Empty name should fail
                    instrument_type="Stock"
                )
            except DataValidationError as e:
                validation_demo["empty_name_validation"] = {
                    "error": str(e),
                    "description": "Domain model validates business rules"
                }
            
            try:
                # This should fail - invalid symbol
                invalid_symbol = FinancialInstrument.create(
                    symbol="INVALID@SYMBOL",  # Invalid characters
                    name="Invalid Test",
                    instrument_type="Stock"
                )
            except DataValidationError as e:
                validation_demo["invalid_symbol_validation"] = {
                    "error": str(e),
                    "description": "Symbol value object validates format"
                }
            
            return create_success_response({
                "instrument": instrument.to_dict(),
                "business_logic": business_logic,
                "validation_examples": validation_demo,
                "execution_time_ms": round(timer.elapsed * 1000, 2),
                "domain_model_benefits": [
                    "Encapsulates business rules and validation",
                    "Type-safe operations with value objects",
                    "Rich behavior methods for business logic",
                    "Independent of database schema",
                    "Testable business logic in isolation",
                    "Clear separation of concerns"
                ]
            }, message=f"Financial instrument created: {instrument}")
            
        except Exception as e:
            logger.error(f"Financial instrument demo failed: {e}")
            raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/risk-assessment-demo")
def demo_risk_assessment(
    symbol: str = Query("MSFT", description="Symbol for risk assessment"),
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate RiskAssessment domain model with CVaR calculations.
    
    Shows complex domain logic for risk analysis.
    """
    
    with Timer("Risk Assessment Demo") as timer:
        try:
            # Create risk assessment
            assessment = RiskAssessment.create(symbol=symbol)
            
            # Add CVaR calculations for different alpha levels
            # 99% CVaR
            cvar_nig_99 = CVaRValue.from_percent(-28.5, 99, "nig")
            cvar_ghst_99 = CVaRValue.from_percent(-26.2, 99, "ghst") 
            cvar_evar_99 = CVaRValue.from_percent(-30.1, 99, "evar")
            
            assessment.add_cvar_calculation(
                AlphaLevel.ALPHA_99,
                nig=cvar_nig_99,
                ghst=cvar_ghst_99,
                evar=cvar_evar_99
            )
            
            # 95% CVaR
            cvar_nig_95 = CVaRValue.from_percent(-22.8, 95, "nig")
            cvar_ghst_95 = CVaRValue.from_percent(-21.5, 95, "ghst")
            cvar_evar_95 = CVaRValue.from_percent(-24.2, 95, "evar")
            
            assessment.add_cvar_calculation(
                AlphaLevel.ALPHA_95,
                nig=cvar_nig_95,
                ghst=cvar_ghst_95,
                evar=cvar_evar_95
            )
            
            # Business logic demonstrations
            business_logic = {
                "primary_risk_metric": {
                    "primary_cvar": assessment.get_primary_cvar().to_dict() if assessment.get_primary_cvar() else None,
                    "description": "Worst CVaR at highest alpha level"
                },
                "risk_rating": {
                    "overall_rating": assessment.overall_risk_rating,
                    "confidence_level": float(assessment.confidence_level.to_decimal()),
                    "description": "Automatically calculated from CVaR values"
                },
                "analysis_by_level": {
                    "worst_at_99": assessment.get_worst_cvar_at_level(99).format() if assessment.get_worst_cvar_at_level(99) else None,
                    "worst_at_95": assessment.get_worst_cvar_at_level(95).format() if assessment.get_worst_cvar_at_level(95) else None,
                    "description": "Worst CVaR from available methods at each level"
                }
            }
            
            # Create another assessment for comparison
            other_assessment = RiskAssessment.create("AAPL")
            other_cvar = CVaRValue.from_percent(-32.5, 99, "nig")  # Higher risk
            other_assessment.add_cvar_calculation(AlphaLevel.ALPHA_99, nig=other_cvar)
            
            comparison = assessment.compare_risk_to(other_assessment)
            
            # Individual CVaR calculation demo
            individual_calc = CVaRCalculation.create(
                symbol=symbol,
                cvar_value=-0.285,  # -28.5%
                alpha=99,
                method="nig",
                data_start=date(2019, 1, 1),
                data_end=date.today(),
                observations_used=1260,
                data_completeness=0.98,
                outliers_detected=15
            )
            
            return create_success_response({
                "risk_assessment": assessment.to_dict(),
                "business_logic": business_logic,
                "risk_comparison": comparison,
                "individual_calculation": individual_calc.to_dict(),
                "calculation_quality": {
                    "is_high_quality": individual_calc.is_high_quality,
                    "quality_score": float(individual_calc.quality_score.to_decimal()),
                    "quality_factors": [
                        f"Data completeness: {individual_calc.data_completeness}",
                        f"Observations: {individual_calc.observations_used}",
                        f"Outliers: {individual_calc.outliers_detected}",
                        f"Recent: {individual_calc.is_recent()}"
                    ]
                },
                "execution_time_ms": round(timer.elapsed * 1000, 2),
                "domain_logic_benefits": [
                    "Automatic risk rating calculation",
                    "Confidence level based on data quality",
                    "Method comparison and worst-case analysis",
                    "Risk comparison between instruments",
                    "Quality assessment of calculations",
                    "Business rules encoded in domain model"
                ]
            }, message=f"Risk assessment completed: {assessment.overall_risk_rating}")
            
        except Exception as e:
            logger.error(f"Risk assessment demo failed: {e}")
            raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/portfolio-demo")
def demo_portfolio(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Demonstrate Portfolio domain model with positions and calculations.
    
    Shows composition patterns and aggregate calculations.
    """
    
    with Timer("Portfolio Demo") as timer:
        try:
            # Create portfolio
            portfolio = Portfolio(
                id=None,  # Will be auto-generated
                name="Tech Growth Portfolio",
                base_currency="USD"
            )
            
            # Create positions
            positions = [
                Position(
                    symbol=Symbol("AAPL"),
                    quantity=100,
                    average_cost=Money.usd(150.00),
                    current_value=Money.usd(185.50)
                ),
                Position(
                    symbol=Symbol("MSFT"),
                    quantity=50,
                    average_cost=Money.usd(280.00),
                    current_value=Money.usd(340.25)
                ),
                Position(
                    symbol=Symbol("GOOGL"),
                    quantity=25,
                    average_cost=Money.usd(2800.00),
                    current_value=Money.usd(3150.75)
                )
            ]
            
            # Add positions to portfolio
            for position in positions:
                portfolio.add_position(position)
            
            # Calculate position metrics
            position_analysis = []
            for position in portfolio.positions:
                analysis = {
                    "symbol": position.symbol.value,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost.format_currency(),
                    "current_value": position.current_value.format_currency() if position.current_value else "N/A",
                    "total_cost": position.total_cost.format_currency(),
                    "market_value": position.market_value.format_currency() if position.market_value else "N/A",
                    "unrealized_pnl": position.unrealized_pnl.format_currency() if position.unrealized_pnl else "N/A"
                }
                position_analysis.append(analysis)
            
            # Portfolio-level metrics
            portfolio_metrics = {
                "total_positions": portfolio.total_positions,
                "symbols": [s.value for s in portfolio.symbols],
                "base_currency": portfolio.base_currency,
                "created": portfolio.created_at.isoformat()
            }
            
            # Demonstrate validation
            validation_demo = {}
            try:
                # Try to add duplicate position
                duplicate_position = Position(
                    symbol=Symbol("AAPL"),  # Already exists
                    quantity=50,
                    average_cost=Money.usd(160.00)
                )
                portfolio.add_position(duplicate_position)
            except BusinessLogicError as e:
                validation_demo["duplicate_position"] = {
                    "error": str(e),
                    "description": "Portfolio prevents duplicate positions"
                }
            
            return create_success_response({
                "portfolio": {
                    "id": str(portfolio.id),
                    "name": portfolio.name,
                    "metrics": portfolio_metrics
                },
                "positions": position_analysis,
                "validation_examples": validation_demo,
                "execution_time_ms": round(timer.elapsed * 1000, 2),
                "domain_benefits": [
                    "Type-safe money calculations with currency consistency",
                    "Automatic P&L calculations", 
                    "Business rule validation (no duplicate positions)",
                    "Rich position analysis methods",
                    "Composition patterns for complex entities",
                    "Encapsulated portfolio logic"
                ]
            }, message=f"Portfolio created with {len(positions)} positions")
            
        except Exception as e:
            logger.error(f"Portfolio demo failed: {e}")
            raise HTTPException(500, f"Demo failed: {str(e)}")


@router.get("/architecture-comparison")
def demo_architecture_comparison(
    _auth: None = Depends(_require_pub_or_basic),
) -> Dict[str, Any]:
    """
    Compare domain model architecture with database-centric approach.
    
    Shows benefits of Clean Architecture with domain models.
    """
    
    # Database-centric approach (old way)
    old_approach = {
        "description": "Database-centric with SQLAlchemy models",
        "problems": [
            "Business logic scattered across routes and services",
            "Database schema drives domain logic",
            "Difficult to test business rules in isolation", 
            "Type safety limited to database fields",
            "Validation mixed with persistence concerns",
            "Domain concepts not clearly modeled"
        ],
        "example_code": '''
# Old way - Database model with mixed concerns
class Symbols(Base):
    symbol = Column(String)
    cvar_95 = Column(Float)  # Just a number, no validation
    
    def calculate_risk(self):  # Business logic in DB model
        if self.cvar_95 and self.cvar_95 < -0.5:
            return "HIGH"
        return "MEDIUM"
'''
    }
    
    # Domain model approach (new way)
    new_approach = {
        "description": "Domain-driven with value objects and business models",
        "benefits": [
            "Business logic centralized in domain models",
            "Type-safe value objects with validation",
            "Testable business rules independent of database",
            "Rich domain concepts clearly modeled",
            "Separation of domain and infrastructure concerns",
            "Immutable value objects prevent data corruption"
        ],
        "example_code": '''
# New way - Rich domain model with value objects
@dataclass
class FinancialInstrument:
    symbol: Symbol  # Type-safe, validated
    classification: InstrumentClassification
    
    def supports_cvar_analysis(self) -> bool:
        return (
            self.classification.supports_cvar and
            self.is_active and
            self.has_sufficient_history
        )

class CVaRValue:
    # Immutable, validated, rich behavior
    value: Percentage
    alpha: AlphaLevel
    
    def is_worse_than(self, other: CVaRValue) -> bool:
        return self.value < other.value
'''
    }
    
    # Architecture layers comparison
    layers_comparison = {
        "old_architecture": {
            "layers": ["Routes", "Services", "SQLAlchemy Models", "Database"],
            "problems": [
                "Business logic mixed across layers",
                "Database schema influences domain logic",
                "Difficult to change business rules",
                "Testing requires database"
            ]
        },
        "new_architecture": {
            "layers": [
                "Routes (Presentation)",
                "Services (Application)", 
                "Domain Models (Business Logic)",
                "Value Objects (Domain Primitives)",
                "Repositories (Infrastructure)",
                "Database Models (Persistence)"
            ],
            "benefits": [
                "Clear separation of concerns",
                "Business logic independent of infrastructure",
                "Easy to test domain logic",
                "Database schema doesn't drive business rules",
                "Value objects ensure type safety"
            ]
        }
    }
    
    # Migration strategy
    migration_benefits = {
        "immediate_benefits": [
            "Type safety with value objects",
            "Rich business behavior in models",
            "Better validation and error handling",
            "More testable code structure"
        ],
        "long_term_benefits": [
            "Easier feature development",
            "Reduced bugs through validation",
            "Better code maintainability",
            "Cleaner architecture evolution"
        ],
        "migration_approach": [
            "1. Create value objects for domain primitives",
            "2. Build domain models using value objects", 
            "3. Create mappers between domain and DB models",
            "4. Gradually migrate services to use domain models",
            "5. Keep DB models for persistence only"
        ]
    }
    
    return create_success_response({
        "architecture_comparison": {
            "old_approach": old_approach,
            "new_approach": new_approach,
            "layers_comparison": layers_comparison
        },
        "migration_benefits": migration_benefits,
        "implementation_status": {
            "value_objects_created": [
                "Money with Currency",
                "Percentage with validation",
                "Symbol with normalization",
                "CVaRValue with risk logic",
                "DateRange with business methods",
                "InstrumentClassification with market data"
            ],
            "domain_models_created": [
                "FinancialInstrument with business rules",
                "RiskAssessment with CVaR analysis",
                "Portfolio with position management",
                "CVaRCalculation with quality metrics",
                "ValidationResult with rule checking"
            ],
            "next_steps": [
                "Create mappers between domain and DB models",
                "Migrate services to use domain models",
                "Add domain events for complex workflows",
                "Implement aggregate patterns for consistency"
            ]
        }
    }, message="Domain model architecture demonstrates Clean Architecture principles")


# Export the router
__all__ = ["router"]
