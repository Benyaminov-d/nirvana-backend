#!/usr/bin/env python3
"""Check country data in the database."""

from core.db import get_db_session
from sqlalchemy import func
from core.models import Symbols

def main():
    """Check database country data."""
    session = get_db_session()
    if not session:
        print("Failed to get database session")
        return

    try:
        # Check distinct country values
        countries = [r[0] for r in session.query(func.distinct(Symbols.country)).all()]
        print(f"Distinct country values: {countries}")

        # Count symbols by country
        for country in countries:
            if country:
                count = session.query(func.count(Symbols.id)).filter(Symbols.country == country).scalar()
                print(f"Country '{country}': {count} symbols")

        # Look specifically for US, UK, CA which should be in the Harvard universe
        for expected in ["US", "UK", "CA"]:
            count = session.query(func.count(Symbols.id)).filter(Symbols.country == expected).scalar()
            print(f"Expected country '{expected}': {count} symbols")

    finally:
        session.close()

if __name__ == "__main__":
    main()
