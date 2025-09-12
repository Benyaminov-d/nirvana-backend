#!/usr/bin/env python3
"""Check anchors in the database."""

from core.db import get_db_session
from core.models import CompassAnchor

def main():
    """Check anchors in the database."""
    sess = get_db_session()
    if not sess:
        print("Failed to get database session")
        return
        
    try:
        anchors = sess.query(CompassAnchor).all()
        print(f"Total anchors: {len(anchors)}")
        
        for a in anchors:
            print(f"- {a.category} (version: {a.version})")
    finally:
        sess.close()

if __name__ == "__main__":
    main()
