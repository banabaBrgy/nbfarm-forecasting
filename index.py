"""Vercel Flask entry (zero-config). Local still uses: python -m flask --app api.index run"""

from api.index import app

__all__ = ["app"]
