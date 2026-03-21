from __future__ import annotations

"""Compatibility entrypoint for serving the separated frontend and backend app."""

from ..backend.api import create_app

app = create_app()
