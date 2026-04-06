"""Shared fixtures for OAT Proxy tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def anyio_backend():
    return "asyncio"
