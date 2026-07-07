"""Unit tests for GDAL environment configuration."""

from __future__ import annotations

import logging
import os

import pytest

from tessera_embeddings.config.environment import configure_gdal_environment

# Every var configure_gdal_environment sets, with its expected default value.
EXPECTED_GDAL_VARS = {
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1",
    "GDAL_HTTP_LOW_SPEED_TIME": "60",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_CACHEMAX": "1024",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}


@pytest.fixture
def restore_pkg_logger():
    """Snapshot the ``tessera_embeddings`` logger's level/handlers and restore them after the test.

    ``configure_gdal_environment`` mutates the process-global
    ``tessera_embeddings`` logger, so without this the logging tests would leak
    state into each other.
    """
    pkg_logger = logging.getLogger("tessera_embeddings")
    saved_level = pkg_logger.level
    saved_handlers = pkg_logger.handlers[:]
    try:
        yield pkg_logger
    finally:
        pkg_logger.setLevel(saved_level)
        pkg_logger.handlers[:] = saved_handlers


class TestGdalEnvironment:
    """Tests for configure_gdal_environment's GDAL variable setup."""

    def test_sets_all_expected_vars(self, monkeypatch):
        """Configuration sets every GDAL var to its documented default."""
        for var in EXPECTED_GDAL_VARS:
            monkeypatch.delenv(var, raising=False)

        configure_gdal_environment()

        for var, expected in EXPECTED_GDAL_VARS.items():
            assert os.environ[var] == expected, var

    @pytest.mark.parametrize(
        "var, override",
        [
            ("GDAL_NUM_THREADS", "4"),
            ("GDAL_HTTP_MAX_RETRY", "99"),
            ("GDAL_CACHEMAX", "256"),
        ],
    )
    def test_respects_existing_values(self, monkeypatch, var, override):
        """Pre-set vars are not clobbered (setdefault semantics)."""
        monkeypatch.setenv(var, override)
        configure_gdal_environment()
        assert os.environ[var] == override


class TestPkgLoggerSetup:
    """Tests for the logging branch of configure_gdal_environment."""

    def test_defaults_to_info_level(self, monkeypatch, restore_pkg_logger):
        """With SRC_LOG_LEVEL unset, the package logger is configured at INFO."""
        monkeypatch.delenv("SRC_LOG_LEVEL", raising=False)
        configure_gdal_environment()
        assert restore_pkg_logger.level == logging.INFO

    @pytest.mark.parametrize(
        "level_str, expected",
        [
            ("WARNING", logging.WARNING),
            ("error", logging.ERROR),  # lower-case is upper-cased before lookup
            ("DEBUG", logging.DEBUG),
        ],
    )
    def test_respects_src_log_level(self, monkeypatch, restore_pkg_logger, level_str, expected):
        """SRC_LOG_LEVEL is parsed (case-insensitively) into the logger level."""
        monkeypatch.setenv("SRC_LOG_LEVEL", level_str)
        configure_gdal_environment()
        assert restore_pkg_logger.level == expected

    def test_invalid_level_falls_back_to_info(self, monkeypatch, restore_pkg_logger):
        """An unrecognized level name falls back to INFO rather than raising."""
        monkeypatch.setenv("SRC_LOG_LEVEL", "NOTALEVEL")
        configure_gdal_environment()
        assert restore_pkg_logger.level == logging.INFO

    def test_attaches_a_handler(self, monkeypatch, restore_pkg_logger):
        """A StreamHandler is attached when the package logger has none."""
        monkeypatch.delenv("SRC_LOG_LEVEL", raising=False)
        restore_pkg_logger.handlers[:] = []
        configure_gdal_environment()
        assert len(restore_pkg_logger.handlers) == 1
        assert isinstance(restore_pkg_logger.handlers[0], logging.StreamHandler)

    def test_does_not_add_duplicate_handlers(self, monkeypatch, restore_pkg_logger):
        """Repeated calls must not stack duplicate handlers (the `if not handlers` guard)."""
        monkeypatch.delenv("SRC_LOG_LEVEL", raising=False)
        restore_pkg_logger.handlers[:] = []
        configure_gdal_environment()
        configure_gdal_environment()
        configure_gdal_environment()
        assert len(restore_pkg_logger.handlers) == 1
