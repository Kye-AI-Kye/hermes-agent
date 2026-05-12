"""Tests for cron-context behavior in HonchoMemoryProvider (Port #4053).

Pins down the three states the provider must distinguish:

  flush context  → fully inactive (_cron_skipped=True, no tools, no reads, no writes)
  cron context   → tools + reads ON, writes OFF (peer pollution guard)
  normal context → everything ON

Without these tests, future refactors can silently re-enable cron writes
(polluting the user's peer representation with synthetic LLM exchanges
from scheduled jobs) or silently disable cron tools (breaking
daily-learning-review and similar jobs that need honcho_*).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from plugins.memory.honcho import HonchoMemoryProvider


def _stub_unconfigured_honcho():
    """Patch HonchoClientConfig so initialize() returns early after the guards.

    We're testing the guards themselves — we don't need a real Honcho
    client. `enabled=False` triggers the "not configured — plugin
    inactive" early return after the cron-write check has run.
    """
    cfg = MagicMock()
    cfg.enabled = False
    cfg.api_key = ""
    cfg.base_url = ""
    return patch(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        return_value=cfg,
    )


class TestInitializeWriteGuard:
    """initialize() sets _cron_skipped / _writes_enabled correctly per context."""

    def test_default_context_keeps_writes_enabled(self):
        provider = HonchoMemoryProvider()
        with _stub_unconfigured_honcho():
            provider.initialize("test-session")
        assert provider._cron_skipped is False
        assert provider._writes_enabled is True

    def test_cli_platform_keeps_writes_enabled(self):
        provider = HonchoMemoryProvider()
        with _stub_unconfigured_honcho():
            provider.initialize("test-session", platform="cli", agent_context="")
        assert provider._writes_enabled is True

    def test_flush_context_disables_everything(self):
        provider = HonchoMemoryProvider()
        provider.initialize("test-session", agent_context="flush")
        assert provider._cron_skipped is True
        # _writes_enabled stays at its default — flush returns before the
        # cron-write check, and _cron_skipped already shadows everything.
        assert provider._writes_enabled is True

    def test_cron_agent_context_disables_writes_only(self):
        provider = HonchoMemoryProvider()
        with _stub_unconfigured_honcho():
            provider.initialize("test-session", agent_context="cron")
        assert provider._cron_skipped is False
        assert provider._writes_enabled is False

    def test_cron_platform_disables_writes_only(self):
        provider = HonchoMemoryProvider()
        with _stub_unconfigured_honcho():
            provider.initialize("test-session", platform="cron")
        assert provider._cron_skipped is False
        assert provider._writes_enabled is False

    def test_subagent_context_disables_writes_only(self):
        provider = HonchoMemoryProvider()
        with _stub_unconfigured_honcho():
            provider.initialize("test-session", agent_context="subagent")
        assert provider._cron_skipped is False
        assert provider._writes_enabled is False


class TestWriteMethodsHonorGuard:
    """sync_turn / on_memory_write / on_session_end are no-ops when writes disabled."""

    def _make(self) -> tuple[HonchoMemoryProvider, MagicMock]:
        """Return (provider, mock_manager) — keeping the mock as a local so
        tests can assert against it without going through the provider's
        Optional[HonchoSessionManager] attribute (which Pyright can't narrow)."""
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        provider._manager = mock_manager
        provider._session_key = "agent:main:test"
        provider._session_initialized = True
        provider._cron_skipped = False
        provider._writes_enabled = False
        cfg = MagicMock()
        cfg.message_max_chars = 25000
        provider._config = cfg
        return provider, mock_manager

    def test_sync_turn_skipped_when_writes_disabled(self):
        provider, mock_manager = self._make()
        provider.sync_turn("hello", "world")
        # No background thread should fire — manager untouched.
        mock_manager.get_or_create.assert_not_called()
        mock_manager._flush_session.assert_not_called()
        assert provider._sync_thread is None

    def test_on_memory_write_skipped_when_writes_disabled(self):
        provider, mock_manager = self._make()
        provider.on_memory_write("add", "user", "I prefer python")
        mock_manager.create_conclusion.assert_not_called()

    def test_on_session_end_skipped_when_writes_disabled(self):
        provider, mock_manager = self._make()
        provider.on_session_end([{"role": "user", "content": "hi"}])
        # on_session_end's only side effect when active is flush_all — must not fire.
        mock_manager.flush_all.assert_not_called()


class TestToolsExposedInCron:
    """The intent of #4053: cron context still exposes the 5 honcho_* tools."""

    def test_get_tool_schemas_returns_all_tools_in_cron(self):
        provider = HonchoMemoryProvider()
        provider._cron_skipped = False  # cron context, not flush
        provider._writes_enabled = False
        provider._recall_mode = "hybrid"  # default; tools visible
        schemas = provider.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {
            "honcho_profile",
            "honcho_search",
            "honcho_reasoning",
            "honcho_context",
            "honcho_conclude",
        }

    def test_get_tool_schemas_empty_in_flush(self):
        provider = HonchoMemoryProvider()
        provider._cron_skipped = True  # flush context
        provider._recall_mode = "hybrid"
        assert provider.get_tool_schemas() == []
