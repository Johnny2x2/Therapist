from __future__ import annotations

from unittest.mock import MagicMock

from src.safety import SafetyMonitor


def test_keyword_escalates_when_classifier_clean(tmp_config):
    client = MagicMock()
    client.chat.return_value = {"message": {"content": '{"flagged": false, "risk_level": "low", "reason": "ok"}'}}
    monitor = SafetyMonitor(tmp_config, client)
    result = monitor.assess("I am going to kill myself tonight")
    assert result.flagged is True
    assert result.source.startswith("keyword")


def test_invalid_json_returns_unknown(tmp_config):
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "not json"}}
    monitor = SafetyMonitor(tmp_config, client)
    result = monitor.assess("benign message")
    assert result.flagged is False
    assert result.risk_level == "unknown"
    assert result.source == "fallback"


def test_classifier_flag_passes_through(tmp_config):
    client = MagicMock()
    client.chat.return_value = {"message": {"content": '{"flagged": true, "risk_level": "high", "reason": "stated intent"}'}}
    monitor = SafetyMonitor(tmp_config, client)
    result = monitor.assess("benign-looking text")
    assert result.flagged is True
    assert result.risk_level == "high"
