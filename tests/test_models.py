import pytest

from common import models as M


def test_registry_entries_are_complete():
    for key in M.ORDER:
        m = M.get(key)
        assert m["hf_id"] and len(m["revision"]) == 40 and m["params_total"] >= m["params_active"] > 0
        assert m["cache"] != m["cache_shortcut"]
    assert len({M.get(k)["cache"] for k in M.ORDER}) == len(M.ORDER)          # no two models share a cache
    assert M.get("gemma26b")["cache"] == "cache"                                # Gemma's existing cache stays valid
    with pytest.raises(KeyError):
        M.get("gpt5")


def test_server_url_pattern_and_override(monkeypatch):
    monkeypatch.setenv("KWTC_MODAL_WORKSPACE", "ws")
    assert M.server_url("gemma26b") == "https://ws--kwtc-vllm-server.us-east.modal.direct"
    assert M.server_url("llama3b") == "https://ws--kwtc-vllm-serverllama3b.us-east.modal.direct"
    monkeypatch.setenv("KWTC_URL_LLAMA8B", "https://example.test/")
    assert M.server_url("llama8b") == "https://example.test"
