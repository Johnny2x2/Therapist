from __future__ import annotations

from src.notebook import NotebookStore


def test_write_read_update_roundtrip(tmp_config, monkeypatch):
    from tests.conftest import FakeClient
    store = NotebookStore(tmp_config, client=FakeClient())

    meta = store.write_note(
        category="people",
        title="Conversation with Sam",
        body="## Observations\nSam felt heard.",
        tags=["family", "boundaries"],
    )
    assert meta["category"] == "people"
    assert meta["id"]

    fetched = store.read_note(meta["id"])
    assert fetched is not None
    assert "Sam felt heard." in fetched.body
    assert set(fetched.meta.tags) == {"family", "boundaries"}

    listing = store.list_notes()
    assert any(m["id"] == meta["id"] for m in listing)

    updated = store.update_note(meta["id"], append="Followed up on Sunday.", set_pinned=True)
    assert updated is not None and updated["pinned"] is True
    again = store.read_note(meta["id"])
    assert again is not None
    assert "Followed up on Sunday." in again.body
    assert again.meta.pinned is True


def test_soft_delete(tmp_config):
    from tests.conftest import FakeClient
    store = NotebookStore(tmp_config, client=FakeClient())
    meta = store.write_note(category="reflections", title="Throwaway", body="x")
    assert store.delete_note(meta["id"]) is True
    assert store.read_note(meta["id"]) is None
    # File moved to trash, not deleted.
    trash_files = list(tmp_config.notebook_trash_dir.glob("*.md"))
    assert trash_files, "soft-deleted note should be in trash"


def test_search_finds_by_keyword(tmp_config):
    from tests.conftest import FakeClient
    store = NotebookStore(tmp_config, client=FakeClient())
    store.write_note(category="triggers", title="Loud meetings", body="loud rooms with arguing")
    store.write_note(category="coping_strategies", title="Box breathing", body="four in, four hold")
    hits = store.search_notes("loud arguing", k=3)
    assert any("Loud meetings" == h["title"] for h in hits)


def test_pinned_notes(tmp_config):
    from tests.conftest import FakeClient
    store = NotebookStore(tmp_config, client=FakeClient())
    store.write_note(category="safety_plan", title="Plan", body="...", pinned=True)
    store.write_note(category="reflections", title="Other", body="...")
    pinned = store.pinned_notes()
    assert len(pinned) == 1 and pinned[0].meta.title == "Plan"


def test_rebuild_index(tmp_config):
    from tests.conftest import FakeClient
    store = NotebookStore(tmp_config, client=FakeClient())
    meta = store.write_note(category="goals", title="Sleep schedule", body="bed by 11")
    # Corrupt the index.
    (tmp_config.notebook_dir / "_index.json").write_text("not json", encoding="utf-8")
    count = store.rebuild_index()
    assert count >= 1
    assert any(m["id"] == meta["id"] for m in store.list_notes())
