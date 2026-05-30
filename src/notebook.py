from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import AppConfig
from .llm_therapist import OllamaClient


SLUG_RE = re.compile(r"[^a-z0-9]+")
NOTES_COLLECTION = "notebook_notes"


def _slugify(text: str) -> str:
    base = SLUG_RE.sub("-", text.lower()).strip("-")
    return (base or "note")[:60]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class NoteMeta:
    id: str
    title: str
    category: str
    tags: List[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    pinned: bool = False
    related: List[str] = field(default_factory=list)
    source_session: str = ""
    path: str = ""


@dataclass
class Note:
    meta: NoteMeta
    body: str

    def to_markdown(self) -> str:
        front = {
            "id": self.meta.id,
            "title": self.meta.title,
            "category": self.meta.category,
            "tags": self.meta.tags,
            "created": self.meta.created,
            "updated": self.meta.updated,
            "pinned": self.meta.pinned,
            "related": self.meta.related,
            "source_session": self.meta.source_session,
        }
        return _dump_frontmatter(front) + "\n" + self.body.rstrip() + "\n"

    def to_dict(self, include_body: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = asdict(self.meta)
        if include_body:
            out["body"] = self.body
        return out


class NotebookStore:
    """Markdown-on-disk notebook with a vector-search index."""

    def __init__(self, config: AppConfig, client: Optional[OllamaClient] = None):
        self.config = config
        self.client = client
        self._lock = threading.RLock()
        self._collection = None
        self._index_path = self.config.notebook_dir / "_index.json"

    # ------------------------------------------------------------ public api
    def list_notes(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        pinned_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            metas = self._load_index()
        results = []
        for meta in metas:
            if category and meta.get("category") != category:
                continue
            if tag and tag not in (meta.get("tags") or []):
                continue
            if pinned_only and not meta.get("pinned"):
                continue
            results.append(meta)
        results.sort(key=lambda m: m.get("updated") or "", reverse=True)
        return results[: max(1, int(limit))]

    def read_note(self, note_id: str) -> Optional[Note]:
        path = self._find_path(note_id)
        if path is None:
            return None
        return self._read_from_path(path)

    def write_note(
        self,
        category: str,
        title: str,
        body: str,
        tags: Optional[List[str]] = None,
        pinned: bool = False,
        related: Optional[List[str]] = None,
        source_session: str = "",
    ) -> Dict[str, Any]:
        category = self._normalize_category(category)
        title = (title or "").strip() or "untitled"
        tags = [t.strip().lower() for t in (tags or []) if t and t.strip()]
        related = [r for r in (related or []) if r]
        now = _now_iso()
        meta = NoteMeta(
            id=_new_id(),
            title=title,
            category=category,
            tags=tags,
            created=now,
            updated=now,
            pinned=bool(pinned),
            related=related,
            source_session=source_session,
        )
        slug = _slugify(title)
        cat_dir = self.config.notebook_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        path = cat_dir / f"{slug}-{meta.id[:6]}.md"
        meta.path = str(path)
        note = Note(meta=meta, body=body.rstrip() + "\n")
        with self._lock:
            path.write_text(note.to_markdown(), encoding="utf-8")
            self._upsert_index(meta)
        self._upsert_embedding(note)
        return note.to_dict(include_body=False)

    def update_note(
        self,
        note_id: str,
        append: Optional[str] = None,
        replace_body: Optional[str] = None,
        set_tags: Optional[List[str]] = None,
        set_pinned: Optional[bool] = None,
        add_related: Optional[List[str]] = None,
        set_title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            note = self.read_note(note_id)
            if note is None:
                return None
            if replace_body is not None:
                note.body = replace_body.rstrip() + "\n"
            elif append:
                sep = "" if note.body.endswith("\n") else "\n"
                note.body = note.body + sep + append.rstrip() + "\n"
            if set_tags is not None:
                note.meta.tags = [t.strip().lower() for t in set_tags if t and t.strip()]
            if set_pinned is not None:
                note.meta.pinned = bool(set_pinned)
            if add_related:
                merged = list(note.meta.related)
                for r in add_related:
                    if r and r not in merged:
                        merged.append(r)
                note.meta.related = merged
            if set_title:
                note.meta.title = set_title.strip()
            note.meta.updated = _now_iso()
            path = Path(note.meta.path)
            path.write_text(note.to_markdown(), encoding="utf-8")
            self._upsert_index(note.meta)
        self._upsert_embedding(note)
        return note.to_dict(include_body=False)

    def link_notes(self, a_id: str, b_id: str) -> Dict[str, Any]:
        a = self.update_note(a_id, add_related=[b_id])
        b = self.update_note(b_id, add_related=[a_id])
        return {"a": a, "b": b}

    def delete_note(self, note_id: str) -> bool:
        with self._lock:
            path = self._find_path(note_id)
            if path is None:
                return False
            trash = self.config.notebook_trash_dir
            trash.mkdir(parents=True, exist_ok=True)
            target = trash / path.name
            if target.exists():
                target = trash / f"{path.stem}-{_new_id()[:4]}{path.suffix}"
            path.rename(target)
            self._remove_from_index(note_id)
        collection = self._ensure_collection()
        if collection is not None:
            try:
                collection.delete(ids=[note_id])
            except Exception:
                pass
        return True

    def search_notes(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        k = max(1, min(int(k), 20))
        vector_hits = self._vector_search(query, k * 3, category=category, tag=tag)
        keyword_hits = self._keyword_search(query, k * 3, category=category, tag=tag)
        # Reciprocal rank fusion.
        scores: Dict[str, float] = {}
        kept: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(vector_hits):
            nid = item["id"]
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (60 + rank)
            kept[nid] = item
        for rank, item in enumerate(keyword_hits):
            nid = item["id"]
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (60 + rank)
            kept.setdefault(nid, item)
        ordered = sorted(kept.values(), key=lambda m: scores.get(m["id"], 0.0), reverse=True)
        return ordered[:k]

    def pinned_notes(self) -> List[Note]:
        out: List[Note] = []
        for meta in self.list_notes(pinned_only=True, limit=20):
            note = self.read_note(meta["id"])
            if note is not None:
                out.append(note)
        return out

    def rebuild_index(self) -> int:
        with self._lock:
            metas: List[NoteMeta] = []
            for md in self.config.notebook_dir.rglob("*.md"):
                if self.config.notebook_trash_dir in md.parents:
                    continue
                note = self._read_from_path(md)
                if note is None:
                    continue
                note.meta.path = str(md)
                metas.append(note.meta)
                self._upsert_embedding(note)
            self._index_path.write_text(
                json.dumps([asdict(m) for m in metas], indent=2),
                encoding="utf-8",
            )
        return len(metas)

    # ------------------------------------------------------------ internals
    def _normalize_category(self, category: str) -> str:
        category = (category or "").strip().lower()
        category = SLUG_RE.sub("_", category).strip("_") or "reflections"
        return category

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self._index_path.exists():
            return []
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except ValueError:
            return []

    def _save_index(self, metas: List[Dict[str, Any]]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(metas, indent=2), encoding="utf-8")

    def _upsert_index(self, meta: NoteMeta) -> None:
        metas = self._load_index()
        as_dict = asdict(meta)
        replaced = False
        for i, existing in enumerate(metas):
            if existing.get("id") == meta.id:
                metas[i] = as_dict
                replaced = True
                break
        if not replaced:
            metas.append(as_dict)
        self._save_index(metas)

    def _remove_from_index(self, note_id: str) -> None:
        metas = [m for m in self._load_index() if m.get("id") != note_id]
        self._save_index(metas)

    def _find_path(self, note_id: str) -> Optional[Path]:
        for meta in self._load_index():
            if meta.get("id") == note_id:
                path = Path(meta.get("path") or "")
                if path.exists():
                    return path
        # Fallback: scan the directory in case the index is stale.
        for md in self.config.notebook_dir.rglob("*.md"):
            if self.config.notebook_trash_dir in md.parents:
                continue
            note = self._read_from_path(md)
            if note and note.meta.id == note_id:
                return md
        return None

    def _read_from_path(self, path: Path) -> Optional[Note]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        front, body = _parse_frontmatter(text)
        meta = NoteMeta(
            id=str(front.get("id") or _new_id()),
            title=str(front.get("title") or path.stem),
            category=str(front.get("category") or path.parent.name),
            tags=list(front.get("tags") or []),
            created=str(front.get("created") or ""),
            updated=str(front.get("updated") or ""),
            pinned=bool(front.get("pinned") or False),
            related=list(front.get("related") or []),
            source_session=str(front.get("source_session") or ""),
            path=str(path),
        )
        return Note(meta=meta, body=body)

    # ------------------------------------------------------------ search
    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError:
            return None
        chroma_client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        self._collection = chroma_client.get_or_create_collection(name=NOTES_COLLECTION)
        return self._collection

    def _upsert_embedding(self, note: Note) -> None:
        if self.client is None:
            return
        collection = self._ensure_collection()
        if collection is None:
            return
        try:
            text = f"{note.meta.title}\n\n{note.body}"
            embedding = self.client.embed(text)
            collection.upsert(
                ids=[note.meta.id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{
                    "category": note.meta.category,
                    "tags": ",".join(note.meta.tags),
                    "pinned": note.meta.pinned,
                    "updated": note.meta.updated,
                }],
            )
        except Exception:
            # Embedding failures must not corrupt the on-disk note.
            pass

    def _vector_search(
        self,
        query: str,
        k: int,
        category: Optional[str],
        tag: Optional[str],
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            return []
        collection = self._ensure_collection()
        if collection is None:
            return []
        try:
            embedding = self.client.embed(query)
            where: Dict[str, Any] = {}
            if category:
                where["category"] = category
            result = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where or None,
            )
        except Exception:
            return []
        ids = (result.get("ids") or [[]])[0]
        out: List[Dict[str, Any]] = []
        index = {m["id"]: m for m in self._load_index()}
        for nid in ids:
            meta = index.get(nid)
            if meta is None:
                continue
            if tag and tag not in (meta.get("tags") or []):
                continue
            out.append(meta)
        return out

    def _keyword_search(
        self,
        query: str,
        k: int,
        category: Optional[str],
        tag: Optional[str],
    ) -> List[Dict[str, Any]]:
        terms = [t for t in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for meta in self._load_index():
            if category and meta.get("category") != category:
                continue
            if tag and tag not in (meta.get("tags") or []):
                continue
            note = self.read_note(meta["id"])
            if note is None:
                continue
            haystack = (note.meta.title + "\n" + " ".join(note.meta.tags) + "\n" + note.body).lower()
            score = sum(haystack.count(t) for t in terms)
            if score > 0:
                scored.append((float(score), meta))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [meta for _, meta in scored[:k]]


# ------------------------------------------------------------ frontmatter
def _dump_frontmatter(data: Dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            rendered = "[" + ", ".join(_yaml_scalar(v) for v in value) + "]"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif value is None:
            rendered = "null"
        else:
            rendered = _yaml_scalar(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in (":", "#", "\n", '"', "'", "[", "]", "{", "}", ",")):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines:
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    front: Dict[str, Any] = {}
    for raw in lines[1:end]:
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        front[key.strip()] = _parse_yaml_value(value.strip())
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return front, body


def _parse_yaml_value(value: str) -> Any:
    if value == "" or value.lower() == "null":
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = []
        for part in _split_top_commas(inner):
            items.append(_parse_yaml_value(part.strip()))
        return items
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1].encode("utf-8").decode("unicode_escape")
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_top_commas(text: str) -> Iterable[str]:
    depth = 0
    in_quote: Optional[str] = None
    buf: List[str] = []
    for ch in text:
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            in_quote = ch
            buf.append(ch)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            yield "".join(buf)
            buf = []
            continue
        buf.append(ch)
    if buf:
        yield "".join(buf)
