"""One validated corpus representation for all ranking and evaluation paths."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import html
import re
import unicodedata
from .io import read_json, digest


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(flatten(value[k]) for k in sorted(value))
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(x) for x in value)
    return re.sub(r"<[^>]*>", " ", html.unescape(str(value)))


def tokens(text):
    return re.findall(r"\b\w+\b", unicodedata.normalize("NFKC", flatten(text)).casefold())


def timestamp(text):
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt).astimezone(timezone.utc)


@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    description: str
    publisher: str
    keywords: str
    themes: str
    modified: str | None
    resource_count: int
    license: str
    url: str = ""

    @classmethod
    def parse(cls, raw):
        identity = raw.get("id", "")
        if not isinstance(identity, str):
            raise ValueError("Dataset id must be a string")
        identity = identity.strip()
        if not identity:
            raise ValueError("A dataset needs a nonempty id")
        count = raw.get("resource_count", raw.get("num_resources"))
        if count is None:
            count = len(raw.get("resources") or [])
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid resource count for {identity}")
        return cls(identity, flatten(raw.get("title")),
                   flatten(raw.get("description") or raw.get("notes")),
                   flatten(raw.get("publisher") or raw.get("organization_name") or raw.get("organization")),
                   flatten(raw.get("keywords") or raw.get("tags")),
                   flatten(raw.get("themes") or raw.get("groups")),
                   raw.get("modified") or raw.get("metadata_modified"), count,
                   flatten(raw.get("license") or raw.get("license_id")),
                   str(raw.get("url") or ""))


class Corpus:
    def __init__(self, datasets):
        self.datasets = sorted(datasets, key=lambda d: d.id)
        self.by_id = {d.id: d for d in self.datasets}
        if not self.datasets or len(self.by_id) != len(self.datasets):
            raise ValueError("Corpus must be nonempty and dataset IDs unique")
        self.hash = digest([asdict(d) for d in self.datasets])

    @classmethod
    def load(cls, path):
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else payload["datasets"]
        return cls([Dataset.parse(row) for row in rows])
