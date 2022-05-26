import json
from pathlib import Path

DEFAULT_DATA_SOURCE = Path("__file__").parent / "data_sources"

def load_latest_json(data_source=None):
    data_source = (
        DEFAULT_DATA_SOURCE
        if data_source is None
        else Path(data_source)
    )
    files = sorted(
        data_source.glob("*.json"),
        key=lambda x: x.stat().st_mtime, reverse=True,
    )
    if not files:
        raise Exception(f"Couldn't find any json files in {data_source}")
    return json.loads(files[0].read_text())


CATEGORY_KEYWORDS = {
    "stout": ["stout"],
    "sour": ["sour", "lambic"],
    "ipa": ["ipa"],
    "lager": ["lager", "pilsner"],
    "other ale": ["ale"],
    "other": [],
}


def categorize(checkin):
    type_str = checkin["beer_type"].lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in type_str for keyword in keywords):
            return category
    return "other"
