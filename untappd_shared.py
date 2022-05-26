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


def categorize(checkin):
    typestr = checkin["beer_type"].lower()
    if "stout" in typestr:
        return "stout"
    #if "porter" in typestr:
    #    return "porter"
    if "sour" in typestr or "lambic" in typestr:
        return "sour"
    if "ipa" in typestr:
        return "ipa"
    if "lager" in typestr or "pilsner" in typestr:
        return "lager"
    if "ale" in typestr:
        return "other ale"
    return "other"
