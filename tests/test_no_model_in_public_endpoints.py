import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_model_in_templates():
    tpl = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    # check for form inputs with name="model"
    assert 'name="model"' not in tpl, "Found a 'model' form input in templates/index.html"


def test_no_model_in_api():
    api = (ROOT / "api.py").read_text(encoding="utf-8")
    # Only fail if 'model' is used as a form parameter in an endpoint signature, e.g.
    # model: str = Form("...") or model = Form("...")
    pattern = re.compile(r"\bmodel\b\s*(?:[:=])[^\n]*\bForm\s*\(")
    m = pattern.search(api)
    assert not m, f"Found a public 'model' Form parameter in api.py near: {m.group(0) if m else ''}"
