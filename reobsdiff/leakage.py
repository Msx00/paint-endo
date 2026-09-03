"""Central fail-closed target-view leakage checks."""

from pathlib import Path
import json
import re

FORBIDDEN_KEY = re.compile(
    r"(^|_)(e1|endo1|endoscope1)_(rgb|image|depth|target|gt)($|_)|"
    r"(^|_)(target_(rgb|image|depth|path)|ground_truth|gt_image)($|_)", re.I)
FORBIDDEN_PATH = re.compile(r"[/\\](endoscope1|endo1)[/\\].*(rgb|depth|[/\\][LR][/\\])?", re.I)


def assert_no_target_view(value, context="training object"):
    def walk(item, trail):
        if isinstance(item, dict):
            for key, child in item.items():
                # Camera geometry is explicitly permitted; image/depth values are not.
                key_l = str(key).lower()
                if FORBIDDEN_KEY.search(str(key)) and not any(x in key_l for x in ("camera", "pose", "intrinsic", "calibration")):
                    raise RuntimeError("target-view leakage key {} in {}".format(".".join(trail + [str(key)]), context))
                walk(child, trail + [str(key)])
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, trail + [str(index)])
        elif isinstance(item, (str, Path)) and FORBIDDEN_PATH.search(str(item)):
            raise RuntimeError("target-view image/depth path {} in {}".format(item, context))
    walk(value, [])


def audit_json(path):
    payload = json.loads(Path(path).read_text())
    assert_no_target_view(payload, str(path))
    return payload
