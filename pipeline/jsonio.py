#!/usr/bin/env python3
# =============================================================================
#  jsonio.py  -  fast JSON with a safe fallback
# -----------------------------------------------------------------------------
#  Uses ujson (faster, as in the paper-1 pipeline) when available, and falls
#  back to the stdlib json transparently when it is not installed. Same calls
#  as json: dump / dumps / load / loads. BACKEND tells which one is active.
#
#  dump/load are implemented via dumps/loads + file I/O so both backends behave
#  identically regardless of minor API differences.
# =============================================================================
try:
    import ujson as _json

    BACKEND = "ujson"
except ImportError:  # pragma: no cover - environment-dependent
    import json as _json

    BACKEND = "json"


def dumps(obj, indent=2):
    # ujson escapes forward slashes by default; keep paths readable.
    if BACKEND == "ujson":
        return _json.dumps(obj, indent=indent, escape_forward_slashes=False)
    return _json.dumps(obj, indent=indent)


def loads(text):
    return _json.loads(text)


def dump(obj, fp, indent=2):
    fp.write(dumps(obj, indent=indent))


def load(fp):
    return loads(fp.read())
