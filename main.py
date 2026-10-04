"""Know-When-To-Check: thin dispatcher over the stage modules.

    python main.py prepare   [args]   -> data.prepare
    python main.py cache     [args]   -> agent.build_cache      (local; use modal_app.py::cache on Modal)
    python main.py train     [args]   -> controller.train
    python main.py sweep     [args]   -> controller.train --sweep
    python main.py evaluate  [args]   -> eval.evaluate
    python main.py infer     [args]   -> agent.infer
    python main.py demo      [args]   -> demo.serve
    python main.py test               -> pytest
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "prepare":
        from data.prepare import main as m
    elif cmd == "cache":
        from agent.build_cache import main as m
    elif cmd == "train":
        from controller.train import main as m
    elif cmd == "sweep":
        from controller.train import main as m

        rest = ["--sweep", *rest]
    elif cmd == "evaluate":
        from eval.evaluate import main as m
    elif cmd == "infer":
        from agent.infer import main as m
    elif cmd == "demo":
        from demo.serve import main as m
    elif cmd == "test":
        import pytest

        sys.exit(pytest.main(["-q", "tests", *rest]))
    else:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        sys.exit(2)
    m(rest)


if __name__ == "__main__":
    main()
