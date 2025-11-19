"""Command-line interface for the unified SfM + 3DGS pipeline."""

import argparse

from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    This intentionally mirrors the arguments of `scripts/run_full_pipeline.py`
    so that existing usage patterns continue to work.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Unified Images/Video → Sharp Frames → SfM → Depth → 3DGS pipeline'
        )
    )
    # Minimal set of required / core options; the Pipeline itself still
    # defines the full behaviour. For now we defer to the original
    # `main()` in run_full_pipeline.py, which already sets up argparse.
    #
    # This parser is mainly here so that future refactors can move the
    # CLI definition into this module without breaking imports.
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Scene root directory, e.g. ./data/garden',
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='balanced',
        help='Name of built-in preset to use (fast, balanced, quality, ultra, ...)',
    )
    # We keep the parser intentionally minimal here to avoid duplicated
    # definitions. The underlying Pipeline still consumes additional
    # keyword arguments via its own argparse setup.
    return parser


def main(argv=None):
    """Entry point used by `run_pipeline.py`.

    At the moment, we simply instantiate `Pipeline` with the parsed
    arguments and call `.run()`. This keeps a clean top-level
    interface while allowing `scripts/run_full_pipeline.py` to
    continue to evolve.
    """
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    # Pipeline currently expects the full argparse.Namespace from the
    # original script. To avoid diverging, we import and delegate to
    # the original `main()` for now if extra arguments are present.
    if unknown:
        # Fallback: call the legacy CLI which knows about all flags.
        from scripts.run_full_pipeline import main as legacy_main

        legacy_main(argv)
        return
    pipeline = Pipeline(args)
    pipeline.run()
