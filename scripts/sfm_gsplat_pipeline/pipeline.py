"""Pipeline wrapper module.

Currently, the core Pipeline implementation lives in
`scripts/run_full_pipeline.py`. This module simply re-exports that
class so that it can be imported via the `sfm_gsplat_pipeline` package
namespace.
"""

from scripts.run_full_pipeline import Pipeline
