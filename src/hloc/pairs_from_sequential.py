import os
import argparse
import collections.abc as collections
from pathlib import Path
from typing import Optional, Union, List

import numpy as np

from hloc import logger
from hloc import pairs_from_retrieval
from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.io import list_h5_names


def main(
    output: Path,
    image_list: Optional[Union[Path, List[str]]] = None,
    features: Optional[Path] = None,
    # name this "overlap" so run_hloc.py can call it directly
    overlap: int = 10,
    quadratic_overlap: bool = True,
    use_loop_closure: bool = False,
    # loop-closure extras
    retrieval_path: Optional[Union[Path, str]] = None,
    retrieval_interval: int = 2,
    num_loc: int = 5,
) -> None:
    """Build sequential pairs, optionally with loop-closure."""
    # 1) figure out image names
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        # if we only have features, list images from the .h5
        names_q = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")

    pairs = []
    N = len(names_q)

    # 2) sequential window
    for i in range(N - 1):
        for j in range(i + 1, min(i + overlap + 1, N)):
            pairs.append((names_q[i], names_q[j]))

            # optional quadratic neighbors, same logic as your version
            if quadratic_overlap:
                q = 2 ** (j - i)
                if q > overlap and i + q < N:
                    pairs.append((names_q[i], names_q[i + q]))

    # 3) optional loop closure: add sparse retrieval pairs on top
    if use_loop_closure:
        if retrieval_path is None:
            raise ValueError("--use-loop-closure was set but no retrieval_path was provided")

        tmp_pairs_path = output.parent / "retrieval-pairs-tmp.txt"

        # choose every retrieval_interval-th image as a query
        query_list = names_q[::retrieval_interval]
        M = len(query_list)
        match_mask = np.zeros((M, N), dtype=bool)

        # mask out neighbors we already added via sequential matching
        for qi, name in enumerate(query_list):
            base_idx = qi * retrieval_interval
            for k in range(overlap + 1):
                if 0 <= base_idx - k < N:
                    match_mask[qi][base_idx - k] = True
                if 0 <= base_idx + k < N:
                    match_mask[qi][base_idx + k] = True

                if quadratic_overlap:
                    step = 2 ** k
                    if 0 <= base_idx - step < N:
                        match_mask[qi][base_idx - step] = True
                    if 0 <= base_idx + step < N:
                        match_mask[qi][base_idx + step] = True

        # run retrieval just for loop-closure images
        pairs_from_retrieval.main(
            retrieval_path,
            tmp_pairs_path,
            num_matched=num_loc,
            match_mask=match_mask,
            db_list=names_q,
            query_list=query_list,
        )

        retrieval = parse_retrieval(tmp_pairs_path)
        for key, vals in retrieval.items():
            for v in vals:
                pairs.append((key, v))

        os.unlink(tmp_pairs_path)

    logger.info(f"Found {len(pairs)} sequential pairs.")
    with open(output, "w") as f:
        f.write("\n".join(f"{i} {j}" for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create image pairs from a sequence, optionally with loop-closure."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument(
        "--quadratic_overlap",
        action="store_true",
        help="Also connect to quadratic neighbors",
    )
    parser.add_argument(
        "--use_loop_closure",
        action="store_true",
        help="Add a few retrieval-based pairs for long-range connections",
    )
    parser.add_argument("--retrieval_path", type=Path)
    parser.add_argument("--retrieval_interval", type=int, default=2)
    parser.add_argument("--num_loc", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))
