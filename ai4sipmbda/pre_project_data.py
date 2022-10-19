import argparse
from pathlib import Path

import numpy as np
import torchio as tio

from utils import fetching
from utils.difumo_utils import project_to_difumo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_images",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-j", "--njobs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--data-dir",
        default="/storage/store2/data/",
        type=str,
    )
    parser.add_argument(
        "-v", "--verbose",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--output_path",
        default="../hcp900_difumo_matrices/difumo_data.npy",
        type=str,
    )
    args = parser.parse_args()

    print(" --- LOADING DATA --- ")
    data = fetching.fetch_nv(
        args.data_dir,
        max_images=args.max_images,
        verbose=args.verbose,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True)

    print(" --- LOADING/CREATING DIFUMO PROJECTORS --- ")
    Z_inv = np.load("../hcp900_difumo_matrices/Zinv.npy")
    mask = np.load("../hcp900_difumo_matrices/mask.npy")

    img_tio_test = [tio.ScalarImage(path) for path in data["images"]]
    print(" --- PROJECTING --- ")
    X_test = project_to_difumo(img_tio_test, Z_inv, mask, n_jobs=args.njobs)

    print(" --- SAVING --- ")
    np.save(output_path, X_test)
