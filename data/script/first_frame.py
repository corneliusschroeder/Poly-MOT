"""
get first frame token for every seq on the NuScenes dataset
TODO: support Waymo dataset
"""

import os, json, sys
import argparse
sys.path.append('../..')
from utils.io import load_file
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

FIRST_TOKEN_ROOT_PATH = '../utils/first_token_table/'


def extract_first_token(dataset_path, detector_path, dataset_name, dataset_version, output_path):
    """
    :param dataset_path: path of dataset
    :param dataset_name: name of dataset
    :param detector_path: path of detection file
    :param dataset_version: version(split) of dataset (trainval/test)
    :return: first frame token table .json
    """
    assert dataset_version in ['trainval', 'test'] and dataset_name in ['NuScenes', 'Waymo'], \
        "unsupported dataset or data version"

    if dataset_name == 'NuScenes':
        nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=dataset_path,
                        verbose=True)
        frame_num = 6019 if dataset_version == 'trainval' else 6008
        seq_num = 150

        # load detector file
        detector_json = load_file(detector_path)
        assert len(detector_json['results']) == frame_num, "wrong detection result"

        # get first frame token of each seq
        first_token_table = []
        print("Extracting first frame token...")
        for sample_token in tqdm(detector_json['results']):
            if nusc.get('sample', sample_token)['prev'] == '':
                first_token_table.append(sample_token)
        assert len(first_token_table) == seq_num, "wrong detection result"

        # write token table
        os.makedirs(output_path + dataset_version, exist_ok=True)
        FIRST_TOKEN_PATH = os.path.join(output_path, dataset_version, "nusc_first_token.json")
        print(f"write token table to {FIRST_TOKEN_PATH}")
        json.dump(first_token_table, open(FIRST_TOKEN_PATH, "w"))

    else:
        raise Exception("Waymo dataset is not currently supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first-frame tokens for NuScenes (or Waymo in the future).")

    # Dataset-related arguments
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the NuScenes dataset directory (e.g., /mnt/share/.../nuscenes)."
    )
    parser.add_argument(
        "--detector_path",
        required=True,
        help="Path to the .json detection file."
    )
    parser.add_argument(
        "--dataset_name",
        default="NuScenes",
        choices=["NuScenes", "Waymo"],
        help="Name of the dataset. Currently only 'NuScenes' is supported."
    )
    parser.add_argument(
        "--dataset_version",
        default="trainval",
        choices=["trainval", "test"],
        help="Dataset version/split (either 'trainval' or 'test')."
    )

    # Output-related arguments
    parser.add_argument(
        "--output_path",
        default="../utils/first_token_table",
        help="Directory where the first-token JSON file will be saved."
    )

    args = parser.parse_args()

    # Call the extract function with parsed arguments
    extract_first_token(
        dataset_path=args.dataset_path,
        detector_path=args.detector_path,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        output_path=args.output_path
    )
