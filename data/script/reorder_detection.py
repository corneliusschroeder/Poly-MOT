"""
Organize detector files in chronological order on the NuScenes dataset
"""

import os, json, sys
import argparse
sys.path.append('../..')
from utils.io import load_file
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

OUTPUT_ROOT_PATH = "../detector/"


def reorder_detection(dataset_path, detector_path, dataset_name,
                      dataset_version, detector_name, first_token_path_arg, output_path):
    """
    :param detector_path: path of detection file
    :param dataset_path: root path of dataset file
    :param dataset_name: name of dataset
    :param dataset_version: version(split) of dataset (trainval/test)
    :param detector_name: name of detector eg: CenterPoint..
    :return: Reorganized detection files .json
    """
    assert dataset_version in ['trainval', 'test'] and dataset_name in ['NuScenes', 'Waymo'], \
        "unsupported dataset or data version"

    if dataset_name == 'NuScenes':
        nusc = NuScenes(version='v1.0-' + dataset_version, dataroot=dataset_path,
                        verbose=True)
        frame_num = 6019 if dataset_version == 'trainval' else 6008

        # load detector file
        chaos_detector_json = load_file(detector_path)
        assert len(chaos_detector_json['results']) == frame_num, "wrong detection result"
        #first_token_path = '../utils/first_token_table/{}/nusc_first_token.json'.format(dataset_version)
        first_token_path = first_token_path_arg.format(dataset_version)
        all_token_table = from_first_to_all(nusc, first_token_path)
        assert len(all_token_table) == frame_num

        # reorder file
        order_file = {
            "results": {token: chaos_detector_json['results'][token] for token in all_token_table},
            "meta": chaos_detector_json["meta"]
        }

        # output file
        version = 'val' if dataset_version == "trainval" else 'test'
        os.makedirs(output_path + version, exist_ok=True)
        OUTPUT_PATH = output_path + version + f"/{version}_{detector_name}.json"
        print(f"write order detection file to {OUTPUT_PATH}")
        json.dump(order_file, open(OUTPUT_PATH, "w"))

    else:
        raise Exception("Waymo dataset is not currently supported")


def from_first_to_all(nusc, first_token_path):
    """
    :param nusc: NuScenes class
    :param first_token_path: path of first frame token for each seq
    :return: list format token table
    """
    first_token_table, seq_num = load_file(first_token_path), 150
    assert len(first_token_table) == seq_num, "wrong token table"
    all_token_table = []
    for first_token in first_token_table:
        curr_token = first_token
        while curr_token != '':
            all_token_table.append(curr_token)
            curr_token = nusc.get('sample', curr_token)['next']

    return all_token_table

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize detector files in chronological order on the NuScenes dataset.")

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
    parser.add_argument(
        "--detector_name",
        default="centerpoint",
        help="Name of detector eg: CenterPoint.."
    )
    parser.add_argument(
        "--first_token_path",
        default="../utils/first_token_table/{}/nusc_first_token.json",
        help="Name of json file result with first frame token for each scene"
    )

    # Output-related arguments
    parser.add_argument(
        "--output_path",
        default="../detector/",
        help="Directory where the reorganized detector file will be saved."
    )

    args = parser.parse_args()

    # Call the extract function with parsed arguments
    reorder_detection(
        dataset_path=args.dataset_path,
        detector_path=args.detector_path,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        detector_name=args.detector_name,
        first_token_path_arg=args.first_token_path,
        output_path=args.output_path
    )
