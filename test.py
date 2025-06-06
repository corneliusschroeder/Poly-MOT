import yaml, argparse, time, os, sys, json, multiprocessing
sys.path.append('/workspaces/Poly-MOT/nuscenes_devkit_uncertainty/python-sdk')
from dataloader.nusc_loader import NuScenesloader
from tracking.nusc_tracker import Tracker
from tqdm import tqdm
from eval import eval
import pdb


# Skipping the following warning from printing
# FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
#####
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The frame.append method is deprecated.*")
#####


parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--process', type=int, default=1)
# paths
localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/workspaces/Poly-MOT/dataset/nuscenes')
parser.add_argument('--config_path', type=str, default='config/nusc_config_velo.yaml')
parser.add_argument('--detection_path', type=str, default='data/detector/val/val_centerpoint.json')
parser.add_argument('--first_token_path', type=str, default='data/utils/first_token_table/trainval/nusc_first_token.json')
parser.add_argument('--result_path', type=str, default='result_debug/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result_debug/')
parser.add_argument('--eval_split', type=str, choices=['train', 'val'], default='val')
args = parser.parse_args()
print(args)


def main(result_path, token, process, nusc_loader):
    # PolyMOT modal is completely dependent on the detector modal
    result = {
        "results": {},
        "meta": {
            "use_camera": False,
            # Changed output information to true since input from OpenPCDet detector is LIDAR only
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
    }
    matched_dets = {}

    # tracking and output file
    nusc_tracker = Tracker(config=nusc_loader.config)
    for frame_data in tqdm(nusc_loader, desc='Running', total=len(nusc_loader) // process, position=token):
        if process > 1 and frame_data['seq_id'] % process != token:
            continue
        sample_token = frame_data['sample_token']

        # track each sequence
        nusc_tracker.tracking(frame_data)
        """
        only for debug
        {
            'np_track_res': np.array, [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
            'box_track_res': np.array[NuscBox], [num,]
            'no_val_track_result': bool
        }
        """
        # output process
        sample_results = []
        associated_dets = []

        if 'no_val_track_result' not in frame_data:
            
            for predict_box in frame_data['box_track_res']:
                
                box_result = {
                    "sample_token": sample_token,
                    "translation": [float(predict_box.center[0]), float(predict_box.center[1]),
                                    float(predict_box.center[2])],
                    "size": [float(predict_box.wlh[0]), float(predict_box.wlh[1]), float(predict_box.wlh[2])],
                    "rotation": [float(predict_box.orientation[0]), float(predict_box.orientation[1]),
                                 float(predict_box.orientation[2]), float(predict_box.orientation[3])],
                    "velocity": [float(predict_box.velocity[0]), float(predict_box.velocity[1])],
                    "covariance": predict_box.covariance.tolist(), # Kalman filter uncertainty estimate
                    "tracking_id": str(predict_box.tracking_id),
                    "tracking_name": predict_box.name,
                    "tracking_score": predict_box.score,
                }
                sample_results.append(box_result.copy())
                
                if predict_box.tracking_id in frame_data['associated_dets']:
                    det_data = frame_data['associated_dets'][predict_box.tracking_id]
                    # np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
                    # Uncertainties: [x_pos, y_pos, z_pos, w_bbox, l_bbox, h_bbox, yaw, vel_x, vel_y]
                    det = {
                        "sample_token": sample_token,
                        "translation": list(det_data['np_array'][:3]),
                        "size": list(det_data['np_array'][3:6]),
                        "rotation":  list(det_data['np_array'][8:12]),
                        "velocity": list(det_data['np_array'][6:8]),
                        "trans_var": list(det_data['uncertainties'][0:3]),
                        "rot_var": det_data['uncertainties'][-3],
                        "vel_var": list(det_data['uncertainties'][-2:]),
                        "class_label":  det_data['np_array'][13],
                        "detection_score": det_data['np_array'][12],
                        "tracking_id": str(predict_box.tracking_id),
                    }
                    associated_dets.append(det.copy())

        # add to the output file
        if sample_token in result["results"]:
            result["results"][sample_token] = result["results"][sample_token] + sample_results
            matched_dets[sample_token] = matched_dets[sample_token] + associated_dets
        else:
            result["results"][sample_token] = sample_results
            matched_dets[sample_token] = associated_dets

        

    # sort track result by the tracking score
    for sample_token in result["results"].keys():
        confs = sorted(
            [
                (-d["tracking_score"], ind)
                for ind, d in enumerate(result["results"][sample_token])
            ]
        )
        result["results"][sample_token] = [
            result["results"][sample_token][ind]
            for _, ind in confs[: min(500, len(confs))]
        ]

    # write file
    if process > 1:
        json.dump(result, open(result_path + str(token) + ".json", "w"))
    else:
        json.dump(result, open(result_path + "/tracking_results.json", "w"))
        json.dump(matched_dets, open(result_path + "/matched_dets.json", "w"))


# def eval(result_path, eval_path, nusc_path):
#     warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
#     from nuscenes.eval.tracking.evaluate import TrackingEval
#     from nuscenes.eval.common.config import config_factory as track_configs
#     cfg = track_configs("tracking_nips_2019")
#     nusc_eval = TrackingEval(
#         config=cfg,
#         result_path=result_path,
#         eval_set="val",
#         output_dir=eval_path,
#         verbose=True,
#         nusc_version="v1.0-trainval",
#         nusc_dataroot=nusc_path,
#     )
#     print("result in " + result_path)
#     metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.eval_path, exist_ok=True)

    # load and keep config
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)
    valid_cfg = config
    json.dump(valid_cfg, open(args.eval_path + "/config.json", "w"))
    print('writing config in folder: ' + os.path.abspath(args.eval_path))

    # load dataloader
    nusc_loader = NuScenesloader(args.detection_path,
                                 args.first_token_path,
                                 config)
    print('writing result in folder: ' + os.path.abspath(args.result_path))

    if args.process > 1:
        result_temp_path = args.result_path + '/temp_result'
        os.makedirs(result_temp_path, exist_ok=True)
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            pool.apply_async(main, args=(result_temp_path, token, args.process, nusc_loader))
        pool.close()
        pool.join()
        results = {'results': {}, 'meta': {}}
        # combine the results of each process
        for token in range(args.process):
            result = json.load(open(os.path.join(result_temp_path, str(token) + '.json'), 'r'))
            results["results"].update(result["results"])
            results["meta"].update(result["meta"])
        json.dump(results, open(args.result_path + '/results.json', "w"))
        print('writing result in folder: ' + os.path.abspath(args.result_path))
    else:
        main(args.result_path, 0, 1, nusc_loader)
        print('writing result in folder: ' + os.path.abspath(args.result_path))

    # eval result
    if os.path.isdir(args.result_path):
        result_path = os.path.join(args.result_path, 'results.json')
    else:
        result_path = args.result_path
    # eval(result_path, args.eval_path, args.nusc_path, args.eval_split)
