import argparse, time, os

# Skipping the following warning from printing
# FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
#####
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*The frame.append method is deprecated.*")
#####


parser = argparse.ArgumentParser()
localtime = ''.join(time.asctime(time.localtime(time.time())).split(' '))
parser.add_argument('--nusc_path', type=str, default='/workspaces/Poly-MOT/dataset/nuscenes')
parser.add_argument('--result_path', type=str, default='result/' + localtime)
parser.add_argument('--eval_path', type=str, default='eval_result2/')
parser.add_argument('--eval_split', type=str, choices=['train', 'val'], default='val')


def eval(result_path, eval_path, nusc_path, eval_split):
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set=eval_split,
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.eval_path, exist_ok=True)
    if os.path.isdir(args.result_path):
        result_path = os.path.join(args.result_path, 'results.json')
    else:
        result_path = args.result_path
    eval(result_path, args.eval_path, args.nusc_path, args.eval_split)