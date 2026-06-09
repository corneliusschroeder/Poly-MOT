# Poly-MOT with Uncertainty Quantification
This is a modified version of the Official Poly-MOT Repo (IROS 2023) that extends the tracking framework with **heteroscedastic measurement uncertainty quantification** from object detections.
![1688699111](https://github.com/lixiaoyu2000/Poly-MOT/blob/main/docs/Poly-MOT.jpg)

> [**Poly-MOT: A Polyhedral Framework For 3D Multi-Object Tracking**](https://arxiv.org/abs/2307.16675),  
> Xiaoyu Li<sup>\*</sup>, Tao Xie<sup>\*</sup>, Dedong Liu<sup>\*</sup>, Jinghan Gao, Kun Dai, Zhiqiang Jiang, Lijun Zhao, Ke Wang,                   
> *arXiv technical report ([arXiv 2307.16675](https://arxiv.org/abs/2307.16675))*,  
> IROS 2023

## Extensions in This Repository

This fork extends the original Poly-MOT framework with the following enhancements:
- **Heteroscedastic Measurement Uncertainty**: The Kalman filter now accepts per-object, per-dimension uncertainty estimates from the detector
- **Uncertainty-Aware EKF**: Extended Kalman Filter is modified to incorporate detection uncertainty into the measurement noise matrix (R)
- **Uncertainty Evaluation**: Added tools for evaluating tracking uncertainty calibration, including Expected Calibration Error (ECE) computation
- **Flexible Configuration**: Support for enabling/disabling uncertainty usage per object class through YAML configuration

These extensions are described in detail in the following paper:
> **Improving EKF Consistency in 3D Multi-Object Tracking via Heteroscedastic Detection Uncertainty**  
> Cornelius Schröder, Felix Fent, Markus Lienkamp  
> *Proceedings of the 12th International Conference on Vehicle Technology and Intelligent Transport Systems (VEHITS 2026)*


## Citation

If you use the **original Poly-MOT framework**, please cite the original paper:
```
@misc{li2023polymot,
      title={Poly-MOT: A Polyhedral Framework For 3D Multi-Object Tracking}, 
      author={Xiaoyu Li and Tao Xie and Dedong Liu and Jinghan Gao and Kun Dai and Zhiqiang Jiang and Lijun Zhao and Ke Wang},
      year={2023},
      eprint={2307.16675},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

If you use the **uncertainty quantification extension** in this fork, please cite both papers:
```
@inproceedings{VEHITS,
  author    = {Schr{\"o}der, Cornelius and Fent, Felix and Lienkamp, Markus},
  title     = {Improving {EKF} Consistency in {3D} Multi-Object Tracking via Heteroscedastic Detection Uncertainty},
  booktitle = {Proceedings of the 12th International Conference on Vehicle Technology and Intelligent Transport Systems},
  series    = {VEHITS},
  year      = {2026},
  pages     = {51--59},
  publisher = {SCITEPRESS -- Science and Technology Publications},
  isbn      = {978-989-758-831-0},
  issn      = {2184-495X}
}
```

And the original Poly-MOT paper above.

## News

- 2024-03-20. Warm-up :fire:! We released [Fast-Poly](https://github.com/lixiaoyu2000/FastPoly), a fast version of Poly-MOT. Welcome to follow.
- 2023-12-09. Warm-up :fire:! The official repo of [RockTrack](https://github.com/lixiaoyu2000/Rock-Track) has been released. We will release code soon. Welcome to follow.
- 2023-09-08. **Version 1.0 has been released.**
- 2023-07-01. Poly-MOT is accepted at IROS 2023 :zap:.
- 2023-03-01. Our method ranks first among all methods on the NuScenes tracking [benchmark](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any) :fire:.

## Release notes

### Implemented functions
- 2023-12-12. In this version, we implemented two extra motion models (CV, CTRV).
- 2023-12-08. In this version, we made d_eucl parallel.
- 2023-09-08. In this version, we implemented API for the `nuScenes` dataset, five Similarity metrics(giou3d, gioubev, iou3d, ioubev, eucl), three Motion models(CTRA, Bicycle, CA), one NMS method(Classic NMS), three matching methods(Greedy, Hungarian, MNN).

### TODO list
- 2023-09-08. More NMS method;

## Abstract
We propose Poly-MOT, an efficient 3D MOT method based on the Tracking-By-Detection framework that enables the tracker to choose the most appropriate tracking criteria for each object category.
Poly-MOT leverages different motion models for various object categories to characterize distinct types of motion accurately. 
We also introduce the constraint of the rigid structure of objects into a specific motion model to accurately describe the highly nonlinear motion of the object.
Additionally, we introduce a two-stage data association strategy to ensure that objects can find the optimal similarity metric from three custom metrics for their categories and reduce missing matches.

<div align=center>
<img src="https://github.com/lixiaoyu2000/Poly-MOT/blob/main/docs/Visualization.gif"/>
</div>

## Highlights

- **Uncertainty-Aware Tracking (New Extension).** :chart_with_upwards_trend:
  - Integrates heteroscedastic measurement uncertainty from object detections into the Extended Kalman Filter
  - Allows the tracker to adapt filter behavior based on per-detection, per-dimension uncertainty estimates
  - Enables uncertainty quantification and calibration evaluation of tracking outputs
  - Supports configurable uncertainty usage per object class

- **Best-performance(75.4 AMOTA).** :chart_with_upwards_trend:
  - Poly-MOT enables the tracker to choose the most appropriate tracking criteria for each object category.
  - With the powerful detector [Largerkernel3D](https://github.com/dvlab-research/LargeKernel3D), Poly-MOT achieves 75.4 AMOTA on the NuScenes test set.
  - Poly-MOT achieves 73.1 AMOTA on the val set with [CenterPoint](https://github.com/tianweiy/CenterPoint) for a fair comparison.
  
- **Real-time(0.3s per frame).** :zap:
  - Poly-MOT follows the Tracking-By-Detection(TBD) framework, and is learning-free.
  - During online tracking, No any additional input(including dataset, images, map, ...) needed besides the detector.
  - We first proposed the *half-parallel GIOU operator* under the `Python` implementation.
  - On the NuScenes, Poly-MOT can run at 3 FPS (Frame Per Second) on Intel 9940X.
  
- **Strong-scalability(one-config-fit-all).** :ledger:
  - Poly-MOT has integrated a variety of tracking technologies in the code, and uses `yaml` to manage these hyperparameters in a unified way, you can customize your own tracker arbitrarily.
  - Uncertainty quantification can be enabled/disabled per object class through configuration
  
- **Well-readability(many comments).** :clipboard:
  - We have recorded each tracking module's design reasons, effects, and ideas in the code. 
  - You can grasp our insight and even start discussing any comments with us.

  
## Main Results

### 3D Multi-object tracking on NuScenes test set

 Method       | Detector      | AMOTA    | AMOTP    | IDS      |   
--------------|---------------|----------|----------|----------|
 Poly-MOT     | LargeKernel3D | 75.4     | 42.2     | 292      |         
 
 
You can find detailed results on the NuScenes test set on this [website](https://eval.ai/web/challenges/challenge-page/476/leaderboard/1321).

### 3D Multi-object tracking on NuScenes val set

 Method        | Detector        | AMOTA    | AMOTP    | IDS      |   
---------------|-----------------|----------|----------|----------|
 Poly-MOT      | Centerpoint     | 73.1     | 52.1     | 281      |  
 Poly-MOT      | LargeKernel3D-L | 75.2     | 54.1     | 252      |

## Uncertainty Quantification

### Overview
This extension adds support for heteroscedastic measurement uncertainty quantification. The Extended Kalman Filter (EKF) can now incorporate per-detection, per-dimension uncertainty estimates from your object detector, enabling more informed state estimation and uncertainty-aware tracking.

### Configuration

The uncertainty features are controlled via the `motion_model` section in your configuration file (YAML):

```yaml
motion_model:
  # Use heteroscedastic measurement noise from detector per class
  # Set to True for classes where detector provides uncertainty estimates
  heteroscedastic_R: {0: True, 1: True, 2: False, 3: False, 4: False, 5: False, 6: False}
  
  # You can also configure which motion models and metrics to use per class
  model: {0: 'CV', 1: 'CV', 2: 'CA', ...}
  # ... other motion model configurations
```

Example configurations are provided in the `config/` directory:
- `nusc_config_no_velo.yaml` - Without velocity uncertainty
- `nusc_config_unc_BL_velo.yaml` - With heteroscedastic uncertainty and velocity

### Input Requirements

For uncertainty quantification to work, your detector outputs must include uncertainty estimates in the detection JSON. The NuscBox class expects:
- `detection_uncertainty`: Per-dimension uncertainty values in the format `[x_pos, y_pos, z_pos, w_bbox, l_bbox, h_bbox, yaw, vel_x, vel_y]`

These uncertainties are automatically incorporated into the measurement noise matrix (R) of the Kalman Filter update step.

### Uncertainty Evaluation

After tracking, you can evaluate the calibration of uncertainty estimates using the provided evaluation tools:

```bash
# Compute Expected Calibration Error (ECE) and confidence intervals
python compute_ece.py --eval_path <path_to_tracking_results>
```

This generates:
- Per-class calibration metrics
- Confidence interval analysis
- ECE values for position and orientation components

## Use Poly-MOT

### 1. Create and activate environment
```
   conda env create -f environment.yaml  
   conda activate polymot
```

### 2. Required Data

#### Download 3D detector

We strongly recommend that you download the detector file `.json` from official websites of Pioneer detector works ([CenterPoint](https://github.com/tianweiy/CenterPoint), etc.).
In online tracking, we need to use detector files in `.json` format.

**Note on Uncertainty**: If your detector provides uncertainty estimates, ensure they are included in the JSON detection format and that the corresponding class in `heteroscedastic_R` configuration is set to `True`.

#### Prepare the token table for online inference

`sample token table` is used to identify the first frame of each scene.

```shell
cd Poly-MOT/data/script
python first_frame.py
```

The file path(detector path, database path, etc.) within the function `extract_first_token` needs to be modified.
The result will be output in `data/utils/first_token_table/{version}/nusc_first_token.json`.

#### Prepare the detector for online inference

The tracker requires that the detectors must be arranged in chronological order.
`reorder_detection.py` is used to reorganize detectors in chronological order.

```shell
cd Poly-MOT/data/script
python reorder_detection.py
```

The file path(detector path, database path, token path, etc.) within the function `reorder_detection` needs to be modified.
The result will be output in `data/detector/first_token_table/{version}/{version}_{detector_name}.json`.

#### Prepare the database for evaluation

Although Poly-MOT does not need the database during online inference, in order to evaluate the tracking effect, the database is still necessary.
Download data and organize it as follows:
```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- keyframes
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- map infos
       ├── v1.0-trainval <-- train/val set metadata 
       ├── v1.0-test     <-- test set metadata
```


### 3. Running and Evaluation

#### Config
All hyperparameters are encapsulated in `config/nusc_config.yaml`, you can change the `yaml` file to customize your own tracker.
**The accuracy with `CenterPoint` in the paper can be reproduced through the parameters above the current `nusc_config.yaml`.**

#### Running
After downloading and organizing the detection files, you can simply run:
```
python test.py
```
The file path(detector path, token path, database path, etc.) within the file needs to be modified. 
Besides, you can also specify the file path using the terminal command, as following:
```
python test.py --eval_path <eval path>
```


#### Evaluation
Tracking evaluation will be performed automatically after tracking all scenarios.

##### Standard Tracking Metrics
The evaluation script computes standard MOT metrics (AMOTA, AMOTP, etc.) based on the NuScenes benchmark.

##### Uncertainty Evaluation (New)
After obtaining tracking results with uncertainty estimates, evaluate the calibration of the uncertainties:

```bash
# Evaluate uncertainty calibration
python eval.py --result_path <tracking_results.json> --eval_path <output_dir>

# Compute Expected Calibration Error (ECE)
python compute_ece.py --eval_path <evaluation_output_dir>
```

This generates:
- `confidence_intervals_all_classes.txt`: Per-class confidence interval analysis
- `metrics_summary.json`: Summary of calibration metrics including ECE values
- Plots of calibration curves for each object class

The ECE metric quantifies how well the uncertainty estimates align with actual tracking errors. Lower ECE indicates better-calibrated uncertainties.


## Visualization
Give the box to render in the specified format and the token of the background to get the trajectory rendering map. For example, `black` boxes represent detection results, and `other colored` boxes represent existing trajectories, see the following:
<div align=center><img width="500" height="500" src="https://github.com/lixiaoyu2000/Poly-MOT/blob/main/docs/2.png"/></div>

You can run the Jupyer notebook [Visualization.ipynb](https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/Visualization.ipynb).


## Contact

For questions about the **original Poly-MOT framework**, please contact:
Xiaoyu Li(李效宇) lixiaoyu12349@icloud.com

For questions about the **uncertainty quantification extension** in this fork, please refer to the GitHub issues in this repository.

Any questions or suggestions about the paper/code are welcome :open_hands:! 
Please feel free to submit PRs if you find any problems or develop better features :raised_hands:!

## License

Poly-MOT and this uncertainty quantification extension are released under the MIT license. See the original Poly-MOT repository for details.


## Acknowledgement

This project is not possible without the following excellent open-source codebases :fist:.

In the detection part, many thanks to the following open-sourced codebases:
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [Largerkernel3D](https://github.com/dvlab-research/LargeKernel3D)

In the tracking part, many thanks to the following open-sourced codebases:
- [AB3DMOT](https://github.com/gideontong/AB3DMOT)
- [EagerMOT](https://github.com/aleksandrkim61/EagerMOT)
- [SimpleTrack](https://github.com/tusen-ai/SimpleTrack)
- [CBMOT](https://github.com/cogsys-tuebingen/CBMOT)

