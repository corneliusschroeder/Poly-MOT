import sklearn.metrics as sk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import copy

CLASSES = ['car', 'bus', 'truck', 'trailer', 'pedestrian']

def plot_PRC_curve(data_dir, aggregate_vehicles=True, data='nuscenes'):
    # Aggregate car, bus, truck into 'vehicles'
    vehicle_classes = ['car', 'bus', 'truck']
    classes = copy(CLASSES)
    
    if data == 'edgarscenes':
        classes.remove('trailer')
    
    num_plots = len(classes)
    
    if aggregate_vehicles:
        num_plots = num_plots - 2

    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots*6, 4))

    # Create PCR curve
    if aggregate_vehicles:
        vehicle_preds = []
        vehicle_gts = []
        for class_name in vehicle_classes:
            path = data_dir + class_name + '_pred_vel.csv'
            df = pd.read_csv(path)
            mask = df['mov_Pred'].notnull() & df['mov_GT'].notnull()
            vehicle_preds.append(df['mov_Pred'][mask].to_numpy())
            vehicle_gts.append(df['mov_GT'][mask].to_numpy())
        vehicle_preds = np.concatenate(vehicle_preds)
        vehicle_gts = np.concatenate(vehicle_gts)
        prec, rec, threshold = sk.precision_recall_curve(vehicle_gts, vehicle_preds, pos_label='moving')
        sk.PrecisionRecallDisplay.from_predictions(vehicle_gts, vehicle_preds, pos_label='moving', ax=axes[0], plot_chance_level=True)
        ax2 = axes[0].twinx()
        ax2.plot(rec[1:], threshold, color='orange', label='Threshold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Threshold (log scale)')
        axes[0].set_title('Vehicles')

    # Plot for remaining classes (trailer, pedestrian)
    plot_idx = 0
    if aggregate_vehicles:
        plot_idx = 1
    for class_name in classes:
        if class_name in vehicle_classes and aggregate_vehicles:
            continue
        path = data_dir + class_name + '_pred_vel.csv'
        df = pd.read_csv(path)
        mask = df['mov_Pred'].notnull() & df['mov_GT'].notnull()
        prediction_arr = df['mov_Pred'][mask].to_numpy()
        GT_arr = df['mov_GT'][mask].to_numpy()
        print(class_name)
        prec, rec, threshold = sk.precision_recall_curve(GT_arr, prediction_arr, pos_label='moving')
        sk.PrecisionRecallDisplay.from_predictions(GT_arr, prediction_arr, pos_label='moving', ax=axes[plot_idx], plot_chance_level=True)
        ax2 = axes[plot_idx].twinx()
        ax2.plot(rec[1:], threshold, color='orange', label='Threshold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Threshold (log scale)')
        axes[plot_idx].set_title(class_name.capitalize())
        plot_idx += 1

    plt.tight_layout()
    plt.show()


def plot_vel_PRC_curve(data_dir, aggregate_vehicles=True, data='nuscenes'):
    # Aggregate car, bus, truck into 'vehicles'
    vehicle_classes = ['car', 'bus', 'truck']
    classes = copy(CLASSES)
    
    if data == 'edgarscenes':
        classes.remove('trailer')
    
    num_plots = len(classes)
    
    if aggregate_vehicles:
        num_plots = num_plots - 2

    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots*6, 4))

    # Create PCR curve
    if aggregate_vehicles:
        vehicle_preds = []
        vehicle_gts = []
        for class_name in vehicle_classes:
            path = data_dir + class_name + '_pred_vel.csv'
            df = pd.read_csv(path)
            mask = df['vel_Pred'].notnull() & df['mov_GT'].notnull()
            vehicle_preds.append(df['vel_Pred'][mask].to_numpy())
            vehicle_gts.append(df['mov_GT'][mask].to_numpy())
        vehicle_preds = np.concatenate(vehicle_preds)
        vehicle_gts = np.concatenate(vehicle_gts)
        prec, rec, threshold = sk.precision_recall_curve(vehicle_gts, vehicle_preds, pos_label='moving')
        sk.PrecisionRecallDisplay.from_predictions(vehicle_gts, vehicle_preds, pos_label='moving', ax=axes[0], plot_chance_level=True)
        ax2 = axes[0].twinx()
        ax2.plot(rec[1:], threshold, color='orange', label='Threshold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Threshold (log scale)')
        axes[0].set_title('Vehicles')

    # Plot for remaining classes (trailer, pedestrian)
    plot_idx = 0
    if aggregate_vehicles:
        plot_idx = 1
    for class_name in classes:
        if class_name in vehicle_classes and aggregate_vehicles:
            continue
        path = data_dir + class_name + '_pred_vel.csv'
        df = pd.read_csv(path)
        mask = df['vel_Pred'].notnull() & df['mov_GT'].notnull()
        prediction_arr = df['vel_Pred'][mask].to_numpy()
        GT_arr = df['mov_GT'][mask].to_numpy()
        print(class_name)
        prec, rec, threshold = sk.precision_recall_curve(GT_arr, prediction_arr, pos_label='moving')
        sk.PrecisionRecallDisplay.from_predictions(GT_arr, prediction_arr, pos_label='moving', ax=axes[plot_idx], plot_chance_level=True)
        ax2 = axes[plot_idx].twinx()
        ax2.plot(rec[1:], threshold, color='orange', label='Threshold')
        ax2.set_yscale('log')
        ax2.set_ylabel('Threshold (log scale)')
        axes[plot_idx].set_title(class_name.capitalize())
        plot_idx += 1

    plt.tight_layout()
    plt.show()