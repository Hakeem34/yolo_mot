# mot_check.py
"""
MOTChallenge形式のGTとTrackerデータからIDF1とMOTAを計算するプログラム
"""
import os
import sys
import csv
import motmetrics as mm
import numpy as np
import pandas as pd
from pathlib import Path


def load_mot_data(filepath):
    """
    MOTChallenge形式のCSVファイルを読み込む
    形式: <frame_id>,<track_id>,<x>,<y>,<width>,<height>,<conf>,<class>,<visibility>
    
    Args:
        filepath: MOTChallenge形式のCSVファイルパス
    
    Returns:
        frame_id: フレームIDの配列
        track_id: トラックIDの配列
        bboxes: バウンディングボックス情報
    """
    data = pd.read_csv(
        filepath,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf', 'ClassId', 'Visibility']
    )
    
    # Conf が -1 の場合は削除（無効なデータ）
    data = data[data['Conf'] >= 0]
    
    return data


def compute_mot_metrics(gt_file, tracker_file):
    """
    GtファイルとTrackerファイルからMOTA、IDF1などのメトリクスを計算
    
    Args:
        gt_file: Ground Truth CSVファイルパス
        tracker_file: Tracker出力CSVファイルパス
    
    Returns:
        dict: メトリクスの結果
    """
    # データを読み込む
    gt_data = load_mot_data(gt_file)
    tracker_data = load_mot_data(tracker_file)
    
    # motmetrics の accumulatorを作成
    acc = mm.MOTAccumulator(auto_id=True)
    
    # フレームごとに処理
    gt_frames = gt_data['FrameId'].unique()
    
    for frame_id in sorted(gt_frames):
        gt_frame = gt_data[gt_data['FrameId'] == frame_id]
        tracker_frame = tracker_data[tracker_data['FrameId'] == frame_id]
        
        # GT と Tracker の座標をバウンディングボックスから中心座標に変換
        if len(gt_frame) > 0:
            gt_centers = gt_frame[['X', 'Y']].values
        else:
            gt_centers = np.empty((0, 2))
        
        if len(tracker_frame) > 0:
            tracker_centers = tracker_frame[['X', 'Y']].values
        else:
            tracker_centers = np.empty((0, 2))
        
        # トラックIDを取得
        gt_ids = gt_frame['Id'].values
        tracker_ids = tracker_frame['Id'].values
        
        # IoUを計算するために、バウンディングボックス情報を使用
        if len(gt_frame) > 0:
            gt_bboxes = gt_frame[['X', 'Y', 'Width', 'Height']].values
        else:
            gt_bboxes = np.empty((0, 4))
        
        if len(tracker_frame) > 0:
            tracker_bboxes = tracker_frame[['X', 'Y', 'Width', 'Height']].values
        else:
            tracker_bboxes = np.empty((0, 4))
        
        # IoUを計算
        ious = compute_iou_matrix(gt_bboxes, tracker_bboxes)
        
        # accumulatorに記録
        acc.update(
            gt_ids.astype(int),
            tracker_ids.astype(int),
            ious
        )
    
    return acc


def compute_iou(box1, box2):
    """
    2つのバウンディングボックスのIoUを計算
    
    Args:
        box1: [x, y, width, height]
        box2: [x, y, width, height]
    
    Returns:
        float: IoU値 (0.0 - 1.0)
    """
    x1_min = box1[0]
    y1_min = box1[1]
    x1_max = box1[0] + box1[2]
    y1_max = box1[1] + box1[3]
    
    x2_min = box2[0]
    y2_min = box2[1]
    x2_max = box2[0] + box2[2]
    y2_max = box2[1] + box2[3]
    
    # 交差領域
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # 和領域
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_iou_matrix(gt_bboxes, tracker_bboxes, iou_threshold=0.5):
    """
    GT と Tracker のバウンディングボックス行列のIoU行列を計算
    
    Args:
        gt_bboxes: GT のバウンディングボックス配列
        tracker_bboxes: Tracker のバウンディングボックス配列
        iou_threshold: IoU のしきい値
    
    Returns:
        np.ndarray: IoU行列（距離行列）
    """
    n_gt = len(gt_bboxes)
    n_tracker = len(tracker_bboxes)
    
    if n_gt == 0 or n_tracker == 0:
        return np.zeros((n_gt, n_tracker))
    
    ious = np.zeros((n_gt, n_tracker))
    
    for i in range(n_gt):
        for j in range(n_tracker):
            iou = compute_iou(gt_bboxes[i], tracker_bboxes[j])
            if iou >= iou_threshold:
                ious[i, j] = 1.0 - iou  # motmetrics では距離が小さいほど良いため反転
            else:
                ious[i, j] = 1.0  # IoU が しきい値未満の場合は距離 1.0（マッチしない）
    
    return ious


def print_metrics(acc):
    """
    計算したメトリクスを表示
    
    Args:
        acc: MOTAccumulator
    """
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames',
            'mota',
            'motp',
            'idp',
            'idr',
            'idf1',
            'num_matches',
            'num_switches',
            'num_false_positives',
            'num_misses',
            'precision',
            'recall',
            'num_detections',
            'num_objects',
        ],
        name='MOT Challenge Metrics'
    )
    
    print("=" * 80)
    print("MOT Challenge Evaluation Metrics")
    print("=" * 80)
    print(summary)
    print("=" * 80)
    
    return summary


def main():
    """
    メイン処理
    """
    # コマンドライン引数からファイルパスを取得
    if len(sys.argv) < 3:
        print("使用方法: python mot_check.py <gt_file> <tracker_file> [--detailed]")
        print("例: python mot_check.py gt.txt tracker_output.txt")
        print("例: python mot_check.py gt.txt tracker_output.txt --detailed")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    tracker_file = sys.argv[2]
    detailed = len(sys.argv) > 3 and sys.argv[3] == '--detailed'
    
    # ファイルが存在することを確認
    if not os.path.exists(gt_file):
        print(f"エラー: GT ファイルが見つかりません: {gt_file}")
        sys.exit(1)
    
    if not os.path.exists(tracker_file):
        print(f"エラー: Tracker ファイルが見つかりません: {tracker_file}")
        sys.exit(1)
    
    print(f"GT ファイル: {gt_file}")
    print(f"Tracker ファイル: {tracker_file}")
    print()
    
    # メトリクスを計算
    acc = compute_mot_metrics(gt_file, tracker_file)
    
    # メトリクスを表示
    summary = print_metrics(acc)
    
    # MOTA と IDF1 を表示
    mota = summary.loc['MOT Challenge Metrics', 'mota']
    idf1 = summary.loc['MOT Challenge Metrics', 'idf1']
    motp = summary.loc['MOT Challenge Metrics', 'motp']
    idp = summary.loc['MOT Challenge Metrics', 'idp']
    idr = summary.loc['MOT Challenge Metrics', 'idr']
    precision = summary.loc['MOT Challenge Metrics', 'precision']
    recall = summary.loc['MOT Challenge Metrics', 'recall']
    num_matches = summary.loc['MOT Challenge Metrics', 'num_matches']
    num_switches = summary.loc['MOT Challenge Metrics', 'num_switches']
    num_false_positives = summary.loc['MOT Challenge Metrics', 'num_false_positives']
    num_misses = summary.loc['MOT Challenge Metrics', 'num_misses']
    
    print()
    print("=" * 80)
    print("主要な評価指標")
    print("=" * 80)
    print(f"MOTA (Multi-Object Tracking Accuracy):  {mota:7.2%}")
    print(f"IDF1 (ID F1 Score):                     {idf1:7.2%}")
    print()
    print("詳細メトリクス:")
    print(f"  MOTP (Tracking Precision):             {motp:7.2%}")
    print(f"  IDP  (ID Precision):                   {idp:7.2%}")
    print(f"  IDR  (ID Recall):                      {idr:7.2%}")
    print(f"  Precision:                             {precision:7.2%}")
    print(f"  Recall:                                {recall:7.2%}")
    print()
    print("カウント:")
    print(f"  Total Matches:                         {int(num_matches):5d}")
    print(f"  ID Switches:                           {int(num_switches):5d}")
    print(f"  False Positives:                       {int(num_false_positives):5d}")
    print(f"  Misses:                                {int(num_misses):5d}")
    print("=" * 80)
    
    if detailed:
        print()
        print("全メトリクス詳細:")
        print(summary.to_string())
        print("=" * 80)


if __name__ == '__main__':
    main()

