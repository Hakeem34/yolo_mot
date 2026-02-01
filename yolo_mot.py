#/* YOLO11物体検知＆追跡テストコード */
from dataclasses import dataclass
import os
import shutil
import sys
import subprocess
import csv
import openpyxl

from ultralytics import YOLO

@dataclass 
class ObjectInfo:
    FrameIndex: int
    ClassId: int
    Score: float
    Xyxy: list
    TrackId: int

@dataclass
class FrameInfo:
    FrameIndex: int
    FrameTimeStamp: int
    ObjecInfos: list


# グローバル変数定義
g_target_file = ''
g_target_dir = ''
g_tracker = ''
g_save = False
g_model = None
g_out_png = False
g_skip_frames = 0
g_frame_count = 0
g_score_thresh = 0.25
g_object_list = []

def make_sub_dir(dir_path, sub_dir_name):
    # サブディレクトリを作成
    sub_dir_path = os.path.join(dir_path, sub_dir_name)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    else:
        print(f'sub_dir_path already exists: {sub_dir_path}')
        shutil.rmtree(sub_dir_path)
        os.makedirs(sub_dir_path)

    return sub_dir_path

def mp4_to_pnmg(input_mp4_file):
    # mp4ファイルをpng連番ファイルに変換
    output_dir = make_sub_dir(os.path.dirname(input_mp4_file), os.path.basename(input_mp4_file).replace('.mp4', '_png'))
    print(f'output_dir: {output_dir}')
    output_png_path = os.path.join(output_dir, 'frame_%05d.png')
    command = f'ffmpeg -i {input_mp4_file} -vf "fps=30" -start_number 0 {output_png_path}'
    print(f'command: {command}')
    subprocess.run(command, shell=True)

    return output_dir

def object_detect_png_file(file_path):
    # 画像ファイルに対して物体検知を実行
    global g_frame_count

    if g_frame_count % (g_skip_frames + 1) != 0:
        g_frame_count += 1
        return None

    results = g_model.predict(source=file_path, show=True, conf=g_score_thresh, save=g_save)
    boxes = results[0].boxes
    for box in boxes:
        print(f'class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}')
        
        obj_info = ObjectInfo(
            FrameIndex=g_frame_count,
            ClassId=int(box.cls),
            Score=float(box.conf),
            Xyxy=box.xyxy[0].tolist(),
            TrackId=None
        )
        g_object_list.append(obj_info)

    g_frame_count += 1
    return results

def object_track_png_file(file_path, tracker):
    # 画像ファイルに対して物体検知＆追跡を実行
    global g_frame_count

    if g_frame_count % (g_skip_frames + 1) != 0:
        g_frame_count += 1
        return None

    results = g_model.track(source=file_path, show=False, conf=g_score_thresh, persist=True, save=g_save, tracker=tracker, vid_stride=g_skip_frames+1)
    boxes = results[0].boxes
    for box in boxes:
        if box.id is not None:
            track_id = int(box.id)
        else:
            track_id = -1

        print(f'class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: {track_id:03}')
        obj_info = ObjectInfo(
            FrameIndex=g_frame_count,
            ClassId=int(box.cls),
            Score=float(box.conf),
            Xyxy=box.xyxy[0].tolist(),
            TrackId=int(box.id) if box.id is not None else -1
        )
        g_object_list.append(obj_info)

    g_frame_count += 1
    return results

def object_detect_mp4_file(file_path):
    # 動画ファイルに対して物体検知＆追跡を実行
    results = g_model.predict(source=file_path, show=False, conf=g_score_thresh, save=g_save, vid_stride=g_skip_frames+1)
    for index, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            obj_info = ObjectInfo(
                FrameIndex=index,
                ClassId=int(box.cls),
                Score=float(box.conf),
                Xyxy=box.xyxy[0].tolist(),
                TrackId=None
            )
            g_object_list.append(obj_info)
            print(f'result[{index:04}]: class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}') 


    return results

def object_track_mp4_file(file_path,tracker):
    # 動画ファイルに対して物体検知＆追跡を実行
    results = g_model.track(source=file_path, show=False, conf=g_score_thresh, persist=True, save=g_save, tracker=tracker)
    for index, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            if box.id is not None:
                track_id = int(box.id)
            else:
                track_id = -1

            print(f'result[{index:04}]: class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: {track_id:03}')
            obj_info = ObjectInfo(
                FrameIndex=index,
                ClassId=int(box.cls),
                Score=float(box.conf),
                Xyxy=box.xyxy[0].tolist(),
                TrackId=track_id
            )
            g_object_list.append(obj_info)

    return results

def handle_target_dir(target_dir):
    print(f'handle_target_dir: {target_dir}')
    for file_name in os.listdir(target_dir):
        print(f'file_name: {file_name}')
        target_file = os.path.join(target_dir, file_name)
        if (os.path.isfile(target_file)):
            handle_target_file(target_file)

    return

def handle_target_file(target_file):
    global g_tracker

    print(f'handle_target_file: {target_file}') 
    if (target_file.endswith('.mp4')):
        if (g_out_png):
            # mp4ファイルをpng連番ファイルに変換
            png_dir = mp4_to_pnmg(target_file)
            # 変換したpng連番ファイルに対して物体検知＆追跡を実行
            handle_target_dir(png_dir)
        else:
            if (g_tracker == ''):
                object_detect_mp4_file(target_file)
            else:
                object_track_mp4_file(target_file, tracker=g_tracker)
    elif (target_file.endswith('.png')):
        if (g_tracker == ''):
            object_detect_png_file(target_file)
        else:
            object_track_png_file(target_file, tracker=g_tracker)

    return


def check_command_line_option():
    global g_target_file
    global g_target_dir
    global g_tracker
    global g_save
    global g_out_png
    global g_skip_frames
    global g_score_thresh

    # Parse arguments.
    args = sys.argv[1:]

    while args:
        # Take an argument.
        arg = args.pop(0)

        if (os.path.isfile(arg)):
            g_target_file = arg
        elif (os.path.isdir(arg)):
            g_target_dir = arg
        elif (arg.startswith('--tracker=')):
            g_tracker = arg.split('=')[1]   
        elif (arg.startswith('--save')):
            g_save = True
        elif (arg.startswith('--png')):
            g_out_png = True
        elif (arg.startswith('--skip_frames')):
            g_skip_frames = int(arg.split('=')[1])
        elif (arg.startswith('--score_thresh')):
            g_score_thresh = float(arg.split('=')[1])
        else:
            print(f'invalid arg : {arg}')


def output_results_to_csv(output_csv_file):
    global g_object_list

    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['FrameIndex', 'ClassId', 'Score', 'Xyxy', 'TrackId']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for obj in g_object_list:
            writer.writerow({
                'FrameIndex': obj.FrameIndex,
                'ClassId': obj.ClassId,
                'Score': obj.Score,
                'Xyxy': obj.Xyxy,
                'TrackId': obj.TrackId
            })

    print(f'Results saved to {output_csv_file}')
    return

def output_results_to_excel(output_excel_file):
    global g_object_list

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "DetectionResults"

    # ヘッダー行の作成
    headers = ['FrameIndex', 'ClassId', 'Score', 'TopLeftX', 'TopLeftY', 'BottomRightX', 'BottomRightY', 'TrackId']
    sheet.append(headers)

    # データ行の作成
    for obj in g_object_list:
        row = [
            obj.FrameIndex,
            obj.ClassId,
            f'{obj.Score*100:.1f}%',
            int(obj.Xyxy[0]),
            int(obj.Xyxy[1]),
            int(obj.Xyxy[2]),
            int(obj.Xyxy[3]),
            obj.TrackId
        ]
        sheet.append(row)

    workbook.save(output_excel_file)
    print(f'Results saved to {output_excel_file}')
    return


def main():
    global g_model
    global g_target_file

    check_command_line_option()

    # YOLOv11のモデルをロード
    g_model = YOLO('yolo11m.pt')

    if (g_target_file != ''):
        handle_target_file(g_target_file)
        output_csv_file = os.path.splitext(g_target_file)[0] + '_results.csv'
    elif (g_target_dir != ''):
        handle_target_dir(g_target_dir)
        output_csv_file = os.path.splitext(g_target_dir)[0] + '_results.csv'

    # 結果をCSVファイルに保存
    output_results_to_csv(output_csv_file)
    output_excel_file = os.path.splitext(output_csv_file)[0] + '.xlsx'
    output_results_to_excel(output_excel_file)
    return


if __name__ == "__main__":
    main()



