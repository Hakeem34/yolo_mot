#/* YOLO11物体検知＆追跡テストコード */
import os
import sys

from ultralytics import YOLO



g_target_file = ''
g_target_dir = ''
g_tracker = 'bytetrack.yaml'
g_save = False
g_model = None


def object_detect_png_file(file_path):
    # 画像ファイルに対して物体検知を実行
    results = g_model.predict(source=file_path, show=True, conf=0.5, save=g_save)
    boxes = results[0].boxes
    for box in boxes:
        print(f'class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}')

    return results

def object_track_png_file(file_path, tracker, persist, save, show):
    # 画像ファイルに対して物体検知＆追跡を実行
    results = g_model.track(source=file_path, show=show, conf=0.5, persist=persist, save=save, tracker=tracker)
    boxes = results[0].boxes
    for box in boxes:
        if box.id is not None:
            print(f'class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: {int(box.id):03}')
        else:
            print(f'class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: None')

    return results

def predict_mp4_file(file_path):
    # 動画ファイルに対して物体検知＆追跡を実行
    results = g_model.predict(source=file_path, show=False, conf=0.5, save=g_save)
    for index, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            print(f'result[{index:04}]: class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}') 

    return results

def track_mp4_file(file_path,tracker):
    # 動画ファイルに対して物体検知＆追跡を実行
    results = g_model.track(source=file_path, show=False, conf=0.5, persist=True, save=g_save, tracker=tracker)
    for index, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            if box.id is not None:
                print(f'result[{index:04}]: class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: {int(box.id):03}')
            else:
                print(f'result[{index:04}]: class_id: {int(box.cls):02}, confidence: {float(box.conf)*100:.1f}%, xyxy: {box.xyxy}, id: None')

    return results

def handle_target_dir(target_dir):
    print(f'handle_target_dir: {target_dir}')
    for file_name in os.listdir(target_dir):
        target_file = os.path.join(target_dir, file_name)
        if (os.path.isfile(target_file)):
            handle_target_file(target_file)

    return

def handle_target_file(target_file):
    global g_tracker
    print(f'handle_target_file: {target_file}') 
    if (target_file.endswith('.mp4')):
        track_mp4_file(target_file, tracker=g_tracker)
    elif (target_file.endswith('.png')):
#       object_detect_png_file(target_file)
        object_track_png_file(target_file, tracker=g_tracker, persist=True, save=False, show=False)

    return


def check_command_line_option():
    global g_target_file
    global g_target_dir
    global g_tracker
    global g_save

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
        else:
            print(f'invalid arg : {arg}')

def main():
    global g_model

    check_command_line_option()

    # YOLOv11のモデルをロード
    g_model = YOLO('yolo11m.pt')

    if (g_target_file != ''):
        handle_target_file(g_target_file)
    elif (g_target_dir != ''):
        handle_target_dir(g_target_dir)

    return


if __name__ == "__main__":
    main()



