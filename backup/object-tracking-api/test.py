import json
import timeit

start_time = timeit.default_timer()

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']

objects = dict(); frames = dict()
video_txt_path = 'deepsort_yolov5/inference/output/video.txt'
with open(video_txt_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        temp = line.split()
        clsName = names[int(temp[2])]
        key = clsName + temp[3]
        x, y, w, h = map(float,temp[4:])
        value = {
            'x': x,
            'y': y,
            'w': w,
            'h': h
            }
        
        if not clsName in objects:
            objects[clsName] = dict()
        objects[clsName][key] = value

        frameName = 'frame' + temp[0]
        if not frameName in frames:
            frames[frameName] = dict()
        frames[frameName] = objects

    with open("test.json", "a") as studs:
        json.dump(frames, studs, indent=4)
        print("JSON Created.......")

end_time = timeit.default_timer()
print("Time: ", end_time - start_time)
