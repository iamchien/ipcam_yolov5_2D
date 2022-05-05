from fastapi import FastAPI
from multiprocessing import Process
from uvicorn import Config, Server
import os

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']

txt_dir = './inference/txt_output'
txt_path = os.path.join(txt_dir, 'output.txt')

class UvicornServer(Process):

    def __init__(self, config: Config):
        super().__init__()
        self.server = Server(config=config)
        self.config = config

    def stop(self):
        self.terminate()

    def run(self, *args, **kwargs):
        self.server.run()

# FastAPI 
app = FastAPI(
    title="Serving YOLO",
    description="""Visit port 8088/docs for the FastAPI documentation.""",
    version="0.0.1",
)

class Robotic_API():
    def __init__(self, x = 0, y = 0, w = 0, h = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def init_server(self):
        config = Config("api:app", host="127.0.0.1", port=8000, reload=True)
        self.instance = UvicornServer(config=config)
        self.instance.start()

    @app.get("/v1")
    async def mot_yolov5():
        try:
            objects = dict(); frames = dict()
            with open(txt_path, 'r') as f:
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
                    frames[frameName] = objects

            return frames
        except:
            return "Error"