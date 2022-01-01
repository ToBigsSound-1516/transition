from flask import Flask, request, jsonify
import os
from time import time

from model import Model
from train import mix
import torch

app = Flask(__name__)

class Args:
    def __init__(self, midi_path1, midi_path2, start1, start2, midi_save_dir):
        self.midi_path1 = midi_path1
        self.midi_path2 = midi_path2
        self.start1 = start1
        self.start2 = start2
        self.mix_margin = -1
        self.midi_save_dir = midi_save_dir
        self.device = device

@app.route("/")
def hello():
    return "Flask is Running!"


@app.route("/dj", methods=["GET",'POST'])
def dj():
    req = request.get_json()
    if req is None or len(req)== 0:
        args = Args("./test_mid/DontLookBackinAnger.mid", "./test_mid/ThinkOutLoud.mid",
                    100, 200,
                    "./result/result.mid")
    else:
        # TODO
        args = Args(os.path.join("./test_mid", req["midi1"]), os.path.join("./test_mid", req["midi2"]),
                    int(req["start1"]), int(req["start2"]),
                    os.path.join("./result", req["username"]))
    try:
        start = time()
        mix(args, model)
        duration = time() - start
    except:
        return "Error"

    # TODO
    if os.path.exists(args.midi_save_dir):
        return "True\t{:.2f}sec".format(duration)
    else:
        return "Error"


if __name__ == "__main__":
    model = Model(128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO
    model.load_state_dict(torch.load("model.pt", map_location = device))

    app.run()


