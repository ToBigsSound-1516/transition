from flask import Flask, request, send_file
from flask_cors import CORS
import os
from time import time

from model import Model
from train import mix
import torch

app = Flask(__name__)
CORS(app)

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
    print(req)
    if req is None or len(req)== 0:
        args = Args("./test_mid/DontLookBackinAnger.mid", "./test_mid/ThinkOutLoud.mid",
                    16*16, 64*16,
                    "./result/result.mid")
    else:
        # TODO
        args = Args(os.path.join("./test_mid", req["midi1"]), os.path.join("./test_mid", req["midi2"]),
                    int(req["start1"]), int(req["start2"]),
                    os.path.join("./result/{}.mid".format(req["username"])))
    try:
        start = time()
        mix(args, model)
        duration = time() - start
    except:
        return "Generation Error"

    # TODO
    if os.path.exists(args.midi_save_dir):
        return send_file(args.midi_save_dir)
    else:
        return "Error"


if __name__ == "__main__":
    model = Model(128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO
    model.load_state_dict(torch.load("model.pt", map_location = device))

    app.run(host="0.0.0.0", port=1516)


