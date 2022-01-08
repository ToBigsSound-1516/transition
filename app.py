from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
from time import time

from model import Model
from train import mix
from mashup import get_dtw_similarity, get_distribution_similarity
import torch

app = Flask(__name__)
CORS(app)

similar_dict = {"dtw": get_dtw_similarity, "distribution": get_distribution_similarity}

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
        if "username" not in req.keys():
            req["username"] = "temp"
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

@app.route("/mashup", methods=["GET", "POST"])
def recommend():
    req = request.get_json()
    print(req)
    
    if req is None or len(req)==0:
        return "No Argument Error"
    
    midi1, midi2, mode = os.path.join("./test_mid", req["midi1"]), os.path.join("./test_mid", req["midi2"]), req["mode"]
    
    if not os.path.exists(midi1) or not os.path.exists(midi2):
        return "No Midi File Error"
    
    if mode not in similar_dict.keys():
        return "No mode error"
    
    if "n_rank" not in req:
        req["n_rank"] = 1
    else:
        req["n_rank"] = int(req["n_rank"])

    if req["n_rank"] < 1 or req["n_rank"] > 100:
        req["n_rank"] = 1
    
    candidate = similar_dict[mode](midi1, midi2)[:req["n_rank"]]
    print(candidate)
    
   # return "start1: {} start2: {} score: {:.4f}".format(candidate[0][0]*16, candidate[0][1]*16, candidate[1])
    return list_to_json(candidate)

def list_to_json(candidates):
    response = []
    for idx, row in enumerate(candidates):
        start1, start2, score = int(row[0][0]), int(row[0][1]), float(row[1])
        response.append({"start1": start1, "start2": start2, "score": score})
    return jsonify(response)

if __name__ == "__main__":
    model = Model(128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO
    model.load_state_dict(torch.load("model.pt", map_location = device))

    app.run(host="0.0.0.0", port=1516)


