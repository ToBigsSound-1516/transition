import os
import gdown
import tarfile
import sys
import parmap
from functools import reduce

from const import *
from util import midi_to_array


def download_data(data_path):
    print("Downloading data in the", data_path)
    os.makedirs(data_path, exist_ok = True)
    gdown.download("https://drive.google.com/uc?id=1yz0Ma-6cWTl6mhkrLnAVJ7RNzlQRypQ5", os.path.join(data_path, "lpd_5_cleansed.tar.gz"))
    gdown.download("https://drive.google.com/uc?id=1hp9b_g1hu_dkP4u8h46iqHeWMaUoI07R", os.path.join(data_path, "id_lists_amg.tar.gz"))
    gdown.download("https://drive.google.com/uc?id=1mpsoxU2fU1AjKopkcQ8Q8V6wYmVPbnPO", os.path.join(data_path, "id_lists_lastfm.tar.gz"))
    for tar_name in ["lpd_5_cleansed.tar.gz", "id_lists_amg.tar.gz", "id_lists_lastfm.tar.gz"]:
        tar = tarfile.open(os.path.join(data_path, tar_name))
        tar.extractall(data_path)
        tar.close()
        os.remove(os.path.join(data_path, tar_name))
    print("Download Completed.")

def get_id_list(data_path):
    id_list = []
    amg_path = os.path.join(data_path, "amg")
    for path in os.listdir(amg_path):
        filepath = os.path.join(amg_path, path)
        if os.path.isfile(filepath):
            with open(filepath) as f:
                id_list.extend([line.rstrip() for line in f])
    id_list = list(set(id_list))
    return id_list

def msd_id_to_dirs(msd_id):
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def id2sample(id_list, dataset_root):
    data = []
    for msd_id in id_list:
        song_dir = os.path.join(dataset_root, msd_id_to_dirs(msd_id))
        filename = os.path.join(song_dir, os.listdir(song_dir)[0])
        song_arr = midi_to_array(filename)
        n_total_measures = song_arr.shape[-2] // measure_resolution
        candidate = n_total_measures - n_measures
        target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
        for idx in np.random.choice(candidate, target_n_samples, False):
            start = idx * measure_resolution
            end = (idx + n_measures) * measure_resolution
            data.append(song_arr[:,start:end,:])
    return data

if __name__=="__main__":
    arguments = sys.argv
    assert len(arguments) > 1, "No argument"

    data_path = arguments[1]
    # assert not os.path.exists(data_path), "Data is already exists."

    # download_data(data_path)
    id_list = get_id_list(data_path)
    dataset_root = os.path.join(data_path, "lpd_5", "lpd_5_cleansed")

    print("Trying to sample {} per song.".format(n_samples_per_song))
    num_cores = os.cpu_count()
    print("# of cores: {}".format(num_cores))
    splited_data = [x.tolist() for x in np.array_split(id_list, num_cores)]
    result = parmap.map(id2sample, splited_data, dataset_root, pm_pbar=True, pm_processes=num_cores)
    data = list(reduce(lambda x, y: x + y, result))

    random.shuffle(data)
    data = np.stack(data)
    print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
    print(f"Data shape : {data.shape}")
    np.save(os.path.join(data_path, "data"), data)
    print("Sampled data {} is saved completely.".format(os.path.join(data_path, "data.npy")))

