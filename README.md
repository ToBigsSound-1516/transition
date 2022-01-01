# Transition (가제)
## Prerequisites
### Set up
```
git clone https://github.com/ToBigsSound-1516/transition.git
cd trainsition
pip install -r requirements.txt
```

### Download Lakh Pianoroll Dataset
```
python3 prepare_data.py ./your_data_path
```
## Training
```
python3 main.py --train --ckpoint ./your_checkpoint_path --data_path ./your_data_path --n_steps 100000
```

**Base Arguments**

`--ckpoint`: str, default = "./ckpoint". checkpoint will be saved here. If it is not .pt file, the latest checkpoint will be loaded.

`--latent_dim`: int, default = 128.

`--cpu`: Use this option to unable CUDA training. When there is no GPU in your environment, this option will be used automatically.


**Training Arguments**

`--train`: Use this option to run the train script.

`--data_path`: str, default = "./data". There must be a `data.npy` in your data_path. Please refer the `prepare_data.py`.

`--lr`: float, default = 1e-5. learning rate.

`--batch_size`: int, default = 128.

`--n_epochs`: int, default = 100. 

`--ckpoint_interval`: int, default = 10.

`--save_sample`: Use this option to save some midi samples while training.

## Mixing
```
python3 main.py --ckpoint ./your_checkpoint_path --midi_path1 ./first_midi_path --midi_path2 ./second_midi_path --start1 100 --start2 200 --midi_save_dir ./mixed_midi_will_be_saved_here
```

**Mixing Arguments**

`--midi_path1`, `--midi_path2`: str. The midi files to mix.

`--midi_save_dir`: str, default = "./data/mixed". Mixed midi file will be saved here.

`--start1`, `--start2`: int. Mixed point.

`--mix_margin`: int, default = 4 * measure_resolution. Mixed file will be saved with the margin.

# Backend

## How to run flask
```
python3 app.py
```
