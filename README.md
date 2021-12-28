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
**Arguments**

`--train`: Use this option to run the train script.

`--data_path`: str, default = "./data". There must be a `data.npy` in your data_path. Please refer the `prepare_data.py`.

`--ckpoint`: str, default = "./ckpoint". checkpoint will be saved here.

`--latent_dim`: int, default = 128.

`--n_epochs`: int, default = 100. 

`--ckpoint_inteval`: int, default = 10.

`--lr`: float, default = 1e-5. learning rate.

`--batch_size`: int, default = 128.

`--save_sample`: Use this option to save some midi samples while training.

`--cpu`: Use this option to unable CUDA training. When there is no GPU in your environment, this option will be used automatically.



