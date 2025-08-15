# RIS-Assisted NOMA with Partial CSI and Mutual Coupling: A Machine Learning Approach

![GitHub](https://img.shields.io/github/license/bilepeng/risnet_noma_partial_csi)

This repository is accompanying the paper "RIS-Assisted NOMA with Partial CSI and Mutual Coupling: A Machine Learning Approach" (Bile Peng,
Karl-Ludwig Besser, Shanpu Shen, Finn Siegismund-Poschmann, Ramprasad Raghunath, Daniel Mittleman, Vahid Jamali, and Eduard A. Jorswieck, IEEE Globecom 2025.


## File List

The following files are provided in this repository:

- `train.py`: Main file to train the model.
- `core.py`: Core classes of RISnet and data loader.
- `util.py`: Utility functions.
- `params.py`: Parameters.



## Usage

Make sure that you have [Python3](https://www.python.org/downloads/) and all
necessary libraries installed on your machine.

Create a folder `data` and download files from https://drive.google.com/file/d/1cXh4ME7bmY7a7llOj4Np2qBakrI2eHwH/view
and put them in the folder.

Create a folder `results`, where the training and testing results will be saved.

Run `python train.py` with the following arguments to train the model:

```bash
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name partial_0
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training.pt --testingchannelpath data/channels_ris_rx_testing.pt --name full_0
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name partial_p
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_5e-5.pt --testingchannelpath data/channels_ris_rx_testing_5e-5.pt --name full_p
python3 train.py --record True  --tsnr=1e12  --partialcsi True --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name partial_iid
python3 train.py --record True  --tsnr=1e12  --partialcsi False --trainingchannelpath data/channels_ris_rx_training_iid.pt --testingchannelpath data/channels_ris_rx_testing_iid.pt --name full_iid
```

You need Tensorboard to illustrate the results.


## Acknowledgements

This research was supported by the Federal Ministry of Education and Research
Germany (BMBF) as part of the 6G Research and Innovation Cluster (6G-RIC) under
Grant 16KISK031
and German Research Foundation under grant ML4RIS (566937681).


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
