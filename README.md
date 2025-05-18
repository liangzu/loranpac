# LoRanPAC (ICLR 2025)
The repo accompanies [our paper](https://arxiv.org/abs/2410.00645) accepted to ICLR 2025. The bib entry is:

    @inproceedings{
        Peng-ICLR2025,
        title={{LoRanPAC}: Low-rank Random Features and Pre-trained Models for Bridging Theory and Practice in Continual Learning},
        author={Liangzu Peng and Juan Elenter and Joshua Agterberg and Alejandro Ribeiro and Rene Vidal},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=bqv7M0wc4x}
    }





## Instructions

The implementation is based on [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT). Their repo has instructions to run the code. In addition to that, the user needs to specify the dataset path in **utils/data.py**.

Our method is implemented in **models/tsvd.py** and **models/tsvd_adapter.py**.

- **models/tsvd.py** implements the LoRanPAC method as described in the paper;
- **models/tsvd_adapter.py** implements the LoRanPAC method with first-session adaptation.

The hyperparameters used in the experiments are in the folder **exps/**.

Note that the method name was originally TSVD, and is now changed to LoRanPAC (May 17, 2025).


## Contact
If you have any comments, questions about the repo, paper, and continual learning in general, feel free to send an email to me.
