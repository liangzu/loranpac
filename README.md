# LoRanPAC (ICLR 2025)
The repo accompanies [our paper](https://arxiv.org/abs/2410.00645) accepted to ICLR 2025. The bib entry is:

    @inproceedings{
        Peng-ICLR2025TSVD,
        title={{TSVD}: Bridging Theory and Practice in Continual Learning with Pre-trained Models},
        author={Liangzu Peng and Juan Elenter and Joshua Agterberg and Alejandro Ribeiro and Rene Vidal},
        booktitle={The Thirteenth International Conference on Learning Representations},
        year={2025},
        url={https://openreview.net/forum?id=bqv7M0wc4x}
    }





## Instructions

The implementation is based on [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT). Their repo has instructions to run the code. In addition to that, the user needs to specify the dataset path in **utils/data.py**.


Our method is implemented in **models/tsvd.py** and **models/tsvd_adapter.py**.

- **models/tsvd.py** implements the TSVD method as described in the paper;
- **models/tsvd_adapter.py** implements the TSVD method with first-session adaptation.

The hyperparameters used in the experiments are in the folder **exps/**.



## Contact
If you have any comments, questions about the repo, paper, and continual learning in general, feel free to send an email to me.
