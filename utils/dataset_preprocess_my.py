import subprocess

from statistics import mean



def process_core50():
    root_dir = '/shared_data/p_vidalr/LP/core50_128x128/'

    all_cmd = ' '
    for i in range(50):
        oidx = i + 1

        str = f'cp -r {root_dir}s7/o{oidx}/* {root_dir}test_3_7_10/o{oidx}\n '
        all_cmd += str

        str = f'cp -r {root_dir}s10/o{oidx}/* {root_dir}test_3_7_10/o{oidx} \n '
        all_cmd += str

    print(all_cmd)

def process_cddb():
    names = ['biggan', 'wild', 'whichfaceisreal', 'san']
    root_dir = '/shared_data/p_vidalr/LP/CDDB/'

    types = ['0_real', '1_fake']

    all_cmd = ' '
    for name in names:
        for t in types:
            str = f"cp -r {root_dir}{name}/val/{t}/* {root_dir}CDDB-hard_val/val/{t}/\n "
            all_cmd += str

    print(all_cmd)




