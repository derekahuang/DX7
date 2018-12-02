from __future__ import print_function

import numpy as np
import argparse
import os
from tqdm import tqdm
import sys

PARAM_2_VAL = {
    'Algo': int,
    'Frequency': float,
    'opMode': float,
    'opFreq': float,
    'opDetune': float,
    'opRateScale': float,
}

def read_array_from_faust2plot_outfile(file):
    read = False
    params_array = []
    array = []
    values = []
    with open(file) as r:
        last_params = None
        count = 1
        for line in r:
            line = line.strip()
            if line == "":
                continue
            if "Algo" in line:
                last_params = line.split(',')
                last_params = [PARAM_2_VAL[p.split(':')[0]](p.split(':')[-1]) for p in last_params]
                if values != []:
                    array.append(values)
                    values = []
                continue
            if "Chunk Boundary" in line:
                continue
            float_value = float(line)
            values.append(float_value)
            if last_params is not None:
                params_array.append(last_params)
                last_params = None
                sys.stdout.flush()
                print("# samples collected: {}".format(count), end='\r')
                count += 1
    if values != []:
        array.append(values)

    assert (len(array) == len(params_array)), ('Received mismatched number of samples ({0}) and labels ({1})...'
        ' make sure file is formatted appropriately'.format(len(array),len(params_array)))
    return array, params_array

def  main(args):
    array = []
    params_array = []
    for file in args.infiles:
        print("reading from {} ...".format(file), end='\n')
        tmp_array, tmp_params_array = read_array_from_faust2plot_outfile(file)
        array += tmp_array
        params_array += tmp_params_array
        print('done')
    outpath_samples = os.path.join(args.outdir, 'samples')
    outpath_labels = os.path.join(args.outdir, 'labels')
    print('Writing {0} samples and {1} lables to {2}.npy and {3}.npy respectively'.format(len(array), len(params_array), outpath_samples, outpath_labels))
    np.save(outpath_samples, np.array(array))
    np.save(outpath_labels, np.array(params_array))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--outdir', default='')
    args = parser.parse_args()
    main(args)