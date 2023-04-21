#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import subprocess


# [[[ ARG PARSE ]]]
parser = argparse.ArgumentParser(description='Process a yaml file.')
parser.add_argument('yaml', help='The input yaml file.')
args = parser.parse_args()

# [[[ Configure ]]]
fl_yaml = args.yaml
basename_yaml = fl_yaml[:fl_yaml.rfind('.yaml')]

# Load the YAML file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# Access the values
# ___/ PeakNet model \___
timestamp = config['timestamp']
epoch     = config['epoch'    ]

# ___/ Experimental data \___
# Psana...
exp = config['exp']
run = config['run']

# ___/ Output \___
dir_results = config["dir_results"]


cmd = \
f'''
indexamajig                                          \
    -j 10                                            \
    -i {dir_results}/{basename_yaml}.peaks.lst             \
    -g ./{exp}/cwang31/psocake/r{run:04d}/.temp.geom \
    --peaks=cxi                                      \
    --int-radius=3,4,5                               \
    --indexing=mosflm,xds,xgandalf                   \
    -o {dir_results}/{basename_yaml}.peaks.stream          \
    --temp-dir={dir_results}                         \
    --tolerance=5,5,5,1.5                            \
    ## --pdb=/reg/data/ana03/scratch/cwang31/pf/cxi04216.cell
'''

output_file = os.path.join(dir_results, f"{basename_yaml}.log")

# Run the subprocess with a pipe for stdout
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell = True)

# Open the file to which you want to write the standard output
with open(output_file, "w") as fh:
    # Read the output from the pipe in chunks
    for line in iter(process.stdout.readline, b''):
        # Convert the bytes to a string and write it to the file
        fh.write(line.decode())
        fh.flush()

    # Wait for the subprocess to complete
    process.wait()
