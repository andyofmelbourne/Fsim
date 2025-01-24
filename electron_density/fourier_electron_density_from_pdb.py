import argparse

description = "Calculate the Fourier transform of the number of electrons per cubic angstrom (unitless) from the molecule decribed in a pdb file. Data is writen as a python pickle to stdout."
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('pdb_id', type=str, \
                    help="protein database ID, e.g. 1sx4")
parser.add_argument('--dq', type=float, default=1.,\
                    help="side length of q-space voxel cube in inverse angstroms")
parser.add_argument('--len', type=int, default=32,\
                    help="side length of diffraction volume in voxels")
parser.add_argument('--B_offset', type=float, default=20.,\
                    help="Temperature factor for the molecule, such that B_offset = 8 pi^2 (mean_squared atomic displacement)")
#parser.add_argument('--symmetry', type=str,\
#                    help="apply symmetry operators to atomic coordinates before rendering: 'D6'")

args = parser.parse_args()

# delay these imports to speed up argparsing
from density_calculator_intelHD import render_Fourier_molecule_from_pdb
import pickle
import sys
import numpy as np


if __name__ == '__main__':
    shape = 3*(args.len,)
    F = render_Fourier_molecule_from_pdb(args.pdb_id, args.dq, shape, B_offset=args.B_offset)
    
    # pip pickled data to stdout
    pickle.dump({'diffraction_volume' : np.abs(F)**2, 'dq': args.dq}, sys.stdout.buffer)


