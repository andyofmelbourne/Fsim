import argparse

description = "Calculate the number of electrons per cubic angstrom from the molecule decribed in a pdb file. Data is writen as a python pickle to stdout."
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('pdb_id', type=str, \
                    help="protein database ID, e.g. 1sx4")
parser.add_argument('--voxel_size', type=float, default=1.,\
                    help="side length of voxel cube in angstroms")
parser.add_argument('--B_offset', type=float, default=20.,\
                    help="Temperature factor for the molecule, such that B_offset = 8 pi^2 (mean_squared atomic displacement)")
#parser.add_argument('--symmetry', type=str,\
#                    help="apply symmetry operators to atomic coordinates before rendering: 'D6'")

args = parser.parse_args()

# delay these imports to speed up argparsing
from density_calculator_intelHD import render_molecule_from_pdb
import pickle
import sys


if __name__ == '__main__':
    den = render_molecule_from_pdb(args.pdb_id, args.voxel_size, B_offset=args.B_offset)
    
    # pip pickled data to stdout
    pickle.dump({'electron_density' : den.T, 'voxel_size': args.voxel_size}, sys.stdout.buffer)

