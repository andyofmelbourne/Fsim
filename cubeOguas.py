import numpy as np
import scipy as sp
import math
import os, sys, getopt
import bagOfns as bg

def main(argv):
    try :
        opts, args = getopt.getopt(argv,"hp:",["pos="])
    except getopt.GetoptError:
      print 'test_3.py -p <pos> '
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
          print 'test_3.py -p <pos>'
          sys.exit()
      elif opt in ("-p", "--pos"):
          pos = float(arg)
    print 'pos is ', pos
    return pos

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()

def rotation_matrix(axis,theta):
    axis = axis/math.sqrt(np.dot(axis,axis))
    a = math.cos(theta/2)
    b,c,d = -axis*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def run(pos = 0, shape = (512, 512), angles = [0.0, 0.0, 0.0], unit_tile = [3,3,3], unit_dim = [10.0e-10, 10.0e-10, 10.0e-10], sigma = 1.0):
    # Geometry stuff
    X         = [100.0e-10, 100.0e-10]
    spacing_x = np.array(X) / np.array(shape, np.float64) 
    spacing_q = 1 / np.array(X)
    h         = 6.62606957e-34
    c         = 299792458.0
    E_ev      = 22.0e3
    E         = E_ev * 1.60217657e-19
    lamb      = c * h / E
    du        = 55.0e-6
    Z         = du * np.array(X).max()/ lamb 
    theta_max = np.arctan(du * shape[0] / (2 * Z))
    Q_max     = theta_max / lamb    

    #
    # Make a list of atomic sites
    #cubic
    unit = []
    unit.append([0, 0, 0])

    # unit cell dimensions
    unit_x = 10.0e-10
    unit_y = 10.0e-10
    unit_z = 10.0e-10

    # tiling of unit cells
    n_x, n_y, n_z = unit_tile
    #
    atomic_sites = []
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                for l in range(len(unit)):
                    x = (unit[l][1] + i - n_x/2 - 1) * unit_x
                    y = (unit[l][0] + j - n_y/2 - 1) * unit_y
                    z = (unit[l][2] + k - n_z/2 - 1) * unit_z
                    atomic_sites.append([y,x,z])
    atomic_sites = np.array(atomic_sites)
    atomic_sites[:, 0] -= atomic_sites[:, 0].mean() 
    atomic_sites[:, 1] -= atomic_sites[:, 1].mean() + pos * unit_x
    atomic_sites[:, 2] -= atomic_sites[:, 2].mean()

    # Rotate the atomic sites about the xyz-axis
    theta_x, theta_y, theta_z = angles
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([0, 0, 1]), theta_z), np.transpose(atomic_sites)) )
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([0, 1, 0]), theta_x), np.transpose(atomic_sites)) )
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([1, 0, 0]), theta_y), np.transpose(atomic_sites)) )

    # make the potential at an atomic site (Fourier space)
    interaction = 1.0e-1
    spacing  = [1.0e-10/X[0], 1.0e-10/X[1]]
    qy2, qx2 = bg.make_xy(shape)
    qy2      = qy2 * spacing[0]
    qx2      = qx2 * spacing[1]
    V_f      = interaction * np.exp(- np.pi**2 * 2.0 * sigma**2 * (qy2**2 + qx2**2))

    n, m = bg.make_xy(shape)
    def V_yxz(atomic_site):
        xi = atomic_site[1]/spacing_x[1] 
        yi = atomic_site[0]/spacing_x[0]
        if np.abs(xi) > shape[1]/2 or np.abs(yi) > shape[0]/2 :
            return 0.0
        ramp = np.exp(-2J * np.pi * (n * yi / float(shape[0]) + m * xi / float(shape[1])))
        return 1J*bg.ifft2(ramp * V_f)
        
    V_proj = np.zeros(shape, np.complex128)
    exitF  = np.zeros(shape, np.complex128)
    for i in range(len(atomic_sites)):
        update_progress(i / float(len(atomic_sites) - 1))
        V_i     = V_yxz(atomic_sites[i])
        V_proj += V_i
    #
    return V_proj

if __name__ == '__main__':
    tile   = np.arange(2, 10) 
    unit_tile = np.array([0, 0, 0])
    sigma  = np.linspace(0.0, 2.0, 5)
    # [111] axis 
    angles = [np.pi / 4.0, math.acos(np.sqrt(6.0) / 3.0), 0.0]
    V_proj = []
    for t in tile:
        for sig in sigma:
            print t, sig
            unit_tile.fill(t)
            V_proj.append(run(angles = angles, unit_tile=unit_tile, sigma = sig))

