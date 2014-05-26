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

def rotation_matrix(axis,theta):
    axis = axis/math.sqrt(np.dot(axis,axis))
    a = math.cos(theta/2)
    b,c,d = -axis*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def run(pos = 0, shape = (2048, 2048), angles = [0.0, 0.0, 0.0], unit_tile = [5,5,5], unit_dim = [10.0e-10, 10.0e-10, 10.0e-10]):
    # Geometry stuff
    X         = [1000.0e-10, 1000.0e-10]
    spacing_x = np.array(X) / np.array(shape, np.float64) 
    spacing_q = 1 / np.array(X)

    h         = 6.62606957e-34
    c         = 299792458.0
    E_ev      = 22.0e3
    E         = E_ev * 1.60217657e-19
    lamb      = c * h / E

    du        = 55.0e-6
    #Z         = du * shape[0] / (2.0 * theta_max)
    Z         = du * np.array(X).max()/ lamb 
    theta_max = np.arctan(du * shape[0] / (2 * Z))
    Q_max     = theta_max / lamb    

    # Make a list of atomic sites
    # Make a unit cell of 
    #number_of_atoms_unit = 3
    #unit   = np.random.rand(number_of_atoms_unit, 3)
    #cubic
    unit = []
    unit.append([0, 0, 0])
    #unit.append([1, 0, 0])
    #unit.append([0, 1, 0])
    #unit.append([1, 1, 0])
    #unit.append([0, 0, 1])
    #unit.append([1, 0, 1])
    #unit.append([0, 1, 1])
    #unit.append([1, 1, 1])

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

    # Rotate the atomic sites about the z-axis
    # Randomply select orientation
    # theta_x, theta_y, theta_z = 0.125 * np.pi * np.random.random(3)
    theta_x, theta_y, theta_z = angles
    #theta_z = np.pi/3
    #theta_x = np.pi/1.5
    #theta_y = np.pi/10
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([0, 0, 1]), theta_z), np.transpose(atomic_sites)) )
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([0, 1, 0]), theta_x), np.transpose(atomic_sites)) )
    atomic_sites = np.transpose( np.dot(rotation_matrix(np.array([1, 0, 0]), theta_y), np.transpose(atomic_sites)) )

    # make a convergent beam illumination
    qx, qy = bg.make_xy(shape, spacing=[1/X[0], 1/X[1]])
    q      = np.sqrt(qx**2 + qy**2)
    q_max  = 30.0e-3 / lamb
    mask   = (q < q_max)
    probeF = np.array(mask, dtype=np.complex128)
    # add defocus
    df     = 3000.0e-10
    qy, qx = bg.make_xy(shape, origin=[0,0], spacing = [1/X[0], 1/X[1]])
    expF   = np.exp(-1J * np.pi * lamb * df * (qx**2 + qy**2))
    probeF *= expF
    probeR = bg.ifft2(probeF)

    def probe_z(z):
        expF = np.exp(-1J * np.pi * lamb * z * (qx**2 + qy**2))
        return bg.ifft2(probeF * expF)

    # make the potential at an atomic site (Fourier space)
    interaction = 1.0e-1
    qy2, qx2 = bg.make_xy(shape, origin=[0,0], spacing = [1.0e-10/X[0], 1.0e-10/X[1]])
    V_f      = interaction * np.exp(- np.pi**2 * 4.0 * (qy2**2 + qx2**2))

    n, m = bg.make_xy(shape)
    def V_yxz(atomic_site):
        xi = atomic_site[1]/spacing_x[1] 
        yi = atomic_site[0]/spacing_x[0]
        if np.abs(xi) > shape[1]/2 or np.abs(yi) > shape[0]/2 :
            return 0.0
        ramp = np.exp(-2J * np.pi * (n * yi / shape[0] + m * xi / shape[1]))
        return 1J*bg.ifft2(ramp * V_f)

        
    V_proj = np.zeros(shape, np.complex128)
    exitF  = np.zeros(shape, np.complex128)
    for i in range(len(atomic_sites)):
        print len(atomic_sites), i
        #probe_i = probe_z(atomic_sites[i][2])
        V_i     = V_yxz(atomic_sites[i])
        #exitF  += bg.fft2(V_i * probe_i)
        V_proj += V_i
    #exitF  += probeF
    #
    return V_proj

if __name__ == '__main__':
    pos = 0
    angles_xyz = [0.0, 0.0, 0.0]
    V_proj = run(pos, angles)
    #
    # output
    #bg.binary_out(V_proj, 'V_proj_'+str(pos)+'_'+str(theta_x)+'_'+str(theta_y)+'_'+str(theta_z), dt=np.complex128, appendDim=True) 
