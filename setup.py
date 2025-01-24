from setuptools import setup

setup(
    name                 = "electron_density",
    version              = "2020.0",
    packages             = ['electron_density'],
    scripts              = ['electron_density/make_noisy_diffraction_volume.py', 'electron_density/make_diffraction_volume.py', 'electron_density/electron_density_from_pdb.py']
    )
