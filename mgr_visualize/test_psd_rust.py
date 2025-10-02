#%% #* test the Gelb and Gubbins PSD calculation code implemented in Rust. 
from ase.io import read
import mofography as mgr
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
#* compute dgrid
mat_atoms = read('../examples/TestCIFs/NU-1000_S1.cif')
dgrid = mgr.dgrid_from_atoms_rust(mat_atoms, spacing=0.25)
#%% #* compute overall psd unifrom sampled across box
# if you want overall PSD for the unit cell and plotly figure with trace, 
# some default options are repeated here for clarity
psd, fig_psd = mgr.compute_psd_uniform_rust(dgrid, mat_atoms, num_samples_total=10000, bins=np.linspace(0,50,200), density=False, material='NU-1000', return_fig=True)
# to show the figure use, 
fig_psd.show( renderer='chrome') # for the chrome browser
fig_psd.write_image('psd_uniform.jpg')
# fig.show() # to show in a notebook

#* bonus: if you want sample points and pore size to compute the histogram yourself
sample_points, pore_sizes = mgr.compute_pore_sizes_uniform_rust(dgrid, mat_atoms, num_samples_total=10000)
#%% # * do periodic segmetnation
# if you want pocket wise psd, use whatever labels you want ot pass, you'll get one psd per lable
# some default options are repeated here for clarity
results = mgr.do_periodic_segmentation_rust(dgrid=dgrid, mat_atoms = mat_atoms, return_dgrid=True, spacing=0.25, extra_padding=0.0, return_padded_regions=False)
#%% #* compute pocket wise psds oneshot
regions_pbc = results['regions_pbc'] # grab the regions labels 
pocket_psds, fig_pocket_psds = mgr.compute_psds_pocketwise_rust(dgrid, regions_pbc, mat_atoms, num_samples_per_region=1000, bins= np.linspace(0,50,200), material='NU-1000', return_fig=True)
fig_pocket_psds.show() # for the notebook
fig_pocket_psds.write_image('psds_pocketwise.jpg')
# fig_pocket_psds.show( renderer='chrome') # for the chrome browser
#%% #* bonus : compute pore sizes to compute the histogram separately or just some regions, use this
sample_points,pore_sizes, sample_point_labels = mgr.compute_pore_sizes_pocketwise_rust(dgrid, regions_pbc, mat_atoms, samples_per_region=1000)
# %%
