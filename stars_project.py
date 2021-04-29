import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd
from scipy.stats import norm

# Define the path to the data sets for a given cluster
cluster = 'M71'
data_dir = '/home/fmendez/Desktop/stars_project/data/'

# Open the data set from the CSV files
data = pd.read_csv(data_dir+cluster+'_data_v2.csv')

# Mask the NaN values from the Paralax
mask = (np.isnan(data.parallax))

# Apply the mask for the rest of the columns
source_id = data.source_id[~mask]
ra = data.ra[~mask]
dec = data.dec[~mask]
parallax = data.parallax[~mask]
parallax_err = data.parallax_error[~mask]
pmra = data.pmra[~mask]
pmra_err = data.pmra_error[~mask]
pmdec = data.pmdec[~mask]
pmdec_err = data.pmdec_error[~mask]
gmag = data.phot_g_mean_mag[~mask]
bpmag = data.phot_bp_mean_mag[~mask]
rpmag = data.phot_rp_mean_mag[~mask]

# Conver the Gaina mag to V and I mags and create a CMD
vmag = gmag - ((-0.1732*(bpmag - rpmag)**2.) - 0.006860*(bpmag - rpmag) - 0.01760)
imag = gmag - ((-0.09631*(bpmag - rpmag)**2.) + 0.7419*(bpmag - rpmag) + 0.02085)


plt.title(cluster+' CMD', size=20)
plt.xlabel(r'$V - I$', size=15)
plt.ylabel(r'$V$', size=15)
plt.plot((vmag - imag), vmag, 'k*')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Create the proper motion diagram for the the cluster
# Calculate the optimal radius of the cluster based ofn the proper motion pyplot
# Determine the mean and std RA and DEC proper Motion
mean_pmra, mean_pmdec = np.mean(pmra*np.cos(dec)), np.mean(pmdec)
std_pmra, std_pmdec = np.std(pmra*np.cos(dec)), np.std(pmdec)

print(mean_pmra, mean_pmdec)

opt_radius = np.mean([mean_pmra + 2*std_pmra - mean_pmdec, mean_pmdec + 2*std_pmdec - mean_pmdec])

print(f'The optimal radius of {cluster} is {round(opt_radius, 4)} mas/yr')

plt.title(cluster+' Proper Motion Diagram', size=20)
plt.xlabel(r'$\mu_{RA}cos\delta$ $[mas$ $y^{-1}]$', size=15)
plt.ylabel(r'$\mu_{\delta}$ $[mas$ $y^{-1}]$', size=15)
plt.plot(pmra*np.cos(dec), pmdec, 'k*')
plt.axvline(x=mean_pmra, color='lightgreen', marker='|', label=r'$\bar \mu$')
plt.axvline(x=mean_pmra+2*std_pmra, color='red', marker='|', label=r'$3\sigma$')
plt.axvline(x=mean_pmra-2*std_pmra, color='red', marker='|')
plt.axhline(y=mean_pmdec, color='lightgreen', marker='_')
plt.axhline(y=mean_pmdec+2*std_pmdec, color='red', marker='_')
plt.axhline(y=mean_pmdec-2*std_pmdec, color='red', marker='_')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# Mask the stars that are outside the PM optimal radius
pm_mask = (pmra*np.cos(dec) < np.mean(pmra*np.cos(dec)) + opt_radius) & \
          (pmra*np.cos(dec) > np.mean(pmra*np.cos(dec)) - opt_radius) & \
          (pmdec < np.mean(pmdec) + opt_radius) & (pmdec > np.mean(pmdec) - opt_radius)

# Create a new CMD with the new selected stars

plt.title(cluster+' CMD', size=20)
plt.xlabel(r'$V - I$', size=15)
plt.ylabel(r'$V$', size=15)
plt.plot((vmag[pm_mask] - imag[pm_mask]), vmag[pm_mask], 'k*')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Construct a histogram of the paralaxes
parallax = parallax[pm_mask]
n, bins, patches = plt.hist(parallax, bins=100, color='black', histtype='step', density=True)

# Fit a Gaussian to the Histogram
mu, sigma = norm.fit(parallax)
best_fit_line = norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)

plt.title(cluster+' Parallax Histogram', size=20)
plt.xlabel(r'Paralax $[mas]$', size=15)
plt.ylabel('Density Distribution', size=15)
plt.tight_layout()
plt.show()

# Calculate the distance modulus of the cluster
robust_parallax = mu
robust_parallax_error = sigma/np.sqrt(parallax.size)
distance_mod = 5.*np.log10((1./robust_parallax)*1000.) - 5.
distance_mod_error = 5.*robust_parallax_error*np.sqrt(1./(1000.*robust_parallax**2.))

print(f'The robust parallax of {cluster} is {round(robust_parallax, 4)} \
+/- {round(robust_parallax_error, 3)} mas')

print(f'The distance modulus of {cluster} is {round(distance_mod, 4)} \
+/- {round(distance_mod_error, 3)}')

# Isolate the stars based on their parallaxes
parallax_mask = (parallax > robust_parallax - 2.*sigma) & (parallax < robust_parallax + 2.*sigma)

# Calculate the absolute magnitude of the stars in the cluster asuming reddening
vmag = vmag[parallax_mask & pm_mask]
imag = imag[parallax_mask & pm_mask]

# Reddening for each cluster
# M67: 0.01 Chaboyer 1998
# NGC 188: 0.114 Meibon
# M44: 0.027 Brandt 2015
# NGC 6791: 0.15 Chaboyer 1998
# M71: 0.19 Samra_2009
EVI = 0.19

Av = 2.5*EVI
Mv = vmag - 5*np.log10((1./parallax[parallax_mask])*1000.) + 5 - Av
Mv_err = (5*parallax_err[pm_mask & parallax_mask])/parallax[parallax_mask]
print(Av)
# Calculate the V-I intrinsic
vi_intrinsic = (vmag - imag) - EVI

# Eliminate the outliers in the data by using a m - M window and eliminate larhge errors
dist_mod_mask = (Mv_err < 1.0) & ((vmag - Mv) > -25.) & ((vmag - Mv) < 25.)

# Calculate the Mv from the robust parallax
Mv_robust = vmag - 5*np.log10(1./robust_parallax) + 5 - Av

# Create the new HR diagram
Mv, Mv_err = Mv[dist_mod_mask], Mv_err[dist_mod_mask]
Mv_robust = Mv_robust[dist_mod_mask]
vi_intrinsic = vi_intrinsic[dist_mod_mask]

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title(cluster+' Intrinsic CMD', size=15)
plt.xlabel(r'$(V - I)_o$', size=15)
plt.ylabel(r'$M_V$', size=15)
plt.plot(vi_intrinsic, Mv, 'k*')
plt.gca().invert_yaxis()

plt.subplot(122)
plt.title(cluster+' Robust Intrinsic CMD', size=15)
plt.xlabel(r'$(V - I)_o$', size=15)
plt.ylabel(r'$M_V$', size=15)
plt.plot(vi_intrinsic, Mv_robust, 'k*')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Create a histogram of the Mv luminosity functions and normalize it
n, bins, patches = plt.hist(Mv, bins=50, color='black', histtype='step', density=True)

mu, sigma = norm.fit(Mv)
best_fit_line = norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line)

plt.title(cluster+' Luminosity Histogram', size=20)
plt.xlabel(r'$M_V$', size=15)
plt.ylabel('Density Distribution', size=15)
plt.tight_layout()
plt.show()
