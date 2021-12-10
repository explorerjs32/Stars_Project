import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.optimize import curve_fit
from scipy import integrate

# First, we are gou=ing to open the data foles using pandas and extract the information
# Let's define the name of the columns for each data file
falc_cols = ['height','tau','m','Temperature','turb_vel','nH','np','ne','ptot','beta','rho']
solspect_cols = ['wave','smoothed_flux','continuum_flux','smoothed_intensity','continuum_intensity']

falc_data = pd.read_csv('./falc.dat.txt',delimiter=r'\s+', header=None, names=falc_cols,engine='python',comment='#')
solspect_data = pd.read_csv('./solspect.dat.txt',delimiter=r'\s+', header=None, names=solspect_cols,engine='python',comment='#')

# Part A - Plot the temperature as a fum=nction of the height
'''
plt.title('Temperature vs Height', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.ylabel(r'Temperature $[K]$', size=15)
plt.plot(falc_data['height'], falc_data['Temperature'], 'k-')
plt.axvline(537., color='red', ls='--')
plt.axvline(1879., color='green', ls='--')
plt.text(25., 50000., 'Photosphere', size=15)
plt.text(1000., 20000., 'Chromosphere', size=15, rotation=90.)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.show()
'''

# Part B - Plot the tm=emperature as a function of the ooptical deoth
# First, we are going to calculate the grey atmosphere solution. We are going to use a Teff value of 5780K
T_grey = 5780. * (.75 * falc_data['tau'] + .5)**.25

# Now, let's create the plot comparing both atmospheric curves
'''
plt.title('temperature vs Optical Depth', size=15)
plt.xlabel(r'$\tau_{5000 \AA}$', size=15)
plt.ylabel(r'Temperature $[K]$', size=15)
plt.plot(falc_data['tau'], falc_data['Temperature'], 'r-', label='Regular Atmosphere')
plt.plot(falc_data['tau'], T_grey, 'k-', label='Grey Atmosphere Solution')
plt.tight_layout()
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
'''

# Part C - Plot the number densities as a function of the height
# We are including in this comparision plot the Temperature and ionization fraction
'''
plt.title('Density Profiles vs Height', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.ylabel(r'Density $[cm^{-3}]$', size=15)
plt.plot(falc_data['height'], falc_data['nH'], 'k-', label='Hydrogen Density')
plt.plot(falc_data['height'], falc_data['np'], 'r-', label='Proton Density')
plt.plot(falc_data['height'], falc_data['ne'], 'b-', label='Electron Density')
plt.axvline(537., color='red', ls='--')
plt.axvline(1879., color='green', ls='--')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()
'''

# Part D - For this part we are going to look at the stratification of the density in the Atmosphere
# First, we are going to calculate what is the density at h=0
rho0 = falc_data['rho'][np.where(falc_data['height'] == 0.)[0]].iloc[0]

# Define the density scale height in the log scale
def Hrho_funct(h, hrho): return np.log(falc_data['rho'][np.where(falc_data['height'] == 0.)[0]].iloc[0]) - (h/hrho)

# Perform a curve fit to derive the scale height
popt, pcov = curve_fit(Hrho_funct, falc_data['height'], np.log(falc_data['rho']))

# Then, let's plot the density as a function of the height
'''
plt.title('Mass Density vs Height', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.ylabel(r'Mass Density $[cm^{-3}]$', size=15)
plt.plot(falc_data['height'], falc_data['rho'], 'k-')
plt.plot(falc_data['height'], Hrho_funct(falc_data['height'], *popt), 'r-')
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()
'''

# Part E - Compare the gas presure of the atmosphere to the gas pressure of the sun
# First, let's calculate the pressure in the sun (Pgas) using the ideal gas law
falc_data['Pgas'] = falc_data['nH']*1.38e-16*falc_data['Temperature']

# Then, let's plot the Total Pressure and Solar Pressure as a function of the height
'''
plt.title('Total Pressure vs Height', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.ylabel(r'Pressure $[dyne \ cm^{-2}]$', size=15)
plt.plot(falc_data['height'], falc_data['ptot'], 'k-', label=r'$P_{Total}$')
plt.plot(falc_data['height'], falc_data['Pgas'], 'r-', label=r'$P_{Gas}$')
plt.tight_layout()
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.show()
'''

# Part F - Compare the ratio betewwn gas pressure and total pressure as a function of the height
# Plot the pressure ratio as a function of the height
'''
plt.title(r'$P_{gas}/P_{Total}$ vs Height', size=15)
plt.ylabel(r'$P_{gas}/P_{Total}$', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.plot(falc_data['height'], falc_data['beta'], 'k-')
plt.tight_layout()
plt.show()
'''

# Part G - Study the relation between total pressure and mass column density
# in order to calculate the surface gravity
# First. let's plot the totla pressure as a function of the mass column density
'''
plt.title('Total Pressure vs Column Mass Density', size=15)
plt.ylabel(r'Pressure $[dyne \ cm^{-2}]$', size=15)
plt.xlabel(r'Mass Density $[g \ cm^{-3}]$', size=15)
plt.plot(falc_data['m'], falc_data['ptot'], 'k-')
plt.tight_layout()
plt.show()
'''

# Now using this linear relation between total pressure and cokumn density, let's solve for the surface gravity
logg = np.log10(np.average(falc_data['ptot']/falc_data['m']))

# Part H - Study the chemical mixing in the Atmosphere
# First, we calculate the total hydrogen mass density
falc_data['rhoH'] = falc_data['nH']*1.673e-24

# Now, we calculate the He mass density
falc_data['rhoHe'] = 1.673e-24 * 3.97 *.1 * falc_data['nH']

# Finally, we calcualate the metal density
falc_data['rhoZ'] = falc_data['rho'] - falc_data['rhoH'] - falc_data['rhoHe']

# Plot each of the different mass densities as a function of the height
'''
plt.title('Total Mass Density vs Height', size=15)
plt.xlabel(r'Height $[Km]$', size=15)
plt.ylabel(r'Mass Density $[g \ cm^{-3}]$', size=15)
plt.plot(falc_data['height'], falc_data['rho'], 'k-', label=r'$\rho_{Tot}$')
plt.plot(falc_data['height'], falc_data['rhoH'], 'r-', label=r'$\rho_H$')
plt.plot(falc_data['height'], falc_data['rhoHe'], 'b-', label=r'$\rho_{He}$')
plt.plot(falc_data['height'], falc_data['rhoZ'], 'g-', label=r'$\rho_Z$')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.show()
'''

# Part I - Studying the microturbulent velocity in the atmosphere
# First, we are going to plot the microturbulent velocity as a function of the height
# Then, we calculate the turbulent pressure uing the following formula
falc_data['Pturb'] = (2.*falc_data['rho']*(falc_data['turb_vel']**2.))**-1.

'''
plt.subplot(121)
plt.title('Turbulen Velocity vs Height', size=15)
plt.xlabel(r'Heigh $[Km]$', size=15)
plt.ylabel(r'Turbulent Velocity $[Km/s]$', size=15)
plt.plot(falc_data['height'], falc_data['turb_vel'], 'k-')

plt.subplot(122)
plt.title('Turbulen Pressure vs Height', size=15)
plt.xlabel(r'Heigh $[Km]$', size=15)
plt.ylabel(r'Turbulent Pressure $[dyne \ cm^{-2}]$', size=15)
plt.plot(falc_data['height'], falc_data['Pturb'], 'k-')

plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.show()
'''

# Part K - Plot the astrophysical flux and intensity as a function of the wavelength
'''
plt.subplot(221)
plt.title('Astrophysical Flux with Spectral Lines', size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Flux $[erg \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], solspect_data['smoothed_flux'], 'k-')

plt.subplot(222)
plt.title('Astrophysical Flux without Spectral Lines', size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Flux $[erg \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], solspect_data['continuum_flux'], 'k-')

plt.subplot(223)
plt.title('Intensity with Spectral Lines', size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Intensity $[erg \ \ str^{-1} \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], solspect_data['smoothed_intensity'], 'k-')

plt.subplot(224)
plt.title('Intensity without Spectral Lines', size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Intensity $[erg \ \ str^{-1} \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], solspect_data['continuum_intensity'], 'k-')

plt.tight_layout()
plt.show()
'''

# Part L - Solving the RTE analytically

# Part M _ Define the Planck's function for any input temperatures
def B(wave, T):
    # Define useful constants
    h = const.h.cgs.value
    c = const.c.cgs.value
    kb = const.k_B.cgs.value

    # Conver wavelength micron units to cm
    wave = wave*1.e-4

    return ((2.*h*c**2)/(wave**5.)) * 1./(np.exp((h*c)/(wave*kb*T)) - 1.)

# Now, let's plot the Planck function for three different atmospheric temperatures
'''
plt.title("Planck's Fuction for Atmospheric Temperatures", size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Intensity $[erg \ \ str^{-1} \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], B(solspect_data['wave'], 10000.), 'b-', label=r'$T=10000 \ [K]$')
plt.plot(solspect_data['wave'], B(solspect_data['wave'], 7000.), 'g-', label=r'$T=7000 \ [K]$')
plt.plot(solspect_data['wave'], B(solspect_data['wave'], 3000.), 'r-', label=r'$T=3000 \ [K]$')
plt.legend()
plt.tight_layout()
plt.yscale('log')
plt.show()
'''

# Part N - Fit the Planck's curve to the solar continuum
# Using the Planck's function above, we are going to fit a curve to the solar flux continuum
popt, pcov = curve_fit(B, solspect_data['wave'], solspect_data['continuum_flux']*1.e14, p0=7000.)#, bounds=(5500., 6000.))
Tsun = popt[0]

'''
plt.title('BB Curve Fit to Solar Continuum', size=15)
plt.xlabel(r'Wavelength $[\mu m]$', size=15)
plt.ylabel(r'Flux $[erg \ s^{-1} \ cm^{-2} \ Hz^{-1}]$', size=15)
plt.plot(solspect_data['wave'], solspect_data['continuum_flux']*1e14, 'k-', label='Solar Continuum')
plt.plot(solspect_data['wave'], B(solspect_data['wave'], *popt), 'r-', label='Best BB Curve Fit')
plt.legend()
plt.tight_layout()
plt.show()
'''

# Part O - Calculate the number of excitations and ionizations of the Na D1 line
# First, we are going to define the Boltzmann and Saha equations
def Boltzmann_Eq(gl, gu, El, Eu, T):
    # Define useful constants
    kb = const.k_B.cgs.value

    # eV to erg conversion
    El, Eu = El*1.e-12, Eu*1.e-12

    return (gu/gl) * np.exp((El - Eu)/(kb*T))

def Saha_Eq(Ul, Uu, El, Eu, T):
    # Define useful constants
    me = const.m_e.cgs.value
    kb = const.k_B.cgs.value
    h = const.h.cgs.value
    ne = 1.04e11
    chi = 5.132e-12

    return (2./ne) * (Uu/Ul) * (((2.*np.pi*me*kb*T) / h**2.)**1.5) * np.exp((-chi)/(kb*T))

# Now, let's calculate the excitation and ionization numbers for the Na D1 line
excitations = Boltzmann_Eq(2., 4., 0., 2.1044, Tsun)
ionizations = Saha_Eq(1., 6., 0., 2.1044, Tsun)

# Part P - Calculate the absorption coeffient of the Na D lines
# First, let's define the oacity function
def alpha_NaD(wave, f, T, saha, boltzmann):
    # Define useful constants
    me = const.m_e.cgs.value
    kb = const.k_B.cgs.value
    c = const.c.cgs.value
    h = const.h.cgs.value
    nH = 2.16e13
    Ae = 1.8e-6
    e = 4.8e-10
    nlnE = (1. + saha + boltzmann)**-1.

    # Convert wavelength units from Angstroms to cm
    wave = wave*1e-8

    return ((np.sqrt(np.pi)*e**2.) / (me*c)) * ((wave**2.) / c) * nlnE * nH * Ae * f * (1. - np.exp((-h*c) / (wave*kb*T)))

# calculate the absorption coefficient
alpha = alpha_NaD(5889.95, .641, Tsun, ionizations, excitations)

# Part Q - Plot the absorption coefficient as a function of temperature
# First, we are going to define the temperature range
Ts = np.linspace(3000., 20000.,  1000)

# Now, let's plug these into the absorption coefficient equation and plot it as  a function of # Temperature
'''
plt.title('Absorption Coefficient vs Temperature', size=15)
plt.xlabel(r'Temperature $[K$]', size=15)
plt.ylabel(r'$\alpha_{\lambda} [cm^{-1}]$', size=15)
plt.plot(Ts, alpha_NaD(5889.95, .641, Ts, ionizations, excitations), 'k-')
plt.yscale('log')
plt.tight_layout()
plt.show()
'''

# Part R - Analyze the line profile
# Copy the code frm the lab instructions
def Voigt(a, u):
    I = integrate.quad(lambda y: np.exp(-y**2)/(a**2 + (u - y)**2),-np.inf, np.inf)[0]
    return (a/np.pi)*I

a = 0.1
u_range = np.linspace(-500.,500.,1000)

# Define a range of impact parameters
a_range = np.linspace(0.001, 1., 5)

# Interpolate over all the impact parameter values
'''
for aval in a_range:

    plt.plot(u_range, [Voigt(aval, u) for u in u_range], label=f'a = {aval}')

plt.title('Line Profiles Based on Impact Parameter', size=15)
plt.xlabel(r'$u$', size=15)
plt.ylabel('Intensity', size=15)
plt.tight_layout()
plt.legend()
plt.show()
'''

# Part S - Write up the Schuster-Schwarzschild two layer approximation
# First, we are going to solve for the optical depth
# The bounds used for the height of the atmosphere were taken from the solar atmosphere file
tau = alpha*max(falc_data['height'])

# Now, let's define the intensity equation
def Intensity(urange, l, Tmin, Tmax):
    # Define useful constants
    c = const.c.cgs.value
    T = 6520.
    vturb = 1.6e5
    kb = const.k_B.cgs.value
    me = const.m_e.cgs.value

    # Convert wavelength to cm
    l = l*1e-8

    # Convert the unitless quantity to wavelength units
    # Solve for the doppler broadening
    dellam = (l/c) * np.sqrt(((2.*kb*T)/me) + vturb**2.)
    waves = (urange*dellam) + l

    # Define the optical depth
    u5000 = Voigt(0.1, (5000.*1.e-8 - l)/dellam)
    t = np.asarray([Voigt(0.1, u) for u in urange])/u5000

    # Define the botlzmann equation for each temperature
    Bmin = B(waves*1.e4, Tmin)
    Bmax = B(waves*1.e4, Tmax)

    return Bmin * np.exp(-t) + Bmax * (1. - np.exp(-t)), waves

# Then, let's re-define the Voigt profile function to implement intensity
def Voigt2(a, urange, l, Tmin, Tmax):
    I = Intensity(urange, l, Tmin, Tmax)[0]
    waves = Intensity(urange, l, Tmin, Tmax)[1]

    return I*(a/np.pi), waves


# Part T - plot the line profile at this intensity

plt.title('Line Profile', size=15)
plt.xlabel(r'$wavelength [\AA]$', size=15)
plt.ylabel('Intensity', size=15)
plt.plot(Voigt2(0.1, u_range, 5889.95, 5700., 4200.)[1]*1.e8, Voigt2(0.1, u_range, 5889.95, 5700., 4200.)[0], 'k-')
plt.tight_layout()
plt.show()
