import os
import pandas as pd
import numpy as np

def load_spectra_curves(script_dir):
    """
    Loads photopic, scotopic, CIE D65 illuminant, and RGB cone curves from the script directory.
    Returns a dictionary with the loaded DataFrames and spectra values.
    """

    curves = {}

    # Load Photopic spectra
    photopic_path = os.path.join(script_dir, "CIE_sle_photopic.csv")
    if os.path.exists(photopic_path):
        print("Loading Photopic_curve from:", photopic_path, "\n")
        Photopic_curve = pd.read_csv(photopic_path)
        Photopic_VLT = Photopic_curve[(Photopic_curve["Wavelength (nm)"] >= 380) & (Photopic_curve["Wavelength (nm)"] <= 780)].reset_index(drop=True)
        curves['Photopic_curve'] = Photopic_curve
        curves['Photopic_VLT'] = Photopic_VLT
        curves['Photopic_curve_wavelengths'] = Photopic_VLT.iloc[:,0]
        curves['Photopic_curve_values'] = Photopic_VLT.iloc[:,1]
        print("Photopic_curve file loaded:", os.path.basename(photopic_path), "\n")
    else:
        print("Photopic_curve file not found at:", photopic_path, "\n")
        curves['Photopic_curve'] = None

    # Load Scotopic spectra
    scotopic_path = os.path.join(script_dir, "Scotopic function.csv")
    if os.path.exists(scotopic_path):
        print("Loading Scotopic spectra from:", scotopic_path, "\n")
        Scotopic_curve = pd.read_csv(scotopic_path)
        Scotopic_VLT = Scotopic_curve[(Scotopic_curve["Wavelength (nm)"] >= 380) & (Scotopic_curve["Wavelength (nm)"] <= 780)].reset_index(drop=True)
        curves['Scotopic_curve'] = Scotopic_curve
        curves['Scotopic_VLT'] = Scotopic_VLT
        curves['Scotopic_curve_wavelengths'] = Scotopic_VLT.iloc[:,0]
        curves['Scotopic_curve_values'] = Scotopic_VLT.iloc[:,1]
        print("Scotopic spectra file loaded:", os.path.basename(scotopic_path), "\n")
    else:
        print("Scotopic spectra file not found at:", scotopic_path, "\n")
        curves['Scotopic_curve'] = None

    # Load CIE D65 illuminant spectra
    CIED65path = os.path.join(script_dir, "CIE_std_illum_D65.csv")
    if os.path.exists(CIED65path):
        print("Loading CIE D65 spectra from:", CIED65path, "\n")
        CIED65_Curve = pd.read_csv(CIED65path)
        CIED65_VLT = CIED65_Curve[(CIED65_Curve["Wavelength (nm)"] >= 380) & (CIED65_Curve["Wavelength (nm)"] <= 780)].reset_index(drop=True)
        curves['CIED65_Curve'] = CIED65_Curve
        curves['CIED65_VLT'] = CIED65_VLT
        curves['CIED65_curve_wavelengths'] = CIED65_VLT.iloc[:,0]
        curves['CIED65_curve_values'] = CIED65_VLT.iloc[:,1]
        print("CIE D65 spectra file loaded:", os.path.basename(CIED65path), "\n")
    else:
        print("CIE D65 spectra file not found at:", CIED65path, "\n")
        curves['CIED65_Curve'] = None

    # Load RGB cones spectra
    RGBConespath = os.path.join(script_dir, "RGB cones.csv")
    if os.path.exists(RGBConespath):
        print("Loading RGB cones curves from:", RGBConespath, "\n")
        RGBConescurves = pd.read_csv(RGBConespath)
        extra_wavelengths = pd.DataFrame({
            'Wavelength (nm)': np.arange(380, 390),
            **{col: 0 for col in RGBConescurves.columns if col != 'Wavelength (nm)'}
        })
        RGBConescurves = pd.concat([extra_wavelengths, RGBConescurves], ignore_index=True)
        RGBConescurves = RGBConescurves.sort_values(by='Wavelength (nm)').reset_index(drop=True)
        RGB_VLT = RGBConescurves[(RGBConescurves["Wavelength (nm)"] >= 380) & (RGBConescurves["Wavelength (nm)"] <= 780)].reset_index(drop = True)
        RGB_VLT = RGB_VLT.fillna(0)
        curves['RGBConescurves'] = RGBConescurves
        curves['RGB_VLT'] = RGB_VLT
        curves['Red_Cone'] = RGB_VLT.iloc[:,1]
        curves['Green_Cone'] = RGB_VLT.iloc[:,2]
        curves['Blue_Cone'] = RGB_VLT.iloc[:,3]
        print("RGB cones spectra file loaded:", os.path.basename(RGBConespath), "\n")
    else:
        print("RGB cones file not found at:", RGBConespath, "\n")
        curves['RGBConescurves'] = None
        
        
    CIEIlluminantApath = os.path.join(script_dir, "CIE_std_illum_A_1nm.csv")
    if os.path.exists(RGBConespath):
        print("Loading CIE Illuminant A curves from:", CIEIlluminantApath, "\n")
        IlluminantACurves = pd.read_csv(CIEIlluminantApath)
        IlluminantACurves = IlluminantACurves.sort_values(by='Wavelength (nm)').reset_index(drop=True)
        IlluminantA_VLT = IlluminantACurves[(IlluminantACurves["Wavelength (nm)"] >= 380) & (IlluminantACurves["Wavelength (nm)"] <= 780)].reset_index(drop = True)
        IlluminantA_VLT = IlluminantA_VLT.fillna(0)
        curves['IlluminantA_Wavelengths'] = IlluminantACurves.iloc[:,0]
        curves['IlluminantA_VLT'] = IlluminantA_VLT.iloc[:,1]
        print("Illuminant A spectra file loaded:", os.path.basename(CIEIlluminantApath), "\n")
    else:
        print("Illuminant A spectra file not found at:", CIEIlluminantApath, "\n")
        curves['IlluminantACurves'] = None

    return curves