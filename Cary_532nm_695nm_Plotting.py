import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
from PyQt5.QtWidgets import QFileDialog, QApplication
import sys
import numpy as np
import os
from VLT_curve_loading import load_spectra_curves
import datetime


# Create unique directory names if they already exist
def get_unique_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}_{timestamp}"



 # Sort by lens serial number:  4 digits
def lens_sort_key(serial):
    import re
    match = re.match(r"[A-Za-z]?(\d{4})", str(serial))
    if match:
        digits = match.group(1)
        return int(digits)
    else:
        return 0  # fallback for non-matching serials
       

# Step 1: Prompt for file

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)
file_path, _ = QFileDialog.getOpenFileName(
    None,
    "Select CSV or Excel File",
    "",
    "CSV or Excel files (*.csv *.xlsx *.xls);;All files (*.*)"
)


CambiumOrange = "#D04327"
CambiumGrey = "#6C6D6D"
CambiumBlue = "#23314A"
Black = "#000000"
Blank = ''
# Step 2: Load the file into a DataFrame
if file_path:
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == '.csv':
            t = time.time()
            df = pd.read_csv(file_path)
            ColumnLabels = df.columns.tolist()
            FirstColumnWithoutNumber = df.iloc[:, 1].isna()  # Find nAn after wavelengths
            for k in range(0, len(FirstColumnWithoutNumber)):
                if FirstColumnWithoutNumber[k] == True:
                    EndOfWavelengths = k-1  # Last row of data b4 scan parameters
                    break
            df2 = df.iloc[:EndOfWavelengths, :]  # Dataframe without scan settings
            Wavelengths = pd.to_numeric(df2[df.columns[0]].loc[1:len(df2)], errors='coerce')  # df.columns[0] is the first column, contains all wavelengths
            DyeAbsorbanceMask = (Wavelengths >= 400) & (Wavelengths <= 560)  # Enables performing find_peaks only within specific wavelength range
            DyeWavelengthMask = (Wavelengths >= 400) & (Wavelengths <= 560)
            VLTmask = (Wavelengths >= 380 ) & ( Wavelengths <= 780)
            VLTWavelengths = Wavelengths[VLTmask]
            VLTWavelengths = VLTWavelengths[::-1].reset_index(drop = True) # orders wavelengths numerically [::-1], resets index value
            if 'Wavelength (nm)' not in df2.columns: # Looks for Wavelength (nm) as column label, doesn't exist in 'raw' csv files
                df2 = df2.shift(-1, axis=1)  # Shifts everything over left by 1
                df2 = df2.loc[:, ~df2.columns.str.contains(r'^Unnamed: \d+$')]  # Removes all the columns called 'Unnamed:', which now are excess wavelength (nm) columns         
                df2.insert(0, 'Wavelength (nm)', Wavelengths.astype(object))  # Insert as object dtype
                df2.loc[0, 'Wavelength (nm)'] = 'Wavelength (nm)'  # Makes the first cell in Wavelength column 'Wavelength (nm)'
                df2 = df2.drop([0]) # Drops the topmost row with labels
                df2 = df2.drop(columns = ['Baseline 100%T'])
                df2_VLT = df2[(df2["Wavelength (nm)"] >= 380) & (df2["Wavelength (nm)"] <= 780)].reset_index(drop=True)
                df2_VLT = df2_VLT[::-1].reset_index(drop = True)
                # Exclude 'Wavelength (nm)' from sorting
                lens_columns = [col for col in df2.columns if col != 'Wavelength (nm)']
                sorted_lens_columns = sorted(lens_columns, key=lens_sort_key)
                
                # Reorder df2 columns: 'Wavelength (nm)' first, then sorted lens columns
                df2 = df2[['Wavelength (nm)'] + sorted_lens_columns]

            else:
                print("'Wavelength (nm)' column already exists, skipping insert.")
                df2_VLT = df2[(df2["Wavelength (nm)"] >= 380) & (df2["Wavelength (nm)"] <= 780)].reset_index(drop=True)
                df2_VLT = df2_VLT[::-1].reset_index(drop = True)
            file_path_xlsx = os.path.splitext(file_path)[0] + "_Script made.xlsx"  # Writes to excel, converts text from dataframe to number in excel

            with pd.ExcelWriter(
                    file_path_xlsx,
                    engine="xlsxwriter",
                    engine_kwargs={"options": {"strings_to_numbers": True}}
            ) as writer:
                df2.to_excel(writer, sheet_name="Data", index=False)
            print(f"Converted {file_path} → {file_path_xlsx}")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            file_path_xlsx = file_path
        else:
            raise ValueError("Unsupported file type.")
        print("Spectra file loaded")

# Load curves for VLT calculations
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        curves = load_spectra_curves(script_dir)
        
        Photopic_curve = curves['Photopic_curve']
        Photopic_VLT = curves['Photopic_VLT']
        Photopic_curve_wavelengths = curves['Photopic_curve_wavelengths']
        Photopic_curve_values = curves['Photopic_curve_values']
        
        Scotopic_curve = curves['Scotopic_curve']
        Scotopic_VLT = curves['Scotopic_VLT']
        Scotopic_curve_wavelengths = curves['Scotopic_curve_wavelengths']
        Scotopic_curve_values = curves['Scotopic_curve_values']
        
        CIED65_curve = curves['CIED65_Curve']
        CIED65_VLT = curves['CIED65_VLT']
        CIED65_curve_wavelengths = curves['CIED65_curve_wavelengths']
        CIED65_curve_values = curves['CIED65_curve_values']
        
        RGBConescurves = curves['RGBConescurves']
        RGB_VLT = curves['RGB_VLT']
        Red_Cone = curves['Red_Cone']
        Green_Cone = curves['Green_Cone']
        Blue_Cone = curves['Blue_Cone']
        
        Illuminant_A_curve = curves['IlluminantA_VLT']
        Illuminant_A_wavelengths = curves['IlluminantA_Wavelengths']

# Step 3: Make folders to save plots
        print("Loaded spectra file:", os.path.basename(file_path))
        FileDirectory = os.path.dirname(file_path)

        
        # Step 3a: Make Illuminant folders
        IlluminantADirectory = os.path.join(FileDirectory, 'Illuminant A')
        IlluminantD65Directory = os.path.join(FileDirectory, 'Illuminant D65')
        IlluminantEDirectory = os.path.join(FileDirectory, 'Illuminant E')
        NoPeakDirectory = os.path.join(FileDirectory, 'No peak found')
        
        os.makedirs(IlluminantADirectory, exist_ok=True)
        os.makedirs(IlluminantD65Directory, exist_ok=True)
        os.makedirs(IlluminantEDirectory, exist_ok=True)
        os.makedirs(NoPeakDirectory, exist_ok = True)
        
        
        #Step 3b: Make label subfolders
        IlluminantA_With_Labels_Directory = os.path.join(IlluminantADirectory, 'Labeled')
        IlluminantA_Without_Labels_Directory = os.path.join(IlluminantADirectory, 'Unlabeled')
        os.makedirs(IlluminantA_With_Labels_Directory, exist_ok=True)
        os.makedirs(IlluminantA_Without_Labels_Directory, exist_ok=True)
        
        IlluminantD65_With_Labels_Directory = os.path.join(IlluminantD65Directory, 'Labeled')
        IlluminantD65_Without_Labels_Directory = os.path.join(IlluminantD65Directory, 'Unlabeled')
        os.makedirs(IlluminantD65_With_Labels_Directory, exist_ok=True)
        os.makedirs(IlluminantD65_Without_Labels_Directory, exist_ok=True)
        
        IlluminantE_With_Labels_Directory = os.path.join(IlluminantEDirectory, 'Labeled')
        IlluminantE_Without_Labels_Directory = os.path.join(IlluminantEDirectory, 'Unlabeled')
        os.makedirs(IlluminantE_With_Labels_Directory, exist_ok=True)
        os.makedirs(IlluminantE_Without_Labels_Directory, exist_ok=True)

        OpticalDataResults = []
        numeric_cols = df2.select_dtypes(include='number').columns

# Step 4: Loop through csv file, calculate Peak OD, λ at peak OD, plot datas
        print("Column labels:")
        for col in df2.columns:
            if col == 'Wavelength (nm)':
                continue
            else:
                # Sanitize the serial number to remove invalid filename characters
                SerialNumber = str(col).strip()  # Remove leading/trailing whitespace
                SerialNumber = SerialNumber.replace('\\', '_').replace('/', '_').replace(':', '_')
                SerialNumber = SerialNumber.replace('*', '_').replace('?', '_').replace('"', '_')
                SerialNumber = SerialNumber.replace('<', '_').replace('>', '_').replace('|', '_')
                print(f"Spectrum for lens: {SerialNumber} \n")
                Absorbance = pd.to_numeric(df2[col].loc[1:len(df2)], errors='coerce')
                VLTAbsorbance = Absorbance[VLTmask]
                DyeAbsorbance = Absorbance[DyeAbsorbanceMask]
                DyeWavelength = Wavelengths[DyeWavelengthMask]
                TransmittancePercent = 10.0**(-Absorbance)*100  # Transmittance calculated from absorption
                VLTTransmittance = 10.0**(-VLTAbsorbance)
                VLTTransmittance = VLTTransmittance[::-1].reset_index(drop = True) # Need to invert since it's backwards
                peaks, properties = find_peaks(DyeAbsorbance, prominence=.2, width=8)


                 # Calculating photopic & scotopic values
                if 'Photopic_curve' in locals() and Photopic_curve is not None:
                    Photopic_x_Transmittance_Illuminant_E = Photopic_curve_values * VLTTransmittance
                    Photopic_x_Transmittance_D65 = Photopic_curve_values * VLTTransmittance * CIED65_curve_values
                    Photopic_x_Transmittance_Illuminant_A = Photopic_curve_values * VLTTransmittance * Illuminant_A_curve
                    VLT_Photopic_Response_Illuminant_E = round(100 * (np.nansum(Photopic_x_Transmittance_Illuminant_E) / np.nansum(Photopic_curve_values)),2)
                    VLT_Photopic_Response_Illuminant_D65 = round(100 * (np.nansum(Photopic_x_Transmittance_D65) / np.nansum(Photopic_curve_values * CIED65_curve_values)),2)
                    VLT_Photopic_Response_Illuminant_A = round(100 * (np.nansum(Photopic_x_Transmittance_Illuminant_A) / np.nansum(Photopic_curve_values * Illuminant_A_curve)),2)
                
                if 'Scotopic_curve' in locals() and Scotopic_curve is not None:
                    Scotopic_x_Transmittance_Illuminant_E = Scotopic_curve_values * VLTTransmittance
                    Scotopic_x_Transmittance_D65 = Scotopic_curve_values * VLTTransmittance * CIED65_curve_values
                    Scotopic_x_Transmittance_Photopic_x_Transmittance_Illuminant_A = Scotopic_curve_values * VLTTransmittance * Illuminant_A_curve
                    VLT_Scotopic_Response_Illuminant_E = round(100 * (np.nansum(Scotopic_x_Transmittance_Illuminant_E) / np.nansum(Scotopic_curve_values)),2)
                    VLT_Scotopic_Response_Illuminant_D65 = round(100 * (np.nansum(Scotopic_x_Transmittance_D65) / np.nansum(Scotopic_curve_values * CIED65_curve_values)),2)
                    VLT_Scotopic_Response_Illuminant_A = round(100 * (np.nansum(Scotopic_x_Transmittance_Photopic_x_Transmittance_Illuminant_A) / np.nansum(Scotopic_curve_values * Illuminant_A_curve)),2)

                    
                # --- RGB VLT calcluation --- 
                if 'Red_Cone' in locals() and Red_Cone is not None:
                    Red_Transmittance_Illuminant_E = round(100 * (np.nansum(Red_Cone * VLTTransmittance)/(np.nansum(Red_Cone))),2)
                    Red_Transmittance_Illuminant_D65 = round(100 * (np.nansum(Red_Cone * VLTTransmittance * CIED65_curve_values)/(np.nansum(Red_Cone * CIED65_curve_values))),2)
                    Red_Transmittance_Illuminant_A = round(100 * (np.nansum(Red_Cone * VLTTransmittance * Illuminant_A_curve)/(np.nansum(Red_Cone * Illuminant_A_curve))),2)
                
                if 'Green_Cone' in locals() and Green_Cone is not None:   
                    Green_Transmittance_Illuminant_E = round(100 * (np.nansum(Green_Cone * VLTTransmittance)/(np.nansum(Green_Cone))),2)
                    Green_Transmittance_Illuminant_D65 = round(100 * (np.nansum(Green_Cone * VLTTransmittance * CIED65_curve_values)/(np.nansum(Green_Cone * CIED65_curve_values))),2)
                    Green_Transmittance_Illuminant_A = round(100 * (np.nansum(Green_Cone * VLTTransmittance * Illuminant_A_curve)/(np.nansum(Green_Cone * Illuminant_A_curve))),2)
                
                if 'Blue_Cone' in locals() and Blue_Cone is not None:
                    Blue_Transmittance_Illuminant_E = round(100 * (np.nansum(Blue_Cone * VLTTransmittance)/(np.nansum(Blue_Cone))),2)
                    Blue_Transmittance_Illuminant_D65 = round(100 * (np.nansum(Blue_Cone * VLTTransmittance * CIED65_curve_values)/(np.nansum(Blue_Cone * CIED65_curve_values))),2)
                    Blue_Transmittance_Illuminant_A = round(100 * (np.nansum(Blue_Cone * VLTTransmittance * Illuminant_A_curve)/(np.nansum(Blue_Cone * Illuminant_A_curve))),2)
                                
                if len(peaks) > 0:
                    # Get index of the tallest peak
                    peak_idx = peaks[DyeAbsorbance.iloc[peaks].argmax()]
                    WavelengthPeak = DyeWavelength.iloc[peak_idx]
                    AbsorbancePeak = round(DyeAbsorbance.iloc[peak_idx], 2)
                    if 532 in DyeWavelength.values:
                        FiveThirtyTwoIndex = DyeWavelength[DyeWavelength == 532].index[0]
                        FiveThirtyTwoOD = round(DyeAbsorbance.loc[FiveThirtyTwoIndex],2)
                        FiveThirtyTwo = 532
                    else:
                        FiveThirtyTwo = None
                        FiveThirtyTwoOD = None
                    if 500 in DyeWavelength.values:
                            FiveHundredIndex = DyeWavelength[DyeWavelength == 500].index[0]
                            FiveHundredOD = round(DyeAbsorbance.loc[FiveHundredIndex],2)
                            FiveHundred = 500
                    else:
                            FiveHundred = None
                            FiveHundredOD = None                                      
                    OpticalDataResults.append([
                        SerialNumber, WavelengthPeak, AbsorbancePeak, FiveThirtyTwo, FiveThirtyTwoOD,
                        FiveHundred, FiveHundredOD, Blank, VLT_Photopic_Response_Illuminant_E, 
                        VLT_Scotopic_Response_Illuminant_E, Red_Transmittance_Illuminant_E,
                        Green_Transmittance_Illuminant_E, Blue_Transmittance_Illuminant_E,
                        Blank, VLT_Photopic_Response_Illuminant_D65, VLT_Scotopic_Response_Illuminant_D65, 
                        Red_Transmittance_Illuminant_D65, Green_Transmittance_Illuminant_D65, 
                        Blue_Transmittance_Illuminant_D65, Blank, VLT_Photopic_Response_Illuminant_A,                        
                        VLT_Scotopic_Response_Illuminant_A, Red_Transmittance_Illuminant_A,
                        Green_Transmittance_Illuminant_A, Blue_Transmittance_Illuminant_A])
                    
# Plot lens data with peak data
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    PlotName = f"{SerialNumber} Illuminant E.png"
                    plt.title(f"{SerialNumber} Illuminant E")
                    fig1.savefig(os.path.join(IlluminantE_Without_Labels_Directory, PlotName), dpi=300, bbox_inches='tight')

                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    PlotName = f"{SerialNumber} Illuminant E.png"
                    plt.title(f"{SerialNumber} Illuminant E")
                    fig1.savefig(os.path.join(IlluminantE_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')           

                    ax1.text(
                    0.78, 0.55,
                    f"VLT (%):\n",
                    color='black', fontsize=32, ha='left', va='top',
                    transform=ax1.transAxes)
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{Red_Transmittance_Illuminant_E:.2f}",
                        color='red', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.45,

                        f"{Green_Transmittance_Illuminant_E:.2f}",
                        color='green', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.40,

                        f"{Blue_Transmittance_Illuminant_E:.2f}",
                        color='blue', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    PlotName = f"{SerialNumber} Illuminant E RGB VLT.png"
                    plt.title(f"{SerialNumber} Illuminant E")

                    fig1.savefig(os.path.join(IlluminantE_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                    # Plot with Photopic VLT using illuminant E
                    
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    ax1.text(
                    0.78, 0.55,f"Photopic (%):\n",
                    color='black', fontsize=28, ha='left', va='top',
                    transform=ax1.transAxes)
                    
                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{VLT_Photopic_Response_Illuminant_E:.2f}",
                        color='black', fontsize=28, ha='left', va='top',
                        transform=ax1.transAxes)
                    PlotName = f"{SerialNumber} Illuminant E photopic.png"
                    plt.title(f"{SerialNumber} Illuminant E")
                    fig1.savefig(os.path.join(IlluminantE_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
# Plot lens data with using Illuminant D65
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    PlotName = f"{SerialNumber} Illuminant D65.png"
                    plt.title(f"{SerialNumber} Illuminant D65")
                    fig1.savefig(os.path.join(IlluminantD65_Without_Labels_Directory, PlotName), dpi=300, bbox_inches='tight')

                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    PlotName = f"{SerialNumber} Illuminant D65.png"
                    plt.title(f"{SerialNumber} Illuminant D65")

                    fig1.savefig(os.path.join(IlluminantD65_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                    ax1.text(
                    0.78, 0.55, f"VLT (%):\n",
                    color='black', fontsize=32, ha='left', va='top',
                    transform=ax1.transAxes)
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{Red_Transmittance_Illuminant_D65:.2f}",
                        color='red', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.45,
                        f"{Green_Transmittance_Illuminant_D65:.2f}",
                        color='green', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.40,
                        f"{Blue_Transmittance_Illuminant_D65:.2f}",
                        color='blue', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    PlotName = f"{SerialNumber} Illuminant D65 RGB VLT.png"
                    plt.title(f"{SerialNumber} Illuminant D65")
                    fig1.savefig(os.path.join(IlluminantD65_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                    # Plot with Photopic VLT using illuminant D65
                    
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    ax1.text(
                    0.78, 0.55,f"Photopic (%):\n",
                    color='black', fontsize=28, ha='left', va='top',
                    transform=ax1.transAxes)
                    
                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                    ax1.text(
                        0.78, 0.50,
                        f"{VLT_Photopic_Response_Illuminant_D65:.2f}",
                        color='black', fontsize=28, ha='left', va='top',
                        transform=ax1.transAxes)           
                    PlotName = f"{SerialNumber} Illuminant D65 photopic.png"
                    plt.title(f"{SerialNumber} Illuminant D65")

                    fig1.savefig(os.path.join(IlluminantD65_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
# Plot lens data with using Illuminant A
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    PlotName = f"{SerialNumber} Illuminant A.png"
                    plt.title(f"{SerialNumber} Illuminant A")
                    fig1.savefig(os.path.join(IlluminantA_Without_Labels_Directory, PlotName), dpi=300, bbox_inches='tight')

                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    PlotName = f"{SerialNumber} Illuminant A.png"
                    plt.title(f"{SerialNumber} Illuminant A")

                    fig1.savefig(os.path.join(IlluminantA_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                    ax1.text(
                    0.78, 0.55, f"VLT (%):\n",
                    color='black', fontsize=32, ha='left', va='top',
                    transform=ax1.transAxes)
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{Red_Transmittance_Illuminant_A:.2f}",
                        color='red', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.45,
                        f"{Green_Transmittance_Illuminant_A:.2f}",
                        color='green', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.40,
                        f"{Blue_Transmittance_Illuminant_A:.2f}",
                        color='blue', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    PlotName = f"{SerialNumber} Illuminant A RGB VLT.png"
                    plt.title(f"{SerialNumber} Illuminant A")
                    fig1.savefig(os.path.join(IlluminantA_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                    # Plot with Photopic VLT using illuminant A
                    
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    ax1.text(0.78, 0.65, F"λ = {WavelengthPeak:.1f} nm \nOD = {AbsorbancePeak:.2f}",
                             color='black', fontsize=28, ha='left', va='top',
                             transform=ax1.transAxes)  # ha & va horizontal/vertical alignment
                    ax1.text(
                    0.78, 0.55,f"Photopic (%):\n",
                    color='black', fontsize=28, ha='left', va='top',
                    transform=ax1.transAxes)
                    
                    ax1.plot(WavelengthPeak, AbsorbancePeak, marker='o',
                             mfc="none", mec='blue', mew=2, markersize=24)
                    ax1.text(
                        0.78, 0.50,
                        f"{VLT_Photopic_Response_Illuminant_A:.2f}",
                        color='black', fontsize=28, ha='left', va='top',
                        transform=ax1.transAxes)
                             
                    PlotName = f"{SerialNumber} Illuminant A photopic.png"
                    plt.title(f"{SerialNumber} Illuminant A")
                    fig1.savefig(os.path.join(IlluminantA_With_Labels_Directory,PlotName), dpi=300,bbox_inches='tight')
                    
                else:
                    print(f"Lens: {col} — No peak found\n")
                    WavelengthPeak = None
                    AbsorbancePeak = None
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22, bbox_to_anchor=(1, 0.75), loc='center right')
                    if 532 in DyeWavelength.values:
                        FiveThirtyTwoIndex = DyeWavelength[DyeWavelength == 532].index[0]
                        FiveThirtyTwoOD = round(DyeAbsorbance.loc[FiveThirtyTwoIndex],2)
                        FiveThirtyTwo = 532
                    else:
                        FiveThirtyTwo = None
                        FiveThirtyTwoOD = None
                    if 500 in DyeWavelength.values:
                            FiveHundredIndex = DyeWavelength[DyeWavelength == 500].index[0]
                            FiveHundredOD = round(DyeAbsorbance.loc[FiveHundredIndex],2)
                            FiveHundred = 500
                    else:
                            FiveHundred = None
                            FiveHundredOD = None                        
                    OpticalDataResults.append([
                        SerialNumber, WavelengthPeak, AbsorbancePeak, FiveThirtyTwo, FiveThirtyTwoOD,
                        FiveHundred, FiveHundredOD, Blank, VLT_Photopic_Response_Illuminant_E, 
                        VLT_Scotopic_Response_Illuminant_E, Red_Transmittance_Illuminant_E,
                        Green_Transmittance_Illuminant_E, Blue_Transmittance_Illuminant_E,
                        Blank, VLT_Photopic_Response_Illuminant_D65, VLT_Scotopic_Response_Illuminant_D65, 
                        Red_Transmittance_Illuminant_D65, Green_Transmittance_Illuminant_D65, 
                        Blue_Transmittance_Illuminant_D65, Blank, VLT_Photopic_Response_Illuminant_A,                        
                        VLT_Scotopic_Response_Illuminant_A, Red_Transmittance_Illuminant_A,
                        Green_Transmittance_Illuminant_A, Blue_Transmittance_Illuminant_A])
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    PlotName = f"{SerialNumber} without peaks.png"
                    plt.title(f"{SerialNumber} without peaks")
                    fig1.savefig(os.path.join(NoPeakDirectory, PlotName), dpi=300, bbox_inches='tight')
                    
                    ax1.text(
                    0.78, 0.55,
                    f"VLT (%):\n",
                    color='black', fontsize=32, ha='left', va='top',
                    transform=ax1.transAxes)
                    
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{Red_Transmittance_Illuminant_E:.2f}",
                        color='red', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.45,
                        f"{Green_Transmittance_Illuminant_E:.2f}",
                        color='green', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    ax1.text(
                        0.78, 0.40,
                        f"{Blue_Transmittance_Illuminant_E:.2f}",
                        color='blue', fontsize=32, ha='left', va='top',
                        transform=ax1.transAxes)
                    
                    PlotName = f"{SerialNumber} without peaks with VLT.png"
                    plt.title(f"{SerialNumber} without peaks with VLT")

                    fig1.savefig(os.path.join(NoPeakDirectory, PlotName), dpi=300, bbox_inches='tight')
                    
                    # Add photopic values separately for each color
                    plt.rcParams.update({'font.family': 'Arial', 'font.size': 30})
                    fig1, ax1 = plt.subplots(figsize=(16, 12))
                    PlotOne, = ax1.plot(Wavelengths, Absorbance, linewidth=4, color=CambiumOrange)
                    ax3 = ax1.twinx()
                    PlotTwo, = ax3.plot(Wavelengths, TransmittancePercent, linewidth=4, linestyle=':', color=CambiumBlue)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Absorbance (OD)')
                    ax3.set_ylabel('Transmission (%)')
                    ax1.set_xlim([380, 780])
                    ax1.set_ylim([0, 7])
                    ax3.set_ylim([0, 100])
                    Curves = [PlotOne, PlotTwo]
                    labels = ["Abs", "%T"]
                    ax1.legend(Curves, labels, frameon=False, fontsize=22,
                               bbox_to_anchor=(1, 0.75), loc='center right')
                    for spine in ax1.spines.values():
                        spine.set_linewidth(2.5)  # Make box around plot thicker
                    ax1.tick_params(direction="in", width=2, length=12, which="both")
                    ax1.text(
                    0.78, 0.55,
                    f'Photopic (%):\n', 
                    color='black', fontsize=28, ha='left', va='top',
                    transform=ax1.transAxes)
                
                # Add colored VLT values separately for each color
                    ax1.text(
                        0.78, 0.50,
                        f"{VLT_Photopic_Response_Illuminant_E:.2f}",
                        color='black', fontsize=28, ha='left', va='top',
                        transform=ax1.transAxes)
                    
                    PlotName = f"{SerialNumber} without peaks and photopic VLT Illuminant E.png"
                    plt.title(f"{SerialNumber} without peaks and photopic VLT Illuminant E", pad = -14)
                    fig1.savefig(os.path.join(NoPeakDirectory,PlotName), dpi=300,bbox_inches='tight')

# Step 5: Output parameters to excel: Serial #, ODs, VLTs
        OpticalDataDataframe = pd.DataFrame(OpticalDataResults, columns=[
                "Lens Serial #", "Peak Wavelength (nm)", "Peak Value (OD)",
                '532 nm', "OD at 532 nm",'500 nm', 'OD at 500 nm', 'Blank',
                "VLT Photopic Illuminant E (%)", 
                'VLT Scotopic Illuminant E (%)', 'Red VLT (Illuminant E)',
                'Green VLT (Illuminant E)', 'Blue VLT (Illuminant E)',
                'Blank', "VLT Photopic Illuminant D65 (%)", 
                'VLT Scotopic Illuminant D65 (%)', 'Red VLT (Illuminant D65)',
                'Green VLT (Illuminant D65)', 'Blue VLT (Illuminant D65)',
                'Blank', "VLT Photopic Illuminant A (%)", 
                'VLT Scotopic Illuminant A (%)', 'Red VLT (Illuminant A)',
                'Green VLT (Illuminant A)', 'Blue VLT (Illuminant A)'])
        


        with pd.ExcelWriter(
            file_path_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            OpticalDataDataframe.to_excel(writer, sheet_name="Peak_Results", index=False)


        print("Peak results saved to new sheet: Peak_Results \n")
        print(f"Final Excel file saved: {file_path_xlsx}")
        ElapsedTime = round((time.time()-t)/60, 3)
        print(f"Time to run code: {ElapsedTime} minutes")
        

    except Exception as e:
        print("Error loading file:", e)
else:
    print("No file selected.")
