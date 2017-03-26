# UmU
Umea University PhD project

###########################

This script is aimed to automate the processing of 2D synchrotron GIXD data files, including extraction of pictures, cross-section data files, removing baselines, interpolating the peaks,
plotting the cross-sections and calculating the crystallographic parameters.

The existing code is fine tuned to process the peaks of P3HT, but after minor modifications diffractograms of any other crystalline materials can be analysed too.



Input:

one or several folders with sample data located in a designated directory, and the addresses of the calibration file and the directory stated in the script.
Required content of a sample data folder: a subfolder "initial mar2300",
the subfolder "initial mar2300" should contain: 1) a mar2300 data file, 2) a csv data file, 3) a txt file "*parameters.txt".
Required structure of the *parameters.txt file: first line: physical sample length in mm; second line: (equivalent) film thickness in nm.



Output:

a number of additional directories are created as a result of the script operation, including:
-- "pics" folder, containing the pictures of the GIXD pattern in three different colour schemes: jet, binary and spectral
-- "cross sections plots" folder, containing the PDF plots of the various cross-sections, plotted as intensity vs q.
    a successfully processed folder should contain the following files: 

    - "*-Z_bgvis.pdf"            -   Z cross-section plot, with background baseline visualized
    - "*-Z_fitvis.pdf"           -   Z cross-section plot, with interpolation visualized
    - "*-Z_nobg.pdf"             -   Z cross-section plot, after background baseline correction
    - "*-Z_nobg_norm.pdf"        -   Z cross-section plot, after background baseline correction and normalization

    - "*-Z_010_bgvis.pdf"        -   same as above, for 010 fragment of the cross-section
    - "*-Z_010_fitvis.pdf"       -   same as above, for 010 fragment of the cross-section
    - "*-Z_010_nobg.pdf"         -   same as above, for 010 fragment of the cross-section
    - "*-Z_010_nobg_norm.pdf"    -   same as above, for 010 fragment of the cross-section

    - "*-Z_100_bgvis.pdf"        -   same as above, for 100 fragment of the cross-section
    - "*-Z_100_fitvis.pdf"       -   same as above, for 100 fragment of the cross-section
    - "*-Z_100_nobg.pdf"         -   same as above, for 100 fragment of the cross-section
    - "*-Z_100_nobg_norm.pdf"    -   same as above, for 100 fragment of the cross-section

    - "*-xy_bgvis.pdf"           -   xy cross-section plot, with background baseline visualized
    - "*-xy_fitvis.pdf"          -   xy cross-section plot, with interpolation visualized
    - "*-xy_nobg.pdf"            -   xy cross-section plot, after background baseline correction
    - "*-xy_nobg_norm.pdf"       -   xy cross-section plot, after background baseline correction and normalization

    - "*-xy_010_bgvis.pdf"       -   same as above, for 010 fragment of the cross-section
    - "*-xy_010_fitvis.pdf"      -   same as above, for 010 fragment of the cross-section
    - "*-xy_010_nobg.pdf"        -   same as above, for 010 fragment of the cross-section
    - "*-xy_010_nobg_norm.pdf"   -   same as above, for 010 fragment of the cross-section

    - "*-xy_100_bgvis.pdf"       -   same as above, for 100 fragment of the cross-section
    - "*-xy_100_fitvis.pdf"      -   same as above, for 100 fragment of the cross-section
    - "*-xy_100_nobg.pdf"        -   same as above, for 100 fragment of the cross-section
    - "*-xy_100_nobg_norm.pdf"   -   same as above, for 100 fragment of the cross-section

    if any of the files listed above is missing, that typically means that there were some problems during peak interpolation, 
    e.g., caused by the peak being too weak; therefore, the plotting of the resulting plot was automatically skipped

-- "cross sections data" folder, containing the data files and the crystallographic parameters of the various cross-sections, plotted as intensity vs q.
    a successfully processed folder should contain the following files: 

    - "*-Z.csv"                  -   comma-separated value text file with the data of the extracted Z cross-section (raw data), with the following layout:
                                     first column:    pixel no.
                                     second column:   Q(xy) value (=zero for the Z cross-section)
                                     third column:    Q(z) value
                                     fourth column:   Q value (Å^-1)
                                     fifth column:    intensity (counts)

    - "*-Z_nobg.dat"             -   data file of the Z cross-section after the subtraction of the baseline:
                                     first column:    Q value (Å^-1)
                                     second column:   intensity (counts)

    - "*-Z_nobg_norm.dat"        -   data file of the Z cross-section after the subtraction of the baseline and normalization (by the sample length, beam monitor value, equivalent film thickness):
                                     first column:    Q value (Å^-1)
                                     second column:   intensity (a.u.)

    - "*-Z_nobg_norm_d.dat"      -   data file with the crystallographic parameters:
                                     first line:      d-spacing (Å)
                                     second line:     error of d-spacing estimation (Å)
                                     third line:      d-spacing of the second peak, if two peaks are analysed (Å), otherwise output set to zero
                                     fourth line:     error of d-spacing estimation (Å), otherwise output set to zero
                                     fifth line:      paracrystallinity
                                     sixth line:      error of paracrystallinity estimation
                                     seventh line:    error of paracrystallinity estimation calculated using an alternative method (use the larger value of the two)

    - "*-Z_nobg_norm_i.dat"      -   data file with the crystallographic parameters:
                                     first line:      normalized peak intensity (a.u.)
                                     second line:     error of peak intensity estimation (a.u.)

    - "*-Z_nobg_norm_L.dat"      -   data file with the crystallographic parameters:
                                     first line:      coherence length (nm)
                                     second line:     error of coherence length estimation (nm)
                                     third line:      coherence length of the second peak, if two peaks are analysed (nm), otherwise output set to zero
                                     fourth line:     error of coherence length estimation (nm), otherwise output set to zero

    Following the description given above, data file of similar structure should be generated for the 100 and 010 peaks separately;
    the same kind of file set should be generated for the xy cross-section.
    Please use the crystallographic parameters of separately intepolated sections (i.e., 100, 010) since they provide better quality of baseline subtraction and peak fitting.

    If any of the files described above is missing, that typically means that there were some problems during peak interpolation, 
    e.g., caused by the peak being too weak; therefore, the calculation of the corresponding parameter was automatically skipped.

-- "integrated plots" folder, containing the PDF plots of the cross-sections, integrated over all the chi angles, plotted as intensity vs q.
    The content of the folder is based on the same logic as presented above for the "cross sections plots" folder. 

-- "integrated data" folder, containing the data files and the crystallographic parameters of the cross-sections, integrated over all the chi values.
    The content of the folder is based on the same logic as presented above for the "cross sections data" folder. 

-- "rays" folder, containing the extracted and processed cake segments, taken as 1 degree chi narrow ray each. This folder includes:
   - subfolder "cs"                        -   contains the raw data of every ray cake segment, with the index at the end of the filename designating the chi value.
   - subfolder "cs-bgremoved"              -   contains the pdf files of the plots of each ray cake segment, plus the corresponding data files, plus the 010 and 100 fragments plotted separately.
   - file "*_100_cake_nobg_rays.dat"       -   the summarized data file, where each point is a 100 peak intensity over a unity chi degree range, after baseline subtraction and normalization;
   - file "*_100_cake_nobg_rays.pdf"       -   same as above, plotted as a PDF file;
   - file "*_010_cake_nobg_rays.dat"       -   same as above, for 010 peak;
   - file "*_010_cake_nobg_rays.pdf"       -   same as above, plotted as a PDF file;
   - file "*_100_cake_nobg_rays_total.dat" -   the total integrated 100 peak intensity based on this ray analysis.


    





