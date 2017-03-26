import csv
import os
import glob
import wxdiff_api
import re
import tkFileDialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import math
import time


#Gaussian function 
def gauss(x, y0, A, mu, sigma):
    return y0 + A*np.exp(-(x-mu)**2/(2.*sigma**2))

#Sum of two Gaussian functions
def gauss2(x, y0, A1, mu1, sigma1, A2, mu2, sigma2):
    return y0 + A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

#Get BStop value from given filename and line
def getBStop(filename,line):
  dataread = open(filename,'r') #Open the file
  data = [] #Empty table to be filled
  datastream = csv.reader(dataread) #Getting the values
  for row in datastream:
    data.append(row)
  return int(data[line][3]) #Grabbing the BStop value

#Making a 2-D table from two 1-D tables
def tabMerge(x, y):
  tab2D = [[0 for i in range(2)] for j in range(len(x))]
  for i in range(len(x)):
    tab2D[i][0]=x[i]
    tab2D[i][1]=y[i]
  return tab2D

#Normalize given data using BStop from given filename and line number
def normalize(data,filename,line):
  BStop = getBStop(filename,line)
  x = [data._plot_wd.varying_dir_range[0] + i*data._plot_wd.scalingfac for i in data._plot_wd.xdata]
  y = data._plot_wd.ydata
  #Making a single table out of that
  value = tabMerge(x,y)
  i=0
  while i<len(x):
    value[i][1]=value[i][1]/BStop * 18000 #Dividing integrated intensity by BStop value and normalizing to 18000 counts
    i += 1
  return value

#Save given data in given filename
def saveData(filename, data):
  dataFile = open(filename, 'w')
  writer = csv.writer(dataFile)
  writer.writerows(data)

#Get CSV filename and line nb corresponding to given image file
def getCSV(filename):
  cut = getBaseName(filename).split('_')
  length = len(cut)
  line = int(cut[length-2]) #Grabbing last counter to select the proper line in the associated csv file
  del cut[length-3:length-1] #Deleting counters
  csvName = '_'.join(cut) + '.csv'
  return (csvName, line)

#Get peak datafile name
def getBaseName(filename):
  baseName = os.path.splitext(filename)[0]
  return baseName

#Getting list of mar2300 files in a folder
def getFileList(directoryy):
#  print directoryy
  return glob.glob(directoryy + '*.mar2300')

#Getting row sum from given dataFile for given Chi and Q values
def getRowSum1(pic__, Q1__, Q2__, Chi1__, Chi2__, fileNameToSave):
  (X1__, Y1__) = pic__.get_XY_from_QChi(Q1__,Chi1__)
  (X2__, Y2__) = pic__.get_XY_from_QChi(Q2__,Chi2__)
  selected__ = pic__.add_cakeseg(X1__,Y1__,X2__,Y2__)
  selected__ = selected__.convert_to_qchi()
  selected__.save_to_png(fileNameToSave)
  colSum__ = selected__.row_sum()
  selected__.close()
  return colSum__

#Getting column sum from given dataFile for given Chi and Q values
def getColSum(pic_2, Q1_2, Q2_2, Chi1_2, Chi2_2, fileNameToSave):
  (X1_2, Y1_2) = pic_2.get_XY_from_QChi(Q1_2,Chi1_2)
  (X2_2, Y2_2) = pic_2.get_XY_from_QChi(Q2_2,Chi2_2)
  selected_2 = pic_2.add_cakeseg(X1_2,Y1_2,X2_2,Y2_2)
  selected_2 = selected_2.convert_to_qchi()
  selected_2.save_to_png(fileNameToSave)
  colSum_2 = selected_2.column_sum()
  selected_2.close()
  return colSum_2

#Processing the given peak
def processPeak(dataFile_,pic_,Q1_,Q2_,Chi1_,Chi2_,peakName_):
    print sample_dir
    if peakName_ == '100':
        toSavePdf11 = sample_dir + '/100 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '-QChi-Vis.png'
    elif peakName_ == '100-circle':
        toSavePdf11 = sample_dir + '/100 circle/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '-QChi-Vis.png'
    elif peakName_ == '200':
        toSavePdf11 = sample_dir + '/200 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '-QChi-Vis.png'
    elif peakName_ == '010':
        toSavePdf11 = sample_dir + '/010 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '-QChi-Vis.png'
    
    integrated_ = getRowSum1(pic_,Q1_,Q2_,Chi1_,Chi2_, toSavePdf11)
    
    if peakName_ == '100':
        toSave = sample_dir + '/100 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '.csv'
        integrated_.save_to_csv(toSave)
    elif peakName_ == '100-circle':
        toSave = sample_dir + '/100 circle/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '.csv'
        integrated_.save_to_csv(toSave)
    elif peakName_ == '200':
        toSave = sample_dir + '/200 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '.csv'
        integrated_.save_to_csv(toSave)
    elif peakName_ == '010':
        toSave = sample_dir + '/010 cake data/' + os.path.splitext(os.path.basename(dataFile_))[0] + '-' + peakName_ + '.csv'
        integrated_.save_to_csv(toSave)

    integrated_.close()


#Processing the given peak - integrated
def processPeakIntegrated(dataFile,pic,Q1,Q2,Chi1,Chi2,peakName):
    toSave = sample_dir + '/integrated data/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + peakName + '.csv'
    toSavePng = sample_dir + '/integrated data/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + peakName + '-QChi-Vis.png'
    integrated = getColSum(pic,Q1,Q2,Chi1,Chi2,toSavePng)
    integrated.save_to_csv(toSave)
    integrated.close()


#Getting line in Z direction from given dataFile
def getZ(picc, Q11, Q22, Chi11, Chi22):
  (X11, Y11) = picc.get_XY_from_QChi(Q11,Chi11)
  (X22, Y22) = picc.get_XY_from_QChi(Q22,Chi22)
  selectedd = picc.add_linecut(X11,Y11,X22,Y22)
  return selectedd

#Processing the given peak
def processPeakZ(dataFile,pic,Q1,Q2,Chi1,Chi2,peakName):
    integrated = getZ(pic,Q1,Q2,Chi1,Chi2)
    toSaveZ = sample_dir + '/cross sections data/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + peakName + '.csv'
    integrated.save_to_csv(toSaveZ)


#Data treatment - extracting the cake segments for various peaks and the Z cross-section from the data file
def dataTreat(directoryyy,calibFile):
  files = getFileList(directoryyy + 'initial mar2300/')
  for dataFile in files:
    #Initialize the image object
    pic = wxdiff_api.wxdiff_diffimage(MDIRoot)
    #Open the file and calibrate it
    pic.fromfile(dataFile,filetypestr='MAR345',calibfname=calibFile, interactive=False)

    processPeak(dataFile,pic,0.25,0.6,0,180,'100')
    processPeak(dataFile,pic,0.6,0.95,0,180,'200')
    processPeak(dataFile,pic,1.5,1.85,0,180,'010')
    processPeakIntegrated(dataFile,pic,0,2.5,0,180,'integrated')

    processPeakZ(dataFile,pic,0.25,2,90,90,'Z')

    pic.close()




#Getting Z cross-section filename
def getFileListZ(directory):
  return glob.glob(directory + '*-Z.csv')

#Getting 100 cake filename
def getFileList100(directory):
  return glob.glob(directory + '*-100.csv')

#Getting parameters filename
def getFileListParams(directory):
  return glob.glob(directory + '*parameters.txt')

#Getting csv filename
def getFileListCsv(directory):
  return glob.glob(directory + '*.csv')

#Getting pic's filename
def getFileListPng(directory):
  return glob.glob(directory + '*.png')

#Determination of peak maxima
def dataMaximaZ(directory1):
  filesZ = getFileListZ(directory1)[0]
  with open(filesZ, "rb") as infile:
    reader = csv.reader(infile, delimiter=",")
    allData = list(reader)
    allData1 = zip(*allData)
    allData2 = allData1[5]
    allData3 = list(allData2)
    del allData3[0]
    allData4 = [float(numeric_string) for numeric_string in allData3]
    maxx = max(allData4)

    K = [i for i, x in enumerate(allData4) if x == maxx][0]
    maxQ = float(allData[K+1][2])
    
  return maxQ


#Determination of peak maxima
def dataMaximaChi(directory1):
  files100 = getFileList100(directory1)[0]
  with open(files100, "rb") as infile:
    reader = csv.reader(infile, delimiter=",")
    allData = list(reader)
    del allData[0]
    allData1 = zip(*allData)
    allData2 = allData1[2]
    allData3 = list(allData2)
    del allData3[0]
    allData4 = allData3[-10:]
    allData4 = [float(numeric_string) for numeric_string in allData4]
    allData4a = [float(numeric_string) for numeric_string in allData3]
    maxx = max(allData4)

    K = [i for i, x in enumerate(allData4a) if x == maxx][0]
    maxChi = float(allData[K+1][0])
    
  return maxChi


#Reading of parameters
def paramRead(directory1a):
  filesTxt = getFileListParams(directory1a + 'initial mar2300/')[0]
  filesTxt2 = getFileListCsv(directory1a + 'initial mar2300/')[0]
  index1 = filesTxt2.find('deg')
  incidenceAngle = np.divide(float(filesTxt2[index1-3:index1]), 100)

  allData = np.genfromtxt(filesTxt)
  sampleLength = allData[0]
  filmThickness = allData[1]

  with open(filesTxt2, "rb") as infile:
    reader = csv.reader(infile, delimiter=",")
    allData = list(reader)

    Monitor = int(allData[1][2])
  
  imageMax_1 = Monitor * 0.00000015 * sampleLength * filmThickness                 # no th norm
#  imageMax_1 = imageMax * Monitor / 7000000 / 9 * sampleLength * filmThickness / 500                      # th norm
  imageMax1 = imageMax_1 - (imageMax_1 % 1)
  imageMaxLog_1 = Monitor * 0.00000015 * sampleLength * filmThickness          # no th norm
#  imageMaxLog_1 = imageMaxLog * Monitor / 7000000 / 9 * sampleLength * filmThickness / 500                # th norm
  imageMaxLog1 = imageMaxLog_1 - (imageMaxLog_1 % 1)

  return imageMax1, imageMaxLog1, sampleLength, filmThickness, incidenceAngle, Monitor


#Data treatment - additional - extracting and saving the GIXD images, extracting xy cross-section
def dataTreat2(directory,calibFile,MaxChi,MaxQ,ImageMaximum,ImageMaximumLog):
  files = getFileList(directory + 'initial mar2300/')
  for dataFile in files:
    #Initialize the image object
    pic = wxdiff_api.wxdiff_diffimage(MDIRoot)
    #Open the file and calibrate it
    pic.fromfile(dataFile,filetypestr='MAR345',calibfname=calibFile, interactive=False)
    #
    directoryPics = directory + 'pics/'
    
    all_parameters = pic.get_image_properties()
    Wavelength = float(all_parameters[13][1])

    if not os.path.exists(directoryPics): os.makedirs(directoryPics)
    os.chdir(directoryPics)
    pngFILEjet = os.path.splitext(os.path.splitext(os.path.basename(dataFile))[0])[0] + '-nothnorm-jet.png'
    pngFILEbinary = os.path.splitext(os.path.splitext(os.path.basename(dataFile))[0])[0] + '-nothnorm-binary.png'
    pngFILEspec = os.path.splitext(os.path.splitext(os.path.basename(dataFile))[0])[0] + '-nothnorm-spec.png'
    pngFILEspecCrop = os.path.splitext(os.path.splitext(os.path.basename(dataFile))[0])[0] + '-nothnorm-spec-crop.png'

    processPeakZ(dataFile,pic,0,2.5,0,MaxChi,'xy')
    processPeak(dataFile,pic,MaxQ-0.013,MaxQ+0.013,0,180,'100-circle')
    
    pic2 = pic.convert_region_to_qxyqz()
    pic.close()
    pic2.set_image_options(vmax=ImageMaximum*2)
    pic2.set_viewed_region([0, 2170], [0, 1200])
    pic2.set_image_options(log=False)
    pic2.save_to_png(pngFILEjet)
    pic2.set_image_options(colormap='binary')
    pic2.save_to_png(pngFILEbinary)
    pic2.set_image_options(colormap='Spectral')
    pic2.save_to_png(pngFILEspec)

    pic2.set_image_options(vmax=ImageMaximum*2)
    pic2.set_viewed_region([850, 1200], [800, 1150])
    pic2.set_image_options(log=False)
    pic2.save_to_png(pngFILEspecCrop)
    
    pic2.close()
    return Wavelength


#Create empty folders in the sample directory
def makedirs(directory):
    mypath1 = directory + 'cross sections plots/'
    if not os.path.isdir(mypath1):
        os.makedirs(mypath1)
        
    mypath2 = directory + 'cross sections data/'
    if not os.path.isdir(mypath2):
        os.makedirs(mypath2)
    
    mypath3 = directory + 'pics/'
    if not os.path.isdir(mypath3):
        os.makedirs(mypath3)
        
    mypath4 = directory + 'rays/'
    if not os.path.isdir(mypath4):
        os.makedirs(mypath4)
    
    mypath5 = directory + 'rays/cs/'
    if not os.path.isdir(mypath5):
        os.makedirs(mypath5)
    
    mypath6 = directory + 'rays/cs-bgremoved/'
    if not os.path.isdir(mypath6):
        os.makedirs(mypath6)
        
    mypath7 = directory + '100 cake plots/'
    if not os.path.isdir(mypath7):
        os.makedirs(mypath7)
        
    mypath8 = directory + '100 cake data/'
    if not os.path.isdir(mypath8):
        os.makedirs(mypath8)
    
    mypath9 = directory + '200 cake plots/'
    if not os.path.isdir(mypath9):
        os.makedirs(mypath9)
        
    mypath10 = directory + '200 cake data/'
    if not os.path.isdir(mypath10):
        os.makedirs(mypath10)
        
    mypath11 = directory + '010 cake plots/'
    if not os.path.isdir(mypath11):
        os.makedirs(mypath11)
        
    mypath12 = directory + '010 cake data/'
    if not os.path.isdir(mypath12):
        os.makedirs(mypath12)
    
    mypath13 = directory + 'integrated plots/'
    if not os.path.isdir(mypath13):
        os.makedirs(mypath13)
        
    mypath14 = directory + 'integrated data/'
    if not os.path.isdir(mypath14):
        os.makedirs(mypath14)
        
    mypath15 = directory + '100 circle/'
    if not os.path.isdir(mypath15):
        os.makedirs(mypath15)
    



######################################### rays extraction, bgremoval and integration ######################################
###########################################################################################################################


#Save given data in given filename
def saveDataRays(filename, data): 
  data.save_to_csv(filename)


#Processing the data array as a set of 1 degree ray cake segments; including beamstop normalization, baseline subtraction and 100, 010 peak intensity calculation
def loadDataRays(filename, remBGname, noBgName, noBgName_dat, noBgName_norm, noBgName_dat_norm, parrrrams):
  data1Q = np.genfromtxt(filename, delimiter=',', skip_header=2)[100:,0] #90
  data2I = np.genfromtxt(filename, delimiter=',', skip_header=2)[100:,4]
  
  data1Q_100 = data1Q[10:90]
  data2I_100 = data2I[10:90]
  data1Q_010 = data1Q[440:760]
  data2I_010 = data2I[440:760]

  yyNoBG = remBGVis(data1Q, data2I, remBGname, ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.1)
  np.savetxt(noBgName_dat, tabMerge(data1Q, yyNoBG), delimiter=',')
  yyNoBG_norm = np.divide(yyNoBG, (parrrrams[2]*parrrrams[3]*parrrrams[5]*0.00000001))
  np.savetxt(noBgName_dat_norm, tabMerge(data1Q, yyNoBG_norm), delimiter=',')
  
  remBGname_100 = remBGname[:len(remBGname)-4] + '_100.pdf'
  remBGname_norm_100_dat = remBGname[:len(remBGname)-4] + '_100_norm.dat'
  noBgName_norm_100 = noBgName_norm[:len(remBGname)-4] + '_100.pdf'

  data2I_100_nobg = remBGVis(data1Q_100, data2I_100, remBGname_100, ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.01)
  data2I_100_norm = np.divide(data2I_100_nobg, (parrrrams[2]*parrrrams[3]*parrrrams[5]*0.00000001))
  np.savetxt(remBGname_norm_100_dat, tabMerge(data1Q_100, data2I_100_norm), delimiter=',')


  remBGname_010 = remBGname[:len(remBGname)-4] + '_010.pdf'
  remBGname_norm_010_dat = remBGname[:len(remBGname)-4] + '_010_norm.dat'
  noBgName_norm_010 = noBgName_norm[:len(remBGname)-4] + '_010.pdf'

  data2I_010_nobg = remBGVis(data1Q_010, data2I_010, remBGname_010, ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.5)
  data2I_010_norm = np.divide(data2I_010_nobg, (parrrrams[2]*parrrrams[3]*parrrrams[5]*0.00000001))
  np.savetxt(remBGname_norm_010_dat, tabMerge(data1Q_010, data2I_010_norm), delimiter=',')


  intensity100 = sum(data2I_100_norm[:])
  intensity100_err = np.sum(np.sqrt(np.absolute(data2I_100_norm[:])))
  intensity200 = sum(yyNoBG_norm[130:300])
  intensity200_err = np.sum(np.sqrt(np.absolute(yyNoBG_norm[130:300])))
  intensity010 = sum(data2I_010_norm[:])
  intensity010_err = np.sum(np.sqrt(np.absolute(data2I_100_norm[:])))

  
  return intensity100, intensity100_err, intensity200, intensity200_err, intensity010, intensity010_err



#Getting row sum from given dataFile for given Chi and Q values
def getRowSum(pic, Q1, Q2, Chi1, Chi2):
  (X1, Y1) = pic.get_XY_from_QChi(Q1,Chi1)
  (X2, Y2) = pic.get_XY_from_QChi(Q2,Chi2)
  selected = pic.add_cakeseg(X1,Y1,X2,Y2)
  selected = selected.convert_to_qchi()
  colSum = selected.column_sum()
  selected.close()
  return colSum


#Processing the given peak
def processPeakRays1(directory,dataFile,pic,Q1,Q2,Chi1,Chi2):
    integrated = getRowSum(pic,Q1,Q2,Chi1,Chi2)
    str1=str(Chi1)
    if len(str1)==1:
        str1='0'+str1
    toSave = directory + 'rays/cs/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1
    saveDataRays(toSave, integrated)
    integrated.close()
#    return integrated


def processPeakRays2(directory,dataFile,Chi1,parrrams):
    str1=str(Chi1)
    if len(str1)==1:
        str1='0'+str1
    toSave = directory + 'rays/cs/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1

    remBgName = directory + 'rays/cs-bgremoved/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1 + '_bgvis.pdf'
    noBgName = directory + 'rays/cs-bgremoved/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1 + '_nobg.pdf'
    noBgName_dat = directory + 'rays/cs-bgremoved/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1 + '_nobg.dat'
    noBgName_norm = directory + 'rays/cs-bgremoved/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1 + '_nobg_norm.pdf'
    noBgName_dat_norm = directory + 'rays/cs-bgremoved/' + os.path.splitext(os.path.basename(dataFile))[0] + '-' + str1 + '_nobg_norm.dat'
    intens100, intens100_err, intens200, intens200_err, intens010, intens010_err = loadDataRays(toSave, remBgName, noBgName, noBgName_dat, noBgName_norm, noBgName_dat_norm, parrrams)

    return intens100, intens100_err, intens200, intens200_err, intens010, intens010_err


#Extraction and/or processing of the 1 degree ray cake segments for a given data file 
def data_rays(directory,calibFile,mode1, parrams):
  files = getFileList(directory + 'initial mar2300/')
  os.chdir(directory + 'rays/cs/')
  for dataFile in files:
    #Initialize the image object
    pic = wxdiff_api.wxdiff_diffimage(MDIRoot)
    #Open the file and calibrate it
    pic.fromfile(dataFile,filetypestr='MAR345',calibfname=calibFile, interactive=False)
    #Log scale so it is visible
    pic.set_image_options(log=True)
    cakeNoBG100=[]
    cakeNoBG200=[]
    cakeNoBG010=[]
    cakeNoBG100_total=0
    cakeNoBG200_total=0
    cakeNoBG010_total=0
    cakeNoBG100_total_err=0
    cakeNoBG200_total_err=0
    cakeNoBG010_total_err=0

    if mode1=='extraction':
        for tt in np.add(90, range(90)):
            processPeakRays1(directory,dataFile,pic,0.0001,2.6,tt,tt+1)
    elif mode1=='processing':
        for tt in np.add(90, range(90)):
            intens100, intens100_err, intens200, intens200_err, intens010, intens010_err = processPeakRays2(directory,dataFile,tt,parrams)
            cakeNoBG100.append(intens100)
            cakeNoBG100_total = cakeNoBG100_total + intens100
            cakeNoBG100_total_err = cakeNoBG100_total_err + intens100_err
            cakeNoBG200.append(intens200)
            cakeNoBG200_total = cakeNoBG200_total + intens200
            cakeNoBG200_total_err = cakeNoBG200_total_err + intens200_err
            cakeNoBG010.append(intens010)
            cakeNoBG010_total = cakeNoBG010_total + intens010
            cakeNoBG010_total_err = cakeNoBG010_total_err + intens010_err


        cakeNoBg100_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_100_cake_nobg_rays.dat'
        cakeNoBg100_total_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_100_cake_nobg_rays_total.dat'
        cakeNoBg100_FWHM_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_100_cake_nobg_rays_fwhm.dat1'
        cakeNoBg100_PdfName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_100_cake_nobg_rays.pdf'

        cakeNoBg200_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_200_cake_nobg_rays.dat'
        cakeNoBg200_total_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_200_cake_nobg_rays_total.dat'
        cakeNoBg200_FWHM_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_200_cake_nobg_rays_fwhm.dat1'
        cakeNoBg200_PdfName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_200_cake_nobg_rays.pdf'

        cakeNoBg010_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_010_cake_nobg_rays.dat'
        cakeNoBg010_total_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_010_cake_nobg_rays_total.dat'
        cakeNoBg010_FWHM_DatName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_010_cake_nobg_rays_fwhm.dat1'
        cakeNoBg010_PdfName = directory + 'rays/' + os.path.splitext(os.path.basename(dataFile))[0] + '_010_cake_nobg_rays.pdf'
    
        np.savetxt(cakeNoBg100_DatName, cakeNoBG100, delimiter=',')
        np.savetxt(cakeNoBg100_total_DatName, [cakeNoBG100_total, cakeNoBG100_total_err], delimiter=',')
        np.savetxt(cakeNoBg200_DatName, cakeNoBG200, delimiter=',')
        np.savetxt(cakeNoBg200_total_DatName, [cakeNoBG200_total, cakeNoBG200_total_err], delimiter=',')
        np.savetxt(cakeNoBg010_DatName, cakeNoBG010, delimiter=',')
        np.savetxt(cakeNoBg010_total_DatName, [cakeNoBG010_total, cakeNoBG010_total_err], delimiter=',')

        cakeNoBG100 = cakeNoBG100[::-1] + cakeNoBG100
        cakeNoBG200 = cakeNoBG200[::-1] + cakeNoBG200
        cakeNoBG010 = cakeNoBG010[::-1] + cakeNoBG010

        popt100, pcov100, gauss_data100 = fitVis(np.subtract(range(len(cakeNoBG100)),90), cakeNoBG100, 0, max(cakeNoBG100), 0, 10, ur'$\u03C7$ (deg)', cakeNoBg100_PdfName)
        popt200, pcov200, gauss_data200 = fitVis(np.subtract(range(len(cakeNoBG200)),90), cakeNoBG200, 0, max(cakeNoBG200), 0, 10, ur'$\u03C7$ (deg)', cakeNoBg200_PdfName)
        popt010, pcov010, gauss_data010 = fitVis(np.subtract(range(len(cakeNoBG010)),90), cakeNoBG010, 0, max(cakeNoBG010), 0, 10, ur'$\u03C7$ (deg)', cakeNoBg010_PdfName)
    
    pic.close()



########################################## z and xy cross-section analysis ##################################################
#############################################################################################################################

#Save given data in given filename
def saveDataCS(remBgName, noBgName, noBgName_dat, noBgName_norm, noBgName_dat_norm, fitVisName, noBgName_dat_norm_d, noBgName_dat_norm_L, noBgName_dat_norm_i, FragmentQ, FragmentI, num_Peaks, peaks_Pos, paramet):
  
  SampleLength = paramet[2]
  FilmThickness = paramet[3]
  Monitor = paramet[5]
  waveleng = paramet[6]
  yyNoBG = remBGVis(FragmentQ, FragmentI, remBgName,  ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.001)
  np.savetxt(noBgName_dat, tabMerge(FragmentQ, yyNoBG), delimiter=',')
  yyNoBG_norm = np.divide(yyNoBG, (SampleLength*0.1*FilmThickness*Monitor*0.0000001))
  np.savetxt(noBgName_dat_norm, tabMerge(FragmentQ, yyNoBG_norm), delimiter=',')

  norm_coeffi = []
  norm_coeffi.append(SampleLength*0.1*FilmThickness*Monitor*0.0000001)
  norm_coeffi.append(SampleLength)
  norm_coeffi.append(FilmThickness)
  norm_coeffi.append(Monitor)

  noBGVis(FragmentQ, yyNoBG, noBgName, ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.001)
  noBGVis(FragmentQ, yyNoBG_norm, noBgName_norm, ur"q ($\mathregular{\u00c5^{-1}}$)", 1, 0.001)
  
  peakIntensity = sum(yyNoBG_norm)
  peakIntensity_err = np.sum(np.sqrt(np.absolute(yyNoBG_norm)))
  
  if num_Peaks == 1:
    peaks_pos1 = [np.argmax(yyNoBG_norm)]
    popt, pcov, gauss_data = fitVis(FragmentQ, yyNoBG_norm, 0, yyNoBG_norm[peaks_pos1[0]], FragmentQ[peaks_pos1[0]], 0.05, ur"q ($\mathregular{\u00c5^{-1}}$)", fitVisName)
    parametersCalc(popt, pcov, waveleng, gauss_data, noBgName_dat_norm_d, noBgName_dat_norm_L, noBgName_dat_norm_i, norm_coeffi)
  elif num_Peaks == 2:
    popt, pcov, gauss_data, ratio_12 = fitVis2(FragmentQ, yyNoBG_norm, 0, yyNoBG_norm[peaks_Pos[0]], FragmentQ[peaks_Pos[0]], 0.005, yyNoBG_norm[peaks_Pos[1]], FragmentQ[peaks_Pos[1]], 0.005, ur"q ($\mathregular{\u00c5^{-1}}$)", fitVisName)
    parametersCalc2(popt, pcov, waveleng, gauss_data, ratio_12, noBgName_dat_norm_d, noBgName_dat_norm_L, noBgName_dat_norm_i)



#Processing the different cross-sections
def processCS(directory, cs_type, num_peaks_, peak_pos_, parame):

    if cs_type=='Z':
        DirToRead = directory + 'cross sections data/'
        DirToPlot = directory + 'cross sections plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-Z.csv')[0]
        CrossSectionQ = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=1)[:,3]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=1)[:,5]
    elif cs_type=='xy':
        DirToRead = directory + 'cross sections data/'
        DirToPlot = directory + 'cross sections plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-xy.csv')[0]
        CrossSectionQ = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=1)[100:,3]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=1)[100:,5]
    elif cs_type=='integrated':
        DirToRead = directory + 'integrated data/'
        DirToPlot = directory + 'integrated plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-integrated.csv')[0]
        CrossSectionQ = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[100:,0]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[100:,4]
    

    Fragment100Q = CrossSectionQ[10:120] 
    Fragment100I = CrossSectionI[10:120] 
    Fragment200Q = CrossSectionQ[140:310]
    Fragment200I = CrossSectionI[140:310]
    Fragment010Q = CrossSectionQ[470:700] 
    Fragment010I = CrossSectionI[470:700]


    if cs_type=='Z':
        if num_peaks_==1:
            peak_pos_=peak_pos_[0]
        elif num_peaks_==2:
            peak_pos_=np.zeros(2)
            peak_pos_[0]=30   
            peak_pos_[1]=43   

    if cs_type=='xy':
        if num_peaks_==1:
            peak_pos_=peak_pos_[0]
   
    if cs_type=='integrated':
        if num_peaks_==1:
            peak_pos_=peak_pos_[0]
        elif num_peaks_==2:
            peak_pos_=np.zeros(2)
            peak_pos_[0]=25 
            peak_pos_[1]=62 
            
        
    
    remBgName100 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_bgvis.pdf'
    fitVisName100 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_fitvis.pdf'
    noBgName100 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg.pdf'
    noBgName_dat100 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg.dat'
    noBgName_norm100 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg_norm.pdf'
    noBgName_dat_norm100 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg_norm.dat'
    noBgName_dat_norm_d_100 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg_norm_d.dat'
    noBgName_dat_norm_L_100 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg_norm_L.dat'
    noBgName_dat_norm_i_100 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_nobg_norm_i.dat'

    remBgName200 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_bgvis.pdf'
    fitVisName200 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_fitvis.pdf'
    noBgName200 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg.pdf'
    noBgName_dat200 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg.dat'
    noBgName_norm200 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg_norm.pdf'
    noBgName_dat_norm200 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg_norm.dat'
    noBgName_dat_norm_d_200 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg_norm_d.dat'
    noBgName_dat_norm_L_200 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg_norm_L.dat'
    noBgName_dat_norm_i_200 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_nobg_norm_i.dat'

    remBgName010 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_bgvis.pdf'
    fitVisName010 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_fitvis.pdf'
    noBgName010 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg.pdf'
    noBgName_dat010 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg.dat'
    noBgName_norm010 = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg_norm.pdf'
    noBgName_dat_norm010 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg_norm.dat'
    noBgName_dat_norm_d_010 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg_norm_d.dat'
    noBgName_dat_norm_L_010 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg_norm_L.dat'
    noBgName_dat_norm_i_010 = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_nobg_norm_i.dat'


    remBgName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_bgvis.pdf'
    fitVisName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_fitvis.pdf'
    noBgName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg.pdf'
    noBgName_dat = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg.dat'
    noBgName_norm = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg_norm.pdf'
    noBgName_dat_norm = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg_norm.dat'
    noBgName_dat_norm_d = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg_norm_d.dat'
    noBgName_dat_norm_L = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg_norm_L.dat'
    noBgName_dat_norm_i = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_nobg_norm_i.dat'

    saveDataCS(remBgName100, noBgName100, noBgName_dat100, noBgName_norm100, noBgName_dat_norm100, fitVisName100, noBgName_dat_norm_d_100, noBgName_dat_norm_L_100, noBgName_dat_norm_i_100, Fragment100Q, Fragment100I, num_peaks_, peak_pos_, parame)
    saveDataCS(remBgName200, noBgName200, noBgName_dat200, noBgName_norm200, noBgName_dat_norm200, fitVisName200, noBgName_dat_norm_d_200, noBgName_dat_norm_L_200, noBgName_dat_norm_i_200, Fragment200Q, Fragment200I, 1, peak_pos_, parame)
    saveDataCS(remBgName010, noBgName010, noBgName_dat010, noBgName_norm010, noBgName_dat_norm010, fitVisName010, noBgName_dat_norm_d_010, noBgName_dat_norm_L_010, noBgName_dat_norm_i_010, Fragment010Q, Fragment010I, 1, peak_pos_, parame)


    return num_peaks_, peak_pos_


    

#################   bgremoval, fitting, parameters extraction ###############################

def repmat1(matrixA, rowFinal, colFinal):
    matrixB = np.array(matrixA)
    finMat=[]
    for i in range(colFinal):
        finMat.append(matrixB.transpose())
    finMat2 = np.array(finMat)
    return finMat2.transpose() 

def remBG(z_cs_Q_100_fr, z_cs_100_fr, ord_, s):
    DataZx = z_cs_Q_100_fr
    DataZy = z_cs_100_fr
    n = DataZx
    y = DataZy

    x1 = DataZx
    y11 = DataZy
    
    fct = 'atq'
    
    # rescaling
    
    N = len(n)
    maxy = max(y)
    dely = (maxy-min(y))/2
    n_=[]
    y_=[]
    for kk in n:
        kk = 2 * (kk-n[N-1]) / (n[N-1]-n[0]) + 1
        n_.append(kk)
    for kk in y:
        kk = (kk-maxy)/dely + 1
        y_.append(kk)


    n1 = n_
    y1 = y_
    
    # Vandermonde matrix
    p = range(0,ord_+1)
    Te = repmat1(n_,1,ord_+1) ** np.tile(p,(N,1))    
    Teinv = np.dot(np.linalg.pinv(np.dot(Te.T,Te)), Te.T)                      
        # Initialisation (least-squares estimation)
    a = np.dot(Teinv,y_)  
    z = np.dot(Te,a) 
    
        # Other variables
    alpha = 0.99 * 1/2      # Scale parameter alpha
    it = 0                  # Iteration number
    zp = [1]*N #np.ones(N,1)          # Previous estimation       = [1]*N

    while sum(np.power((z-zp),2))/sum(np.power(zp,2)) > 1e-9:
        it = it + 1            # Iteration number
        zp = z                 # Previous estimation
        res = y_ - z            # Residual    
        # Estimate d
        if fct == 'sh':
            d = (res*(2*alpha-1)) * (abs(res)<s) + (-alpha*2*s-res) * (res<=-s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'ah':
            d = (res*(2*alpha-1)) * (res<s) + (alpha*2*s-res) * (res>=s)
        elif fct == 'stq':
            d = (res*(2*alpha-1)) * (abs(res)<s) - res * (abs(res)>=s)
        elif fct == 'atq':
            d = (res*(2*alpha-1)) * (res<s) - res * (res>=s)
        # Estimate z
        a = np.dot(Teinv, (y_+d))   # Polynomial coefficients a
        z = np.dot(Te,a)           # Polynomial
    # Rescaling
    z1=[]
    for j1 in z:
        z1.append((j1-1)*dely + maxy)

    yNoBG = np.subtract(y11,z1)
    return z1, yNoBG


def remBGVis(z_cs_Q_100_fragment, z_cs_100_fragment, remBGname, xxName, ordd, ss):
    [yyBGLine, yyNoBG] = remBG(z_cs_Q_100_fragment, z_cs_100_fragment, ordd, ss)
    figBGLine = plt.figure()
    axBGLine = figBGLine.add_subplot(111)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel(xxName)
    plt.plot(z_cs_Q_100_fragment, z_cs_100_fragment, 'b', linewidth=2)
    plt.plot(z_cs_Q_100_fragment, yyBGLine, 'r', linewidth=2)
    plt.grid(True)
    axBGLine.tick_params(axis='x', which='major', pad=15)
    axBGLine.xaxis.tick_bottom()
    axBGLine.tick_params(axis='y', which='major', pad=15)
    axBGLine.yaxis.tick_left() 
    for item in ([axBGLine.title, axBGLine.xaxis.label, axBGLine.yaxis.label] + axBGLine.get_xticklabels() + axBGLine.get_yticklabels()):
        item.set_fontsize(20)
    for axis in ['bottom','left']:
        axBGLine.spines[axis].set_linewidth(2)
    axBGLine.xaxis.set_major_locator(MaxNLocator(5))
    axBGLine.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()

    figBGLine.savefig(remBGname, format='pdf') 
    plt.close(figBGLine)
    return yyNoBG

def noBGVis(z_cs_Q_100_fragment, z_cs_100_fragment, noBGname, xxName, ordd, ss):
    [yyBGLine, yyNoBG] = remBG(z_cs_Q_100_fragment, z_cs_100_fragment, ordd, ss)
    figNoBG = plt.figure()
    axNoBG = figNoBG.add_subplot(111)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel(xxName)
    plt.plot(z_cs_Q_100_fragment, yyNoBG, 'b', linewidth=2)
    plt.grid(True)
    axNoBG.tick_params(axis='x', which='major', pad=15)
    axNoBG.xaxis.tick_bottom()
    axNoBG.tick_params(axis='y', which='major', pad=15)
    axNoBG.yaxis.tick_left()
    for item in ([axNoBG.title, axNoBG.xaxis.label, axNoBG.yaxis.label] + axNoBG.get_xticklabels() + axNoBG.get_yticklabels()):
        item.set_fontsize(20)
    for axis in ['bottom','left']:
        axNoBG.spines[axis].set_linewidth(2)
    axNoBG.xaxis.set_major_locator(MaxNLocator(5))
    axNoBG.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()

    figNoBG.savefig(noBGname, format='pdf') 
    plt.close(figNoBG)

#Interpolation with a single Gaussian peak
def fitVis(z_cs_2theta_100_fr, z_cs_100_fr, y0, A0, x0, fwhm0, xxLabel, fNameZPdf2th100):
    popt1, pcov1 = curve_fit(gauss, z_cs_2theta_100_fr, z_cs_100_fr, [y0, A0, x0, fwhm0])
    Fit2th = [ gauss(x, popt1[0], popt1[1], popt1[2], popt1[3]) for x in z_cs_2theta_100_fr ]
    
    fig_z_cs_100_fragment = plt.figure()
    ax_z_cs_100_fragment = fig_z_cs_100_fragment.add_subplot(111)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel(xxLabel)
        
    plt.plot(z_cs_2theta_100_fr, z_cs_100_fr, 'b', linewidth=2)
    plt.plot(z_cs_2theta_100_fr, Fit2th, 'r', linewidth=1.5)
    plt.grid(True)
    ax_z_cs_100_fragment.tick_params(axis='x', which='major', pad=15)
    ax_z_cs_100_fragment.xaxis.tick_bottom()
    ax_z_cs_100_fragment.tick_params(axis='y', which='major', pad=15)
    ax_z_cs_100_fragment.yaxis.tick_left()
    for item in ([ax_z_cs_100_fragment.title, ax_z_cs_100_fragment.xaxis.label, ax_z_cs_100_fragment.yaxis.label] + ax_z_cs_100_fragment.get_xticklabels() + ax_z_cs_100_fragment.get_yticklabels()):
        item.set_fontsize(20)
    for axis in ['bottom','left']:
        ax_z_cs_100_fragment.spines[axis].set_linewidth(2)
    ax_z_cs_100_fragment.xaxis.set_major_locator(MaxNLocator(5))
    ax_z_cs_100_fragment.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()

    fig_z_cs_100_fragment.savefig(fNameZPdf2th100, format='pdf')     
    
    gauss_data = np.subtract(Fit2th, popt1[0])
        
    return popt1, pcov1, gauss_data

#Interpolation with two Gaussian peaks
def fitVis2(z_cs_2theta_100_fr, z_cs_100_fr, y0, A0, x0, fwhm0, A1, x1, fwhm1, xxLabel, fNameZPdf2th100):
    popt1, pcov1 = curve_fit(gauss2, z_cs_2theta_100_fr, z_cs_100_fr, [y0, A0, x0, fwhm0, A1, x1, fwhm1])
    Fit2th = [ gauss2(x, popt1[0], popt1[1], popt1[2], popt1[3], popt1[4], popt1[5], popt1[6]) for x in z_cs_2theta_100_fr ]

    Fit2th_1 = [ gauss(x, popt1[0], popt1[1], popt1[2], popt1[3]) for x in z_cs_2theta_100_fr ]
    Fit2th_2 = [ gauss(x, popt1[0], popt1[4], popt1[5], popt1[6]) for x in z_cs_2theta_100_fr ]

    ratio_1_2 = np.divide(sum(np.subtract(Fit2th_2, popt1[0])), sum(np.subtract(Fit2th, popt1[0])))
    
    fig_z_cs_100_fragment = plt.figure()
    ax_z_cs_100_fragment = fig_z_cs_100_fragment.add_subplot(111)
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel(xxLabel)
        
    plt.plot(z_cs_2theta_100_fr, z_cs_100_fr, 'b', linewidth=2)
    plt.plot(z_cs_2theta_100_fr, Fit2th, 'r', linewidth=1.5)
    plt.plot(z_cs_2theta_100_fr, Fit2th_1, 'r--', linewidth=0.5)
    plt.plot(z_cs_2theta_100_fr, Fit2th_2, 'r--', linewidth=0.5)
    plt.grid(True)
    ax_z_cs_100_fragment.tick_params(axis='x', which='major', pad=15)
    ax_z_cs_100_fragment.xaxis.tick_bottom()
    ax_z_cs_100_fragment.tick_params(axis='y', which='major', pad=15)
    ax_z_cs_100_fragment.yaxis.tick_left()
    for item in ([ax_z_cs_100_fragment.title, ax_z_cs_100_fragment.xaxis.label, ax_z_cs_100_fragment.yaxis.label] + ax_z_cs_100_fragment.get_xticklabels() + ax_z_cs_100_fragment.get_yticklabels()):
        item.set_fontsize(20)
    for axis in ['bottom','left']:
        ax_z_cs_100_fragment.spines[axis].set_linewidth(2)
    ax_z_cs_100_fragment.xaxis.set_major_locator(MaxNLocator(5))
    ax_z_cs_100_fragment.yaxis.set_major_locator(MaxNLocator(5))
    plt.tight_layout()

    fig_z_cs_100_fragment.savefig(fNameZPdf2th100, format='pdf') 
    
    gauss_data = np.subtract(Fit2th, popt1[0])
    
    return popt1, pcov1, gauss_data, ratio_1_2
 
#Calculation of the crystallographic parameters (d-spacing, coherence length, intensity) and errors thereof based on the interpolation values
#of a single Gaussian peak
def parametersCalc(popt1, pcov1, waveleng, yFitP1, dSpacName, LName, iName, norm_coeffii):
    d1a = 2*math.pi/popt1[2]
    try: 
      d1a_err = (2*math.pi/(popt1[2]-math.sqrt(abs(pcov1[2,2]))) - 2*math.pi/(popt1[2]+math.sqrt(abs(pcov1[2,2]))))/2
    except:
      d1a_err = 0
    
    th1a = math.asin(popt1[2]*waveleng/(4*math.pi))/math.pi*180
    coeff = 2*math.sqrt(2*(math.log(2)))
    fwhm1a = ((math.asin((popt1[2]+(abs(popt1[3])*coeff/2))*waveleng/(4*math.pi))/math.pi*180) - (math.asin((popt1[2]-(abs(popt1[3])*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    LName_FWHM = LName[:-4] + '_FWHM.dat'
    L1a = (0.09*waveleng)/((fwhm1a*math.pi/180)*math.cos(th1a/180*math.pi))
    
    try: 
      th1a_max = math.asin((popt1[2]+math.sqrt(pcov1[2,2]))*waveleng/(4*math.pi))/math.pi*180
      th1a_min = math.asin((popt1[2]-math.sqrt(pcov1[2,2]))*waveleng/(4*math.pi))/math.pi*180
    except:
      th1a_max = th1a
      th1a_min = th1a

    coeff = 2*math.sqrt(2*(math.log(2)))

    try:
      fwhm1a_max = ((math.asin(((popt1[2]+math.sqrt(pcov1[2,2]))+((abs(popt1[3])+math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[2]-math.sqrt(pcov1[2,2]))-((abs(popt1[3])+math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
      fwhm1a_min = ((math.asin(((popt1[2]-math.sqrt(pcov1[2,2]))+((abs(popt1[3])-math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[2]+math.sqrt(pcov1[2,2]))-((abs(popt1[3])-math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    except:
      fwhm1a_max = fwhm1a
      fwhm1a_min = fwhm1a

    np.savetxt(LName_FWHM, [fwhm1a, fwhm1a_max-fwhm1a])

    try:
      L1a_max = (0.09*waveleng)/((fwhm1a_min*math.pi/180)*math.cos(th1a_max/180*math.pi))
      L1a_min = (0.09*waveleng)/((fwhm1a_max*math.pi/180)*math.cos(th1a_min/180*math.pi))
    except:
      L1a_max = L1a
      L1a_min = L1a

    L1a_err = (L1a_max - L1a_min)/2

    yFitP1_nonorm = np.multiply(yFitP1, norm_coeffii[0])

    i1 = sum(yFitP1)
    i1_err = np.divide(sum(np.sqrt(yFitP1_nonorm)), norm_coeffii[0])
    print 'Errors:'
    print i1_err + i1*(np.divide(0.1, norm_coeffii[1]) + np.divide(3, norm_coeffii[2]) + np.divide(math.sqrt(norm_coeffii[3]), norm_coeffii[3]))
    print i1*(np.divide(0.1, norm_coeffii[1]))
    print i1*(np.divide(1, norm_coeffii[2]))
    print i1*(np.divide(math.sqrt(norm_coeffii[3]), norm_coeffii[3]))    

    i1_err = i1_err + i1*(np.divide(0.1, norm_coeffii[1]) + np.divide(1, norm_coeffii[2]) + np.divide(math.sqrt(norm_coeffii[3]), norm_coeffii[3]))
    
    paracryst = math.sqrt(float(abs(popt1[3])) / (2*math.pi*popt1[2]))
    try:
      para_min = math.sqrt(float(abs(popt1[3])-math.sqrt(pcov1[3,3])) / (2*math.pi*(popt1[2]+math.sqrt(pcov1[2,2]))))
      para_max = math.sqrt(float(abs(popt1[3])+math.sqrt(pcov1[3,3])) / (2*math.pi*(popt1[2]-math.sqrt(pcov1[2,2]))))
    except:
      para_min = paracryst
      para_max = paracryst

    para_err = float(para_max-para_min)/2
    para_err2 = para_max - paracryst 
    delta1 = abs(popt1[3])
    qu1 = abs(popt1[2])
    try:
      dedelta2 = math.sqrt(pcov1[3,3])
      dequ2 = math.sqrt(pcov1[2,2])
    except:
      dedelta2 = 0
      dequ2 = 0

    c1 = math.sqrt(float(0.5)/math.pi)
    para_err3 = 0.5*c1*(qu1**(-0.5))*(delta1**(-0.5))*dedelta2 + (-0.5)*c1*(delta1**(0.5))*(qu1**(-1.5))*dequ2

    dSpacings = [d1a, d1a_err, 0, 0, paracryst, para_err, para_err3]
    cohLengths = [L1a, L1a_err, 0, 0]
    totIntensities = [i1, i1_err]
    
    np.savetxt(dSpacName, dSpacings, delimiter=',')
    np.savetxt(LName, cohLengths, delimiter=',')
    np.savetxt(iName, totIntensities, delimiter=',')


#Calculation of the crystallographic parameters (d-spacing, coherence length, intensity) and errors thereof based on the interpolation values
#of two Gaussian peaks
def parametersCalc2(popt1, pcov1, waveleng, yFitP1, ratio12, dSpacName, LName, iName):
    d1a = 2*math.pi/popt1[2]
    d1a_err = (2*math.pi/(popt1[2]-math.sqrt(pcov1[2,2])) - 2*math.pi/(popt1[2]+math.sqrt(pcov1[2,2])))/2
    
    d2a = 2*math.pi/popt1[5]
    d2a_err = (2*math.pi/(popt1[5]-math.sqrt(pcov1[5,5])) - 2*math.pi/(popt1[5]+math.sqrt(pcov1[5,5])))/2
    
    th1a = math.asin(popt1[2]*waveleng/(4*math.pi))/math.pi*180
    coeff = 2*math.sqrt(2*(math.log(2)))
    fwhm1a = ((math.asin((popt1[2]+(abs(popt1[3])*coeff/2))*waveleng/(4*math.pi))/math.pi*180) - (math.asin((popt1[2]-(abs(popt1[3])*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    L1a = (0.09*waveleng)/((fwhm1a*math.pi/180)*math.cos(th1a/180*math.pi))

    th2a = math.asin(popt1[5]*waveleng/(4*math.pi))/math.pi*180
    coeff = 2*math.sqrt(2*(math.log(2)))
    fwhm2a = ((math.asin((popt1[5]+(abs(popt1[6])*coeff/2))*waveleng/(4*math.pi))/math.pi*180) - (math.asin((popt1[5]-(abs(popt1[6])*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    L2a = (0.09*waveleng)/((fwhm2a*math.pi/180)*math.cos(th1a/180*math.pi))

    
    th1a_max = math.asin((popt1[2]+math.sqrt(pcov1[2,2]))*waveleng/(4*math.pi))/math.pi*180
    th1a_min = math.asin((popt1[2]-math.sqrt(pcov1[2,2]))*waveleng/(4*math.pi))/math.pi*180
    coeff = 2*math.sqrt(2*(math.log(2)))
    fwhm1a_max = ((math.asin(((popt1[2]+math.sqrt(pcov1[2,2]))+((abs(popt1[3])+math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[2]-math.sqrt(pcov1[2,2]))-((abs(popt1[3])+math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    fwhm1a_min = ((math.asin(((popt1[2]-math.sqrt(pcov1[2,2]))+((abs(popt1[3])-math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[2]+math.sqrt(pcov1[2,2]))-((abs(popt1[3])-math.sqrt(pcov1[3,3]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    L1a_max = (0.09*waveleng)/((fwhm1a_min*math.pi/180)*math.cos(th1a_max/180*math.pi))
    L1a_min = (0.09*waveleng)/((fwhm1a_max*math.pi/180)*math.cos(th1a_min/180*math.pi))
    L1a_err = (L1a_max - L1a_min)/2
    
    th2a_max = math.asin((popt1[5]+math.sqrt(pcov1[5,5]))*waveleng/(4*math.pi))/math.pi*180
    th2a_min = math.asin((popt1[5]-math.sqrt(pcov1[5,5]))*waveleng/(4*math.pi))/math.pi*180
    coeff = 2*math.sqrt(2*(math.log(2)))
    fwhm2a_max = ((math.asin(((popt1[5]+math.sqrt(pcov1[5,5]))+((abs(popt1[6])+math.sqrt(pcov1[6,6]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[5]-math.sqrt(pcov1[5,5]))-((abs(popt1[6])+math.sqrt(pcov1[6,6]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    fwhm2a_min = ((math.asin(((popt1[5]-math.sqrt(pcov1[5,5]))+((abs(popt1[6])-math.sqrt(pcov1[6,6]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180)-(math.asin(((popt1[5]+math.sqrt(pcov1[5,5]))-((abs(popt1[6])-math.sqrt(pcov1[6,6]))*coeff/2))*waveleng/(4*math.pi))/math.pi*180))*2
    L2a_max = (0.09*waveleng)/((fwhm2a_min*math.pi/180)*math.cos(th1a_max/180*math.pi))
    L2a_min = (0.09*waveleng)/((fwhm2a_max*math.pi/180)*math.cos(th1a_min/180*math.pi))
    L2a_err = (L2a_max - L2a_min)/2

    i1 = sum(yFitP1)
    i1_err = sum(np.sqrt(yFitP1))
    
    dSpacings = [d1a, d1a_err, d2a, d2a_err]
    cohLengths = [L1a, L1a_err, L2a, L2a_err]
    totIntensities = [i1, i1_err, ratio12]
    
    np.savetxt(dSpacName, dSpacings, delimiter=',')
    np.savetxt(LName, cohLengths, delimiter=',')
    np.savetxt(iName, totIntensities, delimiter=',')





##########################################################################################
############################# cake normalization + fitting ###############################

#Processing the cake segment of a given peak
def processCake(directory, cake_type):

    if cake_type=='100':
        DirToRead = directory + '100 cake data/'
        DirToPlot = directory + '100 cake plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-100.csv')[0]
        CrossSectionChi = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,0]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,2]
        CrossSectionI_norm = np.divide(CrossSectionI, (SampleLength*0.1*FilmThickness))
        
        cakeNorm_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_cake_norm.dat'
        np.savetxt(cakeNorm_DatName, tabMerge(CrossSectionChi, CrossSectionI_norm), delimiter=',')

        cakeFit_PdfName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_cake.pdf'
        popt, pcov, gauss_data = fitVis(np.subtract(CrossSectionChi,90), CrossSectionI, 0, max(CrossSectionI), 0, 50, ur'$\u03C7$ (deg)', cakeFit_PdfName)
        cakeFWHM_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_cake_fwhm.dat'
        np.savetxt(cakeFWHM_DatName, [popt[3], math.sqrt(pcov[3,3])], delimiter=',')
    elif cake_type=='200':
        DirToRead = directory + '200 cake data/'
        DirToPlot = directory + '200 cake plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-200.csv')[0]
        CrossSectionChi = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,0]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,2]
        CrossSectionI_norm = np.divide(CrossSectionI, (SampleLength*0.1*FilmThickness))
        
        cakeNorm_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_cake_norm.dat'
        np.savetxt(cakeNorm_DatName, tabMerge(CrossSectionChi, CrossSectionI_norm), delimiter=',')

        cakeFit_PdfName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_cake.pdf'
        popt, pcov, gauss_data = fitVis(np.subtract(CrossSectionChi,90), CrossSectionI, 0, max(CrossSectionI), 0, 50, ur'$\u03C7$ (deg)', cakeFit_PdfName)
        cakeFWHM_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_200_cake_fwhm.dat'
        np.savetxt(cakeFWHM_DatName, [popt[3], math.sqrt(pcov[3,3])], delimiter=',')
    elif cake_type=='010':
        DirToRead = directory + '010 cake data/'
        DirToPlot = directory + '010 cake plots/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-010.csv')[0]
        CrossSectionChi = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,0]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,2]
        CrossSectionI_norm = np.divide(CrossSectionI, (SampleLength*0.1*FilmThickness))
        
        cakeNorm_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_010_cake_norm.dat'
        np.savetxt(cakeNorm_DatName, tabMerge(CrossSectionChi, CrossSectionI_norm), delimiter=',')

    elif cake_type=='100 circle':
        DirToRead = directory + '100 circle/'
        DirToPlot = directory + '100 circle/'
        os.chdir(DirToRead)
        CrossSectionFile = glob.glob(DirToRead + '*-circle.csv')[0]
        CrossSectionChi = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,0]
        CrossSectionI = np.genfromtxt(CrossSectionFile, delimiter=',', skip_header=2)[:,2]
        CrossSectionI_norm = np.divide(CrossSectionI, (SampleLength*0.1*FilmThickness))
        
        cakeNorm_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_circle_norm.dat'
        np.savetxt(cakeNorm_DatName, tabMerge(CrossSectionChi, CrossSectionI_norm), delimiter=',')

        cakeFit_PdfName = DirToPlot + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_circle.pdf'
        popt, pcov, gauss_data = fitVis(np.subtract(CrossSectionChi,90), CrossSectionI, 0, max(CrossSectionI), 0, 50, ur'$\u03C7$ (deg)', cakeFit_PdfName)
        cakeFWHM_DatName = DirToRead + os.path.splitext(os.path.basename(CrossSectionFile))[0] + '_100_circle_fwhm.dat'
        np.savetxt(cakeFWHM_DatName, [popt[3], math.sqrt(pcov[3,3])], delimiter=',')




############################## main part ###############################################
########################################################################################


def main_loop(dirre):
  dirre = dirre + '/'
  os.chdir(dirre)

  directory2 = dirre + '/initial mar2300/'
  makedirs(dirre)

  dataTreat(dirre,calibFile)
  directoryParams1=dirre + '/cross sections data/'

  MaxQ1 = dataMaximaZ(directoryParams1)
  directoryParams2=dirre + '/100 cake data/'
  MaxChi1 = dataMaximaChi(directoryParams2)

  [ImageMax, ImageMaxLog, SampleLength, FilmThickness, IncidenceAngle, Monitor] = paramRead(dirre)
  params = [ImageMax, ImageMaxLog, SampleLength, FilmThickness, IncidenceAngle, Monitor]
  ImageMax = ImageMax*0.5   #tuning the maximum value of the image colour scale without going into the source code of the functions above 

  try:
    waveleng = dataTreat2(dirre,calibFile,MaxChi1,MaxQ1,ImageMax,ImageMaxLog)
  except:
    waveleng = 0.9744
  params.append(waveleng)
  

 # data_rays(dirre,calibFile,'extraction', params)                #processing the data file as a set of 1 degree chi rays, extraction of the data into a ray: time-consuming operation!
 # data_rays(dirre,calibFile,'processing', params)                #processing the data file as a set of 1 degree chi rays, processing (background removal, fitting) of a ray: time-consuming operation!

  try:
    peaksNo, peaksPos = processCS(dirre,'Z', 1, [1, 2], params)
  except:
    pass
#  try:
#    peaksNo, peaksPos = processCS(dirre,'integrated', 1, [1, 2], params)
#  except:
#    pass
  try:
    peaksNo, peaksPos = processCS(dirre,'xy', 1, [1,2], params)
  except:
    pass




#=================================

#Calibration file: specific for every GIXD trip
calibFile = 'Z:/media/sv2/data/Stanford2/beamline 11-3/calibration/VII-5_398mm_60_013-z_01_01_12050851.mar2300.calib'

#Directory where the directories of the samples to be analysed are located; use no more than 5 samples to be analysed at a time, 
#otherwise the garbage is collected in the WxDiff inner modules and the program sends an error notification.
directory0 = 'Z:/media/sv2/data/Stanford2/beamline 11-3/Serie_I/I-1a/'



directories_to_process = []
os.chdir(directory0)
dir0_content = glob.glob(directory0 + '/*')
for dirr0_subdir in dir0_content:
  if os.path.isdir(dirr0_subdir):
    directories_to_process.append(dirr0_subdir)

print directories_to_process


for sample_dir in directories_to_process:
  main_loop(sample_dir)

print 'batch done'
