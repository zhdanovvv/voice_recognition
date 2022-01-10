#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 16:10:49 2021

@author: zhdanov
"""


import os
from pathlib import Path
from scipy.io import wavfile
import scipy.io
from os.path import dirname, join as pjoin
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, dct, rfft, rfftfreq
from python_speech_features import mfcc
from sklearn import preprocessing
from hmmlearn import hmm
from python_speech_features import mfcc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def search_mfcc_coeff(file_name):               # Функция поиска мел кепстральных коэффициентов
    samplerate, data= wavfile.read(file_name)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
# =============================================================================
#     plt.plot(time, (data[:]), label="Plot")
#     plt.xlabel("Time [s]")
#     plt.ylabel("Amplitude")
#     plt.show()
# =============================================================================
    
    f, sp_density = signal.periodogram(data, samplerate) # Спектральная плотность мощности
# =============================================================================
#     plt.semilogy(f, sp_density)
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Linear spectrum [V RMS]')
#     plt.show()
# =============================================================================
    
    b, a = signal.butter(17, 0.075, 'lowpass')          # Конфигурационный фильтр 8 указывает порядок фильтра
    filtedData = abs(signal.filtfilt(b, a, sp_density)) # Фильтрация спектральной плотности
# =============================================================================
#     plt.xlabel('Frequency of filted data [Hz]')
#     plt.ylabel('Amplitude')
#     plt.plot(f, filtedData)
#     plt.show()
# =============================================================================
    _filtedData = np.sqrt(np.log(filtedData))           # квадрат логарифма отфильтрованных данных
# =============================================================================
#     plt.plot(f, _filtedData)
#     plt.show()
# =============================================================================
    
    mfcc_coeff = mfcc(_filtedData[:1000], samplerate)   # расчет мел кепстральных коэффициентов
    return mfcc_coeff

def KNeighbors(your_path):                              # Функция поиска мел кепстральных коэффициентов
                                                        # для обучающего набора данных и классификация алгоритмом
                                                        # К ближайших соседей  
                                                        
    featureMatrix_fem = np.zeros((0, 13))               # Матрица мел кепстральных коэффициентов для женских голосов
    featureMatrix_mel = np.zeros((0, 13))               # Матрица мел кепстральных коэффициентов для мужских голосов
    
    # Подстрока имени файла, с женским голосом диктора
    fem_subStr_one = 'A30000'
    fem_subStr_two = 'A30002'
    # Подстрока имени файла, с мужским голосом диктора
    mel_subStr_one = 'A30001'
    mel_subStr_two = 'A30003'
            
    with os.scandir(your_path) as listOfEntries:  
        for entry in listOfEntries:            
            if entry.is_file(): 
                # проверка вхождения подстроки в строку и печать всех записей, являющихся файлами 
                if (fem_subStr_one or fem_subStr_two) in entry.name: 
                    #print(entry.name,  'Female voice', '\n')
                    # Расчет мел кепстральных коэффициентов для женских голосов                   
                    mfc_coef = search_mfcc_coeff(pjoin(your_path, entry.name))
                    # функция обнуления nan
                    for x in range(len(mfc_coef)):
                        for y in range(len(mfc_coef[x])):
                            v = mfc_coef[x][y]
                            if v!=v:
                                mfc_coef[x][y] = 0.0                    
                    featureMatrix_fem = np.append(featureMatrix_fem, mfc_coef, axis=0)
                # проверка вхождения подстроки в строку и печать всех записей, являющихся файлами 
                elif (mel_subStr_one or mel_subStr_two) in entry.name:
                    #print(entry.name, ' Male voice', '\n')
                    # Расчет мел кепстральных коэффициентов для мужских голосов                    
                    mfc_coef = search_mfcc_coeff(pjoin(your_path, entry.name))
                    # функция обнуления nan
                    for x in range(len(mfc_coef)):
                        for y in range(len(mfc_coef[x])):
                            v = mfc_coef[x][y]
                            if v!=v:
                                mfc_coef[x][y] = 0.0                          
                    featureMatrix_mel = np.append(featureMatrix_mel, mfc_coef, axis=0)
                                           
    pca = PCA(n_components=13) # Уменьшение размерности полученных, после расчета мел кепстральных коэффициентов, данных для женских голосов
    pc_fem = pca.fit(featureMatrix_fem).transform(featureMatrix_fem) 
    class_fem = np.zeros((561, 1)) # Метка класса для женского голоса
    pc_fem = np.append(pc_fem, class_fem, axis=1)
    
    pca = PCA(n_components=13)  # Уменьшение размерности полученных, после расчета мел кепстральных коэффициентов, данных для мужских голосов
    pc_mel = pca.fit(featureMatrix_mel).transform(featureMatrix_mel)
    class_mel = np.ones((561, 1))  # Метка класса для мужского голоса 
    pc_mel = np.append(pc_mel, class_mel, axis=1)
         
    Total_matrix = np.concatenate((pc_fem, pc_mel)) # Итоговая матрица признаков
    neigh = KNeighborsClassifier(n_neighbors=3, p=2) # Классификатор
           
    plt.plot(pc_mel[:, 0], pc_mel[:, 1],'.')    
    plt.plot(pc_fem[:, 0], pc_fem[:, 1],'.')
    plt.legend(['Blue - male voice', 'Orange - female voice'])
    plt.title('The first two principal components')
    plt.show()    
    
    print('------------------', '\n', 'Train data:', '\n')    
    print('Class: 0', '- Female voice', '\n')
    print('Class: 1', '- Male voice', '\n', '------------------')
    
    return neigh.fit(Total_matrix[:, :12], Total_matrix[:, 13]) # Классификатор

def print_Res(classPredict):
    print('---------------', 'Test data', '\n')
    if classPredict == 0:
        print('Class:', classPredict, '- Female voice', '\n', '-----------------')
    elif classPredict == 1:
        print('Class: ', classPredict, '- Male voice', '\n', '------------------')
        
def mfcc_test_Files(your_path_):
        
    featureMatrix = np.zeros((0, 13))    
    with os.scandir(your_path_) as listOfEntries:  
        for entry in listOfEntries:            
            if entry.is_file(): 
                # проверка вхождения подстроки в строку и печать всех записей, являющихся файлами                  
                #print(entry.name,  'Female voice', '\n')
                # Расчет мел кепстральных коэффициентов для женских голосов                    
                mfc_coef = search_mfcc_coeff(pjoin(your_path_, entry.name))
                featureMatrix = np.append(featureMatrix, mfc_coef, axis=0)
                                
    return featureMatrix

def main():    
    train_path = os.path.normpath(pjoin(os.getcwd(),"wav_data/train_data"))
    test_path = os.path.normpath(pjoin(os.getcwd(),"wav_data/test_data"))    
    class_voice = KNeighbors(train_path)   # Классификация для обучающего набора данных 
    mfc_test_coef = mfcc_test_Files(test_path)      # Проверка классификатора на тестовых файлах
    result = np.zeros((2, 0))
    for i in range(0, 2, 1):               # Вывод результатов в консоль
        result = np.append(result, class_voice.predict([mfc_test_coef[i, :12]]))
        print_Res(result[i])

if __name__ == '__main__':
    main()
