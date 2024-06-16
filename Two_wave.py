import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import openpyxl 
from scipy.integrate import simps
import adi
import pandas as pd
from tqdm import tqdm

font_path = 'C:\\Windows\\Fonts\\simsun.ttc'  # 宋體
font_prop = FontProperties(fname=font_path)

mode = 'show'   # 'show' or 'save'
patient = 'patient' # 'normal' or 'patient'


def butter(DataL, DataR, cut_low, cut_high, sample_rate, order):
    nyqs = sample_rate * 0.5
    H_cut = cut_high / nyqs
    L_cut = cut_low / nyqs
    sos = signal.butter(order, [L_cut, H_cut], analog=False, btype='bandpass', output='sos')
    Filter_Left = signal.sosfiltfilt(sos, DataL)
    Filter_Right = signal.sosfiltfilt(sos, DataR)
    return Filter_Left, Filter_Right

def bassel(DataL, DataR, cut_low, cut_high, sample_rate, order):
    #貝塞爾
    nyqs = sample_rate * 0.5
    H_cut = cut_high / nyqs
    L_cut = cut_low / nyqs
    sos=signal.bessel(order, [L_cut, H_cut] ,  btype='bandpass',  analog=False,  output='sos')
    Filter_Left = signal.sosfiltfilt(sos,  DataL) 
    Filter_Right = signal.sosfiltfilt(sos,  DataR) 

    return Filter_Left, Filter_Right

def find_peak(Filter_Data):
    valley_x, valley_y = find_peaks(Filter_Data * -1, height=0, distance=500)
    cardiac_cycle = np.diff(valley_x)
    peaks_x, peaks_y = find_peaks(Filter_Data, height=0, distance=500)
    peaks_y = peaks_y['peak_heights']
    valley_y = valley_y['peak_heights']
    return peaks_x, peaks_y, valley_x, valley_y, cardiac_cycle

def derivative(Data, Level, values=[]):
    result = np.gradient(Data)

    if Level == 0:
        values.append(Data)
        return 0
    else:
        values.append(Data)
        return derivative(result, Level - 1, values)

def process_wave(cycle):
    values = []
    derivative(cycle, 3, values)
    values = np.array(values)
    origin, derivative_1, derivative_2, derivative_3 = values
    derivative_1 = derivative_1 * 100
    derivative_2 = derivative_2 * 5000
    derivative_3 = derivative_3 * 100000

    return origin, derivative_2, derivative_3

def Find_Path(path):

    File_path = []

    #find all Data_file path 
    for root,  subfolders,  filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'):
                continue
            filepath = root +'/'+ filename
            new_filepath = filepath.replace("\\", "/")
            File_path.append(new_filepath)

    return File_path

def get_Imformation(path,locate, imformation=[]):
    test=path.split('/')
    Name = test[locate[0]]
    Date = test[locate[1]]
    State_check = test[locate[2]]
    if State_check =='易堵':
        State = '0'
    else: 
        State = '1'

    file_name = test[len(test)-1]
    name_check = file_name.find('R')
    if name_check != -1:
        Status = 'Right'
    else:
        Status = 'Left'
    
    imformation =[Name, Date, State, Status]

    return imformation, Name

def plot(L_cycle,R_cycle):
    plt.plot(L_cycle)
    plt.plot(R_cycle)
    plt.show()

def main():
    channel1_id = 2
    channel2_id = 4
    record_id = 1

    if patient == 'normal':
        File_path = Find_Path("F:\\正常人Data") #!正常人
    else:
        File_path = Find_Path("F:\\第二次收案") #!病患
    print('找到資料筆數', len(File_path))

    Features = []
    df_c = pd.DataFrame()
    for j, path in tqdm(enumerate(File_path), total=len(File_path), desc='Processing'):
        Data = adi.read_file(path)

        Right = Data.channels[channel1_id - 1].get_data(record_id)
        Left = Data.channels[channel2_id - 1].get_data(record_id)

        Filter_Left,Filter_Right = bassel(Left, Right, 0.7, 9, 1000, 4)

        L_wave = Filter_Left[100000:120000] * 10
        R_wave = Filter_Right[100000:120000] * 10

        L_valley_x, L_valley_y = find_peaks(L_wave * -1, height=0, distance=150)
        R_valley_x, R_valley_y = find_peaks(R_wave * -1, height=0, distance=150)

        L_valley_y = L_valley_y['peak_heights']
        R_valley_y = R_valley_y['peak_heights']

        if patient == 'normal':
            Imformation,Name = get_Imformation(path,locate=[2,3,4]) #!正常人
        else:
            Imformation,Name = get_Imformation(path,locate=[3,4,2]) #!病患

        if len(L_valley_x) > len(R_valley_x): #找最小的cycle
            min_cycle = len(R_valley_x)
        else:
            min_cycle = len(L_valley_x)

        for i in range(0,min_cycle - 2,2):
            diff = np.abs(L_valley_x[i] - R_valley_x[i]) #time diff
            L_cycle = L_wave[L_valley_x[i]:L_valley_x[i + 2]] #two cycle
            L_cycle_cut = [L_wave[L_valley_x[i]:L_valley_x[i + 1]], L_wave[L_valley_x[i + 1]:L_valley_x[i + 2]]] #divide 2

            L_peaks_x, L_peaks_y = find_peaks(L_cycle, height=0, distance=500)
            L_peaks_y = L_peaks_y['peak_heights']

            R_cycle = R_wave[L_valley_x[i]:L_valley_x[i + 2]]
            R_cycle_cut = [R_wave[R_valley_x[i]:R_valley_x[i + 1]], R_wave[R_valley_x[i + 1]:R_valley_x[i + 2]]]  # vivide 2

            R_peaks_x, R_peaks_y = find_peaks(R_cycle, height=0, distance=500)
            R_peaks_y = R_peaks_y['peak_heights']

            if len(L_cycle) < 1100 or len(L_peaks_y) != 2 or len(R_peaks_y) != 2 or len(R_cycle) < 1100:
                continue

            if L_peaks_y[0] < 0.5:
                L_cycle *= 0.5 / L_peaks_y[0]
                L_peaks_y[0] = 0.5
                L_peaks_y[1] = 0.5

            if R_peaks_y[0] < 0.5:
                R_cycle *= 0.5 / R_peaks_y[0]
                R_peaks_y[0] = 0.5
                R_peaks_y[1] = 0.5

            plot(L_cycle, R_cycle)
                


if __name__ == '__main__':
    main()
