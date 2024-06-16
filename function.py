def on_key(event):
    if event.key == 'z':
        plt.close()

def plot_wave(peaks_x, origin, derivative_2, L_zero_TDPPG, L_zero_origin, Name, i):
    x = np.linspace(0, len(origin), len(origin))
    plt.plot(x, origin, label='PPG')
    plt.plot(x, derivative_2, label='SDPPG')

    plt.plot(x[peaks_x], origin[peaks_x], 'o', label='systolic peak')
    plt.plot(x[L_zero_origin], origin[L_zero_origin], 'o', label='width')
    # plt.plot(x[e_point], origin[e_point], '*', label='diastolic notch',color = 'r')
    # plt.plot(x[f_point], origin[f_point], '*', label='diastolic peak',color = 'r')
    plt.plot(x[L_zero_TDPPG], derivative_2[L_zero_TDPPG], '*', label='diastolic peak',color = 'r')
    #plt.plot(x, derivative_3, label='TDPPG')
    plt.xlim(5, len(origin)-5)
    plt.ylim(-0.6, 0.6)
    plt.grid()
    plt.legend()
    plt.title(f'{Name}, {i + 1}th wave',fontproperties=font_prop)

    if mode == 'show':
        plt.show()
    else:
        plt.savefig(f'F:\\TDPPG\\Two Waves with Derivative\\{Name}, {i + 1}th wave.jpg')

    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.close()
def Caculate_Area(origin,e_point):
    A1 = origin[0:e_point]
    A2 = origin[e_point:len(origin)]
    x1 = np.linspace(0, len(A1)-1, len(A1))
    x2 = np.linspace(0, len(A2)-1, len(A2))
    area_1 = simps(A1, x1)
    area_2 = simps(A2, x2)
    area_1 = np.abs(area_1)
    area_2 = np.abs(area_2)
    return area_2 / area_1

def get_Features(origin,derivative_3,peaks_x,peaks_y):
    
    zero_TDPPG = np.where(np.diff(np.sign(derivative_3)))[0]
    differences = np.abs(zero_TDPPG - 400)
    index_of_closest_point = np.argmin(differences)

    zero_origin = np.where(np.diff(np.sign(origin)))[0]
    width = np.diff(zero_origin)

    e_point = zero_TDPPG[int(index_of_closest_point)-1]
    f_point = zero_TDPPG[int(index_of_closest_point)]
    ct = peaks_x
    deltaT = e_point - peaks_x
    cardiac_cycle = len(origin)
    A21 =Caculate_Area(origin,int(e_point))

    Features = [e_point, f_point, deltaT, ct, cardiac_cycle,A21]
    #print(Features)
    
    Features = [float(item) for item in Features]   #!缺width
    
    return Features, zero_origin, zero_TDPPG[0:6]
def Write_Excel(All_imformation):
    workbook = openpyxl.load_workbook("F:\\TDPPG\\output.xlsx")
    sheet1 = workbook.worksheets[0]

    Data_Row = sheet1.max_row+1

    length = len(All_imformation)

    for i in range(1, length+1):
        sheet1.cell(Data_Row , i).value= All_imformation[i-1]
    workbook.save("F:\\TDPPG\\output.xlsx")
def plot_cycle(L_origin,R_origin, Name, i, diff):
    x = np.linspace(0, len(L_origin), len(L_origin))
    y = np.linspace(0, len(L_origin), len(L_origin))
    plt.plot(x, L_origin, label='Left PPG')
    plt.plot(y, R_origin, label='Right PPG')
    plt.xlim(5, len(L_origin)-5)
    plt.ylim(-0.6, 0.6)
    plt.grid()
    plt.legend()
    plt.title(f'{Name}, {i + 1}th Left_Right',fontproperties=font_prop)
    plt.savefig(f'F:\\TDPPG\\Two_wave\\{Name}, {i + 1}th Left_Right.jpg')
    def on_key(event):
        if event.key == 'z':
            plt.close()
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    #plt.show()
    plt.close()
def parameter_setting(L_cycle, L_peaks_x, L_peaks_y, R_cycle, R_peaks_x, R_peaks_y):
    L_origin, R_origin = [], []
    L_derivative_2, R_derivative_2 = [], []
    L_derivative_3, R_derivative_3 = [], []
    L_Features, R_Features = [], []
    L_zero_origin, R_zero_origin = [], []
    L_zero_TDPPG, R_zero_TDPPG = [], []

    for i in range(2):

        L_origin_i, L_derivative_2_i, L_derivative_3_i = process_wave(L_cycle[i]) #微分
        R_origin_i, R_derivative_2_i, R_derivative_3_i = process_wave(R_cycle[i]) #微分

        L_Features_i, L_zero_origin_i, L_zero_TDPPG_i = get_Features(L_origin_i, L_derivative_3_i, L_peaks_x[i], L_peaks_y[i])
        R_Features_i, R_zero_origin_i, R_zero_TDPPG_i = get_Features(R_origin_i, R_derivative_3_i, R_peaks_x[i], R_peaks_y[i])

        L_origin.append(L_origin_i)
        L_derivative_2.append(L_derivative_2_i)
        L_derivative_3.append(L_derivative_3_i)
        L_Features.append(L_Features_i)
        L_zero_origin.append(L_zero_origin_i)
        L_zero_TDPPG.append(L_zero_TDPPG_i.tolist())

        R_origin.append(R_origin_i)
        R_derivative_2.append(R_derivative_2_i)
        R_derivative_3.append(R_derivative_3_i)
        R_Features.append(R_Features_i)
        R_zero_origin.append(R_zero_origin_i)
        R_zero_TDPPG.append(R_zero_TDPPG_i.tolist())

    L_zero_TDPPG[1] = [x + len(L_cycle[0]) for x in L_zero_TDPPG[1]]
    L_zero_origin[1] = [x + len(L_cycle[0]) for x in L_zero_origin[1]]
    L_zero_TDPPG = np.hstack((L_zero_TDPPG[0],L_zero_TDPPG[1]))
    L_zero_origin = np.hstack((L_zero_origin[0],L_zero_origin[1]))
    L_derivative_2 = np.hstack((L_derivative_2[0],L_derivative_2[1]))
    
    return L_derivative_2, L_zero_TDPPG, L_zero_origin
#main            
#L_origin, L_derivative_2, L_derivative_3, L_Features, L_zero_origin, L_zero_TDPPG, R_origin, R_derivative_2, R_derivative_3, R_Features, R_zero_origin, R_zero_TDPPG =parameter_setting(L_cycle, L_peaks_x, L_peaks_y, R_cycle, R_peaks_x, R_peaks_y)
            L_derivative_2, L_zero_TDPPG, L_zero_origin = parameter_setting(L_cycle, L_peaks_x, L_peaks_y, R_cycle, R_peaks_x, R_peaks_y)

            plot_cycle(L_2cycle,R_2cycle, Name, i, diff)
            plot_wave(L_peaks_x, L_2cycle, L_derivative_2, L_zero_TDPPG, L_zero_origin, Name, i)