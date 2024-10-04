import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json

# 初始化特徵點和檔案名稱的列表
feature_points = []
current_file_name = None
file_index = 0  # 用於追蹤當前處理的文件索引

progress_file = 'PPG\\output\\progress.json'  # 進度保存文件

# 滑鼠點擊事件處理函數
def onclick(event):
    global feature_points
    if event.inaxes:  # 確保點擊在圖形範圍內
        if len(feature_points) < 12:
            # 取得點擊的 X 值，並找到對應的 Y 值 (波形數據)
            x_val = int(event.xdata)  # X 值是點擊的位置，取整數
            y_val = waveform_data[x_val]  # Y 值來自於波形數據
            feature_points.append(x_val)
            # 在圖上標記點 (X,Y)
            plt.scatter(x_val, y_val, color='red', zorder=5)
            plt.draw()
            print(f'Feature {len(feature_points)} from {current_file_name}: (X={x_val}, Y={y_val})')
        if len(feature_points) == 12:
            plt.close()  # 如果標記了6個點，自動關閉圖表並跳到下一個檔案

# 清除按鍵事件處理函數
def onkey(event):
    global feature_points
    if event.key == 'c':  # 按下 'C' 鍵清除點擊
        feature_points = []  # 清空點擊的特徵點
        plt.cla()  # 清空圖表
        plot_waveform(waveform_data)  # 重新繪製波型
        plt.draw()
        print("已清除所有標記的特徵點")

# 保存特徵點到CSV的函數
def save_points_to_csv(file_name):
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:  # 使用 'a' 來追加數據
        writer = csv.writer(file)
        if not file_exists:
            # 如果文件不存在，寫入表頭
            writer.writerow(['Name', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'])  
        # 確保保存每個文件的名字和6個 X 值
        row = [current_file_name] + feature_points
        writer.writerow(row)
    print(f"特徵點已保存到 {file_name}")

# 保存進度到 JSON 文件
def save_progress():
    with open(progress_file, 'w') as file:
        json.dump({
            'file_index': file_index,
            'feature_points': feature_points
        }, file)
    print(f"進度已保存到 {progress_file}")

# 恢復進度
def load_progress():
    global file_index, feature_points
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as file:
            progress_data = json.load(file)
            file_index = progress_data.get('file_index', 0)
            feature_points = progress_data.get('feature_points', [])
        print(f"進度已從 {progress_file} 加載，從檔案 {file_index} 開始")

# 繪製波型的函數
def plot_waveform(waveform_data):
    plt.plot(waveform_data, label='Waveform Data')
    plt.title(f'Waveform - {current_file_name} - Click to Mark Features')
    plt.xlabel('Data Point Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

# 讀取資料夾內所有 .npy 檔案
def load_all_npy_files(directory):
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    return npy_files

# 主程序邏輯
def main():
    global waveform_data, current_file_name, file_index, feature_points

    # 設定你的資料夾路徑
    folder_path = 'PPG\\output\\bad signal'  # 替換成你的資料夾路徑
    npy_files = load_all_npy_files(folder_path)

    if not npy_files:
        print("資料夾內沒有發現任何 .npy 檔案")
        return

    # 恢復進度（如果存在）
    load_progress()

    # 顯示所有波形並等待點擊特徵
    while file_index < len(npy_files):
        current_file_name = npy_files[file_index]
        file_path = os.path.join(folder_path, current_file_name)
        waveform_data = np.load(file_path)

        # 創建圖形並連接點擊事件和按鍵事件
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_waveform(waveform_data)

        # 連接滑鼠點擊事件
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # 連接鍵盤事件
        kid = fig.canvas.mpl_connect('key_press_event', onkey)

        # 顯示圖形
        plt.show()

        # 保存特徵點
        csv_file_name = 'PPG\\output\\feature_points.csv'  # 保存CSV文件名
        save_points_to_csv(csv_file_name)

        # 下一個文件
        file_index += 1
        feature_points = []  # 清空特徵點列表以便標記下一張圖片

        # 保存當前進度
        save_progress()

if __name__ == '__main__':
    main()
