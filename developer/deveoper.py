import re
from pathlib import Path
import dill as pickle
import numpy as np
import pandas as pd


def generate_pkl(dir_ls: list[str], patch_name: str):
    """将PEER上下载的地震动进行解析并写入pkl
    records字典的数据结构：
    records = {
        1: {  # 第一个PEER文件夹
            'RSN': xxx,
            'Tp': xxx,
            ...: xxx,  # 地震动信息
            'data':  # 时程数据
                {
                'H1':  # H1分量
                    {
                    RSN1: [(A_H1, dt_A_H1, th), (V_H1, dt_V_H1, th), (D_H1, dt_D_H1, th)] | None
                    RSN2: ...
                    },
                'H2':  # H2分量
                    {
                    RSN1: [(A_H1, dt_A_H1, th), (V_H1, dt_V_H1, th), (D_H1, dt_D_H1, th)] | None
                    RSN2: ...
                    },
                'V': {...}  # V分量
                }
           }
        2: ...  # 第二个PEER文件夹
        ...: ...
    }

    Args:
        dir_ls (list[str]): 包含PEER上下载的地震动压缩包解压后的文件夹的列表
        patch_name (str): 补丁文件名(包括后缀)
    """
    records = {}
    for n, dir_ in enumerate(dir_ls, 1):
        records[n] = {}
        dir_ = Path(dir_)
        df = pd.read_csv(dir_/'_SearchResults.csv', header=None, delimiter='\t')
        data1 = df.values[29:100]
        data2 = []
        RSN = []
        Tp = []
        D5_75 = []
        D5_95 = []
        arias = []
        earthquake_name = []
        year = []
        station = []
        magnitude = []
        mechanism = []
        Rjb = []
        Rrub = []
        v30 = []
        H1_file = []
        H2_file = []
        V_file = []
        # 解析_searchResults.csv
        for i, line in enumerate(data1):
            if line[0].startswith(' These records'):
                break
            data2.append(data1[i][0].split(','))
            for j in range(len(data2[i])):
                data2[i][j] = data2[i][j].strip()
        # 提取地震动信息
        for i, line in enumerate(data2):
            RSN.append(line[2])
            Tp.append(line[5])
            D5_75.append(line[6])
            D5_95.append(line[7])
            arias.append(line[8])
            earthquake_name.append(line[9])
            year.append(line[10])
            station.append(line[11])
            magnitude.append(line[12])
            mechanism.append(line[13])
            Rjb.append(line[14])
            Rrub.append(line[15])
            v30.append(line[16])
            H1_file.append(line[19])
            H2_file.append(line[20])
            V_file.append(line[21])
        records[n]['RSN'] = RSN
        records[n]['Tp'] = Tp
        records[n]['D5_75'] = D5_75
        records[n]['D5_95'] = D5_95
        records[n]['arias'] = arias
        records[n]['earthquake_name'] = earthquake_name
        records[n]['year'] = year
        records[n]['station'] = station
        records[n]['magnitude'] = magnitude
        records[n]['mechanism'] = mechanism
        records[n]['Rjb'] = Rjb
        records[n]['Rrub'] = Rrub
        records[n]['v30'] = v30
        records[n]['H1_file'] = H1_file
        records[n]['H2_file'] = H2_file
        records[n]['V_file'] = V_file
        records[n]['data'] = {'H1': {}, 'H2': {}, 'V': {}}
        # 解析地震动时程
        for i, RSN_ in enumerate(RSN):
            file_A_H1: Path = dir_ / H1_file[i]
            file_A_H2: Path = dir_ / H2_file[i]
            file_A_V: Path = dir_ / V_file[i]
            file_V_H1 = file_A_H1.parent / f'{file_A_H1.stem}.VT2'
            file_V_H2 = file_A_H2.parent / f'{file_A_H2.stem}.VT2'
            file_V_V = file_A_V.parent / f'{file_A_V.stem}.VT2'
            file_D_H1 = file_A_H1.parent / f'{file_A_H1.stem}.DT2'
            file_D_H2 = file_A_H2.parent / f'{file_A_H2.stem}.DT2'
            file_D_V = file_A_V.parent / f'{file_A_V.stem}.DT2'
            if not file_A_H1.exists():
                print(f'{file_A_H1.name}不存在')
                records[n]['data']['H1'][RSN_] = None
            else:
                A_H1, dt_A_H1, th_A_H1 = parse_file(file_A_H1)
                V_H1, dt_V_H1, th_V_H1 = parse_file(file_V_H1)
                D_H1, dt_D_H1, th_D_H1 = parse_file(file_D_H1)
                records[n]['data']['H1'][RSN_] = [(A_H1, dt_A_H1, th_A_H1), (V_H1, dt_V_H1, th_V_H1), (D_H1, dt_D_H1, th_D_H1)]
            if not file_A_H2.exists():
                print(f'{file_A_H1.name}不存在')
                records[n]['data']['H2'][RSN_] = None
            else:
                A_H2, dt_A_H2, th_A_H2 = parse_file(file_A_H2)
                V_H2, dt_V_H2, th_V_H2 = parse_file(file_V_H2)
                D_H2, dt_D_H2, th_D_H2 = parse_file(file_D_H2)
                records[n]['data']['H2'][RSN_] = [(A_H2, dt_A_H2, th_A_H2), (V_H2, dt_V_H2, th_V_H2), (D_H2, dt_D_H2, th_D_H2)]
            if not file_A_V.exists():
                print(f'{file_A_H1.name}不存在')
                records[n]['data']['V'][RSN_] = None
            else:
                A_V, dt_A_V, th_A_V = parse_file(file_A_V)
                V_V, dt_V_V, th_V_V = parse_file(file_V_V)
                D_V, dt_D_V, th_D_V = parse_file(file_D_V)
                records[n]['data']['V'][RSN_] = [(A_V, dt_A_V, th_A_V), (V_V, dt_V_V, th_V_V), (D_V, dt_D_V, th_D_V)]
    with open(patch_name, 'wb') as f:
        pickle.dump(records, f)


def parse_file(file: Path) -> tuple[int, float, np.ndarray]:
    """解析原初.AT2、.VT2、.DT2文件"""
    with open(file, 'r') as f:
        text = f.read()
        lines = text.split('\n')
    p1 = re.compile(r'NPTS=([ 0-9.]+)')
    p2 = re.compile(r'DT=([ 0-9.]+)SEC')
    res1 = re.findall(p1, text)
    res2 = re.findall(p2, text)
    file_name = file.absolute().as_posix()
    if not res1:
        raise ValueError(f'NPTS未匹配成功 ({file_name})')
    if not res2:
        raise ValueError(f'DT未匹配成功 ({file_name})')
    NPTS = int(res1[0])
    dt = float(res2[0])
    th = []
    for line in lines[4:]:
        th_line = line.split()
        th_line = [float(i) for i in th_line]
        th += th_line
    th = np.array(th)
    return NPTS, dt, th


if __name__ == "__main__":
    
    dir_ls = [
        'temp/records1',
        'temp/records2',
        'temp/records3',
    ]  # PEER上下载的地震动压缩包解压后的文件夹

    generate_pkl(dir_ls, patch_name='developer/update_21_22.patch')


"""
把PEER上下载的地震动压缩包解压，解压后文件夹放到temp中，
dir_ls中写入文件夹路径
"""