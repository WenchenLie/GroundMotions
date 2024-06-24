"""
基于NGAWest2地震动数据库的选波程序  
开发者：Vincent  
日期：2024年2月4日
更新：2024.03.13 增加输出缩放后反应谱
更新：2024.03.31 增加输出反应谱对比图、各个匹配规则的误差值、选波参数的记录文档
更新：2024.04.11 优化了梯度下降法的初值计算方法
更新：2024.05.07 更新Info.hdh5文件格式（2.0）
更新: 2024.06.23 选波结果写入pickle与origin
"""
import os
import sys
import shutil
from pathlib import Path
from PIL import Image
from tkinter import messagebox

import h5py
import originpro
import originpro as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SeismicUtils.Records import Records


class Selecting:
    version = '2.0'
    RSN_expected = set([i for i in range(1, 21541)])  # 官网宣称有的RSN（但实际不全）

    def __init__(self, output_dir: Path | str):
        """基于NGA West2地震动数据库的选波程序

        Args:
            output_dir (str | Path): 输出文件夹路径
        """
        self.T_spec = np.arange(0, 10.01, 0.01)
        self.file_accec = None
        self.file_vel = None
        self.file_disp = None
        self.file_spec = None
        self.file_info = None
        self.T_targ = None
        self.Sa_targ = None
        self.approach = None
        self.para_scaling = None
        self.rules = None
        self.para_match = None
        self.range_scale_factor = None
        self.range_PGA = None
        self.range_magnitude = None
        self.range_Rjb = None
        self.range_Rrup = None
        self.range_vs30 = None
        self.range_D5_95 = None
        self.range_strike_slip = 'all'
        self.range_pulse = 'all'
        self.range_N_events = None
        self.range_RSN = None
        self.range_component = ['H1', 'H2', 'V']
        self.norm_weight = None
        self.selecting_text = ''
        self.records = Records()  # 导出的波库
        # 打开文件
        output_dir = Path(output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir)
        else:
            res = messagebox.askokcancel('警告', f'"{output_dir.absolute()}"已存在，是否删除？', icon='warning')
            if res:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                print('已删除')
            else:
                res1 = messagebox.askokcancel('警告', '是否覆盖？', icon='warning')
                if res1:
                    print('将覆盖数据')
                else:
                    print('退出选波')
                    return
        self.output_dir = output_dir


    def import_files(self,
            file_acc: str | Path,
            file_vel: str | Path,
            file_disp: str | Path,
            file_spec: str | Path,
            file_info: str | Path,
        ):
        """导入hdf5文件"""
        files = [file_acc, file_vel, file_disp, file_spec, file_info]
        self.file_accec = file_acc
        self.file_vel = file_vel
        self.file_disp = file_disp
        self.file_spec = file_spec
        self.file_info = file_info
        for file in files:
            print(f'正在校验文件 - {file}')
            self._check_file(file)

    def check_database(self):
        """进行数据库统计"""
        print('正在统计地震动本地数据库')
        if not all([self.file_accec, self.file_vel, self.file_disp,
                    self.file_spec, self.file_info]):
            print('【Warining】请先导入五个hdf5文件')
            return
        # 检查RSN数量
        RSN_exists = set()
        f_info = h5py.File(self.file_info, 'r')
        for item in f_info:
            ds = f_info[item]
            RSN_exists.add(ds.attrs['RSN'])
        f_info.close()
        # 检查地震波分量总数量
        f_accec = h5py.File(self.file_accec, 'r')
        n = 0
        for item in f_accec:
            n += 1
        f_accec.close()
        # 缺失地震波
        RSN_missing = list(self.RSN_expected - RSN_exists)
        RSN_missing = sorted(RSN_missing)
        print(f'库存地震动：{len(RSN_exists)}组，共{n}条')
        print(f'缺失地震动：{len(RSN_missing)}组')
        print('缺失地震动RSN：')
        print(RSN_missing)

    def target_spectra(self, file: str | Path, plot_spectrum: bool=False, scale: float=1):
        """定义目标谱（两列数据，周期(s)-谱值(g)）

        Args:
            file (str | Path): 文件路径
            plot_spectrum (bool, optional): 是否绘制规范谱，默认False
            scale (float, optional): 将目标谱进行缩放，缩放系数默认为1
        """
        data = np.loadtxt(file)
        self.T_targ0 = data[:, 0]  # 目标谱周期
        self.Sa_targ0 = data[:, 1] * scale  # 目标谱加速度
        if max(self.T_targ0) > 10:
            raise ValueError('【Error】目标谱周期范围不应超过10s')
        self.T_targ = np.arange(self.T_targ0[0], self.T_targ0[-1], 0.01)
        linear_interp = interp1d(self.T_targ0, self.Sa_targ0, kind='linear', fill_value=0, bounds_error=False)
        self.Sa_targ = linear_interp(self.T_targ)  # 将目标谱转换为0.01步长
        if plot_spectrum:
            plt.plot(self.T_targ0, self.Sa_targ0)
            plt.show()
        
    def scaling_approach(self, approach: str, para: float | tuple=None):
        """定义地震动缩放方法

        Args:
            approach (str): 缩放方法
            * [a] 按Sa(0)(即PGA)匹配反应谱, para=None or Sa  
            * [b] 按Sa(Ta)匹配反应谱, para=Ta or (Ta, Sa)  
            * [c] 按Sa(Ta~Tb)匹配反应谱(几何平均数), 最小化RMSE, para=(Ta, Tb)  
            * [d] 按Sa,avg(Ta~Tb)匹配反应谱, para=(Ta, Tb) or (Ta, Tb, Sa)  
            * [e] 指定缩放系数, para=SF\n
            para (float | tuple): 缩放参数，与`approach`的取值有关
        """
        self.approach = approach
        self.para_scaling = para
        self._write('反应谱缩放方式：', end='')
        match approach:
            case 'a':
                self._write('按PGA缩放')
                if para:
                    self._write(f'(已指定PGA={para:.4f}g)')
            case 'b':
                if type(para) is not tuple:
                    self._write(f'按Sa({round(para, 6)})缩放')
                else:
                    self._write(f'按Sa({round(para[0], 6)})缩放')
                    self._write(f'(已指定a({round(para[0], 6)})={round(para[1], 6)}g)')
            case 'c':
                self._write(f'按{round(para[0], 6)}~{round(para[1], 6)}周期范围进行缩放(令RSME最小)')
            case 'd':
                self._write(f'按{round(para[0], 6)}~{round(para[1], 6)}周期范围内的Sa_avg进行缩放(几何平均数)')
            case 'e':
                self._write(f'指定缩放系数({para})')


    def matching_rules(self, rules: list[str], para: list[float | tuple], weight: list=None):
        """定义地震动匹配规则，可选多种，依次判断

        Args:
            rules (list[str]): 匹配规则
            * [full] 按照给定的反应谱的最大周期范围完全匹配(归一化军方法误差NRSME最小), para=[None]
            * [a] 按Sa(0)(即PGA)匹配反应谱, para=[None]  
            * [b] 按Sa(Ta)匹配反应谱, para=[Ta]  
            * [c] 按Sa(Ta~Tb)匹配反应谱(几何平均数), 最小化RMSE, para=[(Ta, Tb)]  
            * [d] 按Sa,avg(Ta~Tb)匹配反应谱, para=[(Ta, Tb)]\n
            para (list[float | tuple]): 缩放参数，与`approach`的取值有关  
            weight (list[tuple]): 各匹配规则的权重系数

        Example:
            >>> matching_rules(rules=['full', 'b', 'c'], para=[None, 1.5, (1, 2)], weight=[1, 1.2, 0.8])
            表示在匹配反应谱（计算拟合误差）时综合考虑全周期范围、Sa(1.5)处谱值和Sa(1~2)范围的匹配程度，
            对应的权重依次为1、1.2、和0.8

        Note:
            `scaling_approach`方法中选用的`approach`参数不得作为该方法的`rules`参数
            （这表示之前定义的缩放方法不得用于判断反应谱匹配程度）
        """
        if (self.approach in rules) and (self.approach not in ['c', 'd']):
            raise ValueError(f'【Error】地震动缩放方法("{self.approach}")不得再次作为匹配规则！')
        self.rules = rules
        self.para_match = para
        if not weight:
            weight = [1] * len(rules)
        self.norm_weight = [i / sum(weight) for i in weight]
        if len(rules) != len(para):
            raise ValueError('【Error】参数rules和para的长度应一致')
        self._write('反应谱的匹配规则：')
        for i, rule in enumerate(rules):
            match rule:
                case 'full':
                    self._write(f'({i+1}) 按目标谱全周期范围的RSME值进行匹配，权重={weight[i]}')
                case 'a':
                    self._write(f'({i+1}) 按PGA匹配，权重={weight[i]}')
                case 'b':
                    self._write(f'({i+1}) 按Sa({round(para[i], 6)})匹配，权重={weight[i]}')
                case 'c':
                    self._write(f'({i+1}) 按{round(para[i][0], 6)}~{round(para[i][1], 6)}周期范围的RSME值进行匹配，权重={weight[i]}')
                case 'd':
                    self._write(f'({i+1}) 按{round(para[i][0], 6)}~{round(para[i][1], 6)}周期范围的Sa_avg值(几何平均数)进行匹配，权重={weight[i]}')


    def constrain_range(self, scale_factor: tuple=None, PGA: tuple=None, magnitude: tuple=None, Rjb: tuple=None, Rrup: tuple=None,
                        vs30: tuple=None, D5_95: tuple=None, duration: tuple=None, strike_slip: str='all', pulse: str | bool='all',
                        N_events: int=None, RSN: tuple=None, component: list=['H1', 'H2', 'V']):
        """定义约束范围

        Args:
            scale_factor (tuple, optional): 缩放系数，默认None
            PGA (tuple, optional): PGA，默认None
            magnitude (tuple, optional): 震级，默认None
            Rjb (tuple, optional): 默认None
            Rrup (tuple, optional): 默认None
            vs30 (tuple, optional): 剪切波速，默认None
            D5_95 (tuple, optional): 有效持时，默认None
            duration (tuple, optional): 持时，默认None
            strike_slip (str, optional): 振源机制，默认'all'
            * [all] all types
            * [a] strike slip
            * [b] normal/oblique
            * [c] reverse/oblique
            * [d] strike slip + normal/oblique
            * [e] strike slip + reverse/oblique
            * [f] normal/oblique + reverse/oblique\n
            pulse (str | bool, optional): 脉冲型地震，默认'all'
            * [all] 不限定范围
            * [True] 仅脉冲型
            * [False] 仅非脉冲型\n
            N_events (int, optional): 相同地震事件所允许的最大出现次数，默认None   
            RSN (tuple, optional): RSN，默认None
            component (list, optional): 地震动分量，默认['H1', 'H2', 'V']，可根据需要删减列表元素

        """
        self.range_scale_factor = scale_factor
        self.range_PGA = PGA
        self.range_magnitude = magnitude
        self.range_Rjb = Rjb
        self.range_Rrup = Rrup
        self.range_vs30 = vs30
        self.range_D5_95 = D5_95
        self.duration = duration
        self.range_strike_slip = strike_slip
        self.range_pulse = pulse
        self.range_N_events = N_events
        if RSN:
            self.range_RSN = (int(RSN[0]), int(RSN[1]))
        self.range_component = component

    def run(self, number: int) -> tuple[list[str], dict]:
        """选波计算

        Args:
            number (int): 需要的地震动数量

        Returns:
            list[str]: 包括所有选波结果地震动名（无后缀）的列表
            dict: {地震动名: (缩放系数, 匹配误差)}
        """
        records = self.records
        print('正在进行初步筛选...')
        files_within_range = []  # 约束范围内（除了PGA,scale_factor,N_events）的备选波
        f_spec = h5py.File(self.file_spec, 'r')
        f_info = h5py.File(self.file_info, 'r')
        # 1 初步筛选
        for i, item in enumerate(f_info):
            # if i == 1000:
            #     print(f' --------------- 调试模式，只考虑数据库中前{i}条地震波 --------------- ')
            #     break  # NOTE: for test
            if item == 'VERSION':
                continue
            print(f'  {int(i/len(f_info)*100)}%   \r', end='')
            ds = f_info[item]
            H1_file = ds.attrs['H1_file']
            H2_file = ds.attrs['H2_file']
            V_file = ds.attrs['V_file']
            RSN = int(ds.attrs['RSN'])
            Rjb = float(ds.attrs['Rjb'])
            Rrup = float(ds.attrs['Rrup'])
            Tp = ds.attrs['Tp']
            Tp = float(Tp) if type(Tp)==np.float64 else str(Tp)
            arias = ds.attrs['arias']
            arias = float(arias) if type(arias)!=str else ''
            duration_5_75 = ds.attrs['duration_5_75']
            duration_5_75 = float(duration_5_75) if type(duration_5_75)!=str else ''
            duration_5_95 = ds.attrs['duration_5_95']
            duration_5_95 = float(duration_5_95) if type(duration_5_95)!=str else ''
            duration = float(ds.attrs['duration'])
            earthquake_name = ds.attrs['earthquake_name']
            magnitude = float(ds.attrs['magnitude'])
            mechanism = ds.attrs['mechanism']
            station = ds.attrs['station']
            vs30 = float(ds.attrs['vs30'])
            year = int(ds.attrs['year'])
            if self.range_magnitude and not self.range_magnitude[0] <= magnitude <= self.range_magnitude[1]:
                continue
            if self.range_Rjb and not self.range_Rjb[0] <= Rjb <= self.range_Rjb[1]:
                continue
            if self.range_Rrup and not self.range_Rrup[0] <= Rrup <= self.range_Rrup[1]:
                continue
            if self.range_vs30 and not self.range_vs30[0] <= vs30 <= self.range_vs30[1]:
                continue
            if self.range_D5_95:
                if type(duration_5_95) is not float:
                    continue
                if not self.range_D5_95[0] <= duration_5_95 <= self.range_D5_95[1]:
                    continue
            if self.duration:
                if not self.duration[0] <= duration <= self.duration[1]:
                    continue
            if self.range_strike_slip != 'all':
                if self.range_strike_slip == 'a' and mechanism != 'strike slip':
                    continue
                elif self.range_strike_slip == 'b' and mechanism not in ['Normal Oblique', 'Normal']:
                    continue
                elif self.range_strike_slip == 'c' and mechanism != 'Reverse Oblique':
                    continue
                elif self.range_strike_slip == 'd' and mechanism not in ['strike slip', 'Normal Oblique']:
                    continue
                elif self.range_strike_slip == 'e' and mechanism not in ['strike slip', 'Reverse Oblique']:
                    continue
                elif self.range_strike_slip == 'f' and mechanism not in ['Normal Oblique', 'Reverse Oblique']:
                    continue
            if self.range_pulse != 'all':
                if self.range_pulse and type(Tp) is str:
                    continue
                elif not self.range_pulse and type(Tp) is float:
                    continue
            if self.range_RSN and not self.range_RSN[0] <= RSN <= self.range_RSN[1]:
                continue
            if 'H1' in self.range_component:
                files_within_range.append(H1_file)
            if 'H2' in self.range_component:
                files_within_range.append(H2_file)
            if 'V' in self.range_component and V_file != '-':
                files_within_range.append(V_file)
        # 2 二次筛选
        print('正在进行选波计算...')
        T = self.T_spec  # 0-10s
        file_SF = {}  # {地震动: 缩放系数}
        file_error = {}  # {地震动: 匹配误差}
        file_event = {}  # {地震动: 事件}
        file_PGA = {}  # {地震动: PGA}
        for i, file in enumerate(files_within_range):
            # print(f'i = {i}  \r', end='')
            print(f'  {int(i/len(files_within_range)*100)}%   \r', end='')
            ds = f_spec[file]
            RSN = str(ds.attrs['RSN'])
            event = f_info['RSN'+RSN].attrs['earthquake_name']
            file_event[file] = event
            PGA = ds.attrs['PGA']
            file_PGA[file] = PGA
            Sa = ds[:]  # 当前地震动反应谱谱值
            if self.approach == 'a':
                if self.para_scaling:
                    Sa0 = self.para_scaling  # 目标值
                else:
                    if min(self.T_targ > 0):
                        raise ValueError('【Error】反应谱缺少T=0的谱值')
                    Sa0 = self._get_y(self.T_targ, self.Sa_targ, 0)
                SF = Sa0 / Sa[0]
            elif self.approach == 'b':
                if type(self.para_scaling) is tuple and len(self.para_scaling) == 2:
                    Sa_a, Ta = self.para_scaling  # 目标值
                else:
                    Ta = self.para_scaling
                    Sa_a = self._get_y(self.T_targ, self.Sa_targ, Ta)
                Sa_spec = self._get_y(T, Sa, Ta)  # 当前值
                SF = Sa_a / Sa_spec
            elif self.approach == 'c':
                Ta, Tb = self.para_scaling
                learning_rate = 0.01  # 学习率
                num_iterations = 1000  # 迭代次数
                Sa_spec_list = Sa[(Ta<=T) & (T<=Tb)]  # 当前值
                Sa_targ_list = self.Sa_targ[(Ta<=self.T_targ) & (self.T_targ<=Tb)]  # 目标值
                init_SF = np.mean(Sa_targ_list) / np.mean(Sa_spec_list)  # 初始缩放系数
                SF = self._gradient_descent(Sa_spec_list, Sa_targ_list, init_SF, learning_rate, num_iterations)
            elif self.approach == 'd':
                if len(self.para_scaling) == 2:
                    Ta, Tb = self.para_scaling
                    Sa_list_targ = self.Sa_targ[(Ta<=self.T_targ) & (self.T_targ<=Tb)]
                    Sa_avg_targ = self._geometric_mean(Sa_list_targ)  # 目标值
                elif len(self.para_scaling) == 3:
                    Ta, Tb, Sa_avg_targ = self.para_scaling  # 目标值
                Sa_list_spec = Sa[(Ta<=T) & (T<=Tb)]
                Sa_avg_spec = self._geometric_mean(Sa_list_spec)  # 当前值
                SF = Sa_avg_targ / Sa_avg_spec
            elif self.approach == 'e':
                SF = self.para_scaling
            else:
                raise ValueError('【Error】参数approach错误')
            # 3 计算匹配分数 (缩放后值-目标谱值)/(目标谱值)
            error = 0  # 误差
            error_ls = []
            Sa *= SF  # 当前地震动缩放后的反应谱
            for j, rule in enumerate(self.rules):
                weight = self.norm_weight[j]
                para = self.para_match[j]
                if rule == 'full':
                    Ta = min(self.T_targ)
                    Tb = max(self.T_targ)
                    Sa_spec_list = Sa[(Ta<=T) & (T<=Tb)]
                    Sa_targ_list = self.Sa_targ[(Ta<=self.T_targ) & (self.T_targ<=Tb)]
                    NRMSE = self._RMSE(Sa_spec_list, Sa_targ_list) / np.mean(Sa_targ_list)
                    error += NRMSE * weight
                    error_ls.append(NRMSE)
                elif rule == 'a':
                    Ta = 0
                    Sa_spec_0 = self._get_y(T, Sa, Ta)
                    Sa_targ_0 = self._get_y(self.T_targ, self.Sa_targ, Ta)
                    error += abs(Sa_spec_0 - Sa_targ_0) / Sa_targ_0 * weight
                    error_ls.append((Sa_spec_0 - Sa_targ_0) / Sa_targ_0)
                elif rule == 'b':
                    Ta = para
                    Sa_spec_0 = self._get_y(T, Sa, Ta)
                    Sa_targ_0 = self._get_y(self.T_targ, self.Sa_targ, Ta)
                    error += abs(Sa_spec_0 - Sa_targ_0) / Sa_targ_0 * weight
                    error_ls.append((Sa_spec_0 - Sa_targ_0) / Sa_targ_0)
                elif rule == 'c':
                    Ta, Tb = para
                    Sa_spec_list = Sa[(Ta<=T) & (T<=Tb)]
                    Sa_targ_list = self.Sa_targ[(Ta<=self.T_targ) & (self.T_targ<=Tb)]
                    NRMSE = self._RMSE(Sa_spec_list, Sa_targ_list) / np.mean(Sa_targ_list)
                    error += NRMSE * weight
                    error_ls.append(NRMSE)
                elif rule == 'd':
                    Ta, Tb = para
                    Sa_spec_list = Sa[(Ta<=T) & (T<=Tb)]
                    Sa_targ_list = self.Sa_targ[(Ta<=self.T_targ) & (self.T_targ<=Tb)]
                    Sa_spec_avg = self._geometric_mean(Sa_spec_list)
                    Sa_targ_avg = self._geometric_mean(Sa_targ_list)
                    error += abs(Sa_spec_avg - Sa_targ_avg) / Sa_targ_avg * weight
                    error_ls.append((Sa_spec_avg - Sa_targ_avg) / Sa_targ_avg)
                else:
                    raise ValueError('【Error】参数rule错误')
            file_SF[file] = SF
            file_error[file] = (error, error_ls)
        # 4 筛选缩放系数，地震动事件数量，PGA
        if self.range_scale_factor:
            SF_a, SF_b = self.range_scale_factor
            for file, SF in file_SF.copy().items():
                if not SF_a <= SF <= SF_b:
                    del file_SF[file]
                    del file_error[file]
        if self.range_N_events:
            event_number = {}  # {地震动: 已出现的次数}
            for file in file_SF.copy().keys():
                event = file_event[file]
                if event not in event_number.keys():
                    event_number[event] = 1
                    continue
                if event_number[event] < self.range_N_events:
                    event_number[event] += 1
                    continue
                del file_SF[file]
                del file_error[file]
        if self.range_PGA:
            PGA_a, PGA_b = self.range_PGA
            for file in file_SF.copy().keys():
                PGA = file_PGA[file]
                if not PGA_a <= PGA <= PGA_b:
                    del file_SF[file]
                    del file_error[file]
        if len(file_SF) < number:
            self._write(f'【Warning】符合条件的地震动数量({len(file_SF)})小于期望值({number})')
            number = len(file_SF)
        print(f'符合条件的地震动数量：{len(file_SF)}')
        files_selection = sorted(file_error, key=lambda k: file_error[k][0])[:number]  # 选波结果（带排列）
        # 绘制反应谱
        f_accec = h5py.File(self.file_accec, 'r')
        Sa_sum = np.zeros(len(T))
        label = 'Individual'
        individual_spec = T
        for file in files_selection:
            Sa = f_spec[file][:] * file_SF[file]
            Sa_sum += Sa
            plt.plot(T, Sa, color='#A6A6A6', label=label)
            individual_spec = np.column_stack((individual_spec, Sa))
            if label:
                label = None
        plt.plot(self.T_targ0, self.Sa_targ0, label='Target', color='black', lw=3)
        plt.plot(T, Sa_sum / len(files_selection), color='red', label='Mean', lw=3)
        plt.xlim(min(self.T_targ0), max(self.T_targ0))
        plt.title('Selected records')
        plt.xlabel('T [s]')
        plt.ylabel('Sa [g]')
        plt.legend()
        f_info.close()
        f_spec.close()
        f_accec.close()
        records.target_spec = np.column_stack((self.T_targ0, self.Sa_targ0))
        records.individual_spec = individual_spec
        records.mean_spec = np.column_stack((T, Sa_sum / len(files_selection)))
        file_SF_error = {}  # {地震名: (缩放系数, 匹配误差)}
        for file in files_selection:
            SF = file_SF[file]
            error, error_ls = file_error[file][0], file_error[file][1]
            file_SF_error[file] = (SF, error, error_ls)
        return files_selection, file_SF_error


    def extract_records(self, RSN: int=None, RSN_list: list[int]=None,
                        RSN_range: list[int, int]=None, files: list=[], file_SF_error: dict[str, tuple[float, float, list]]={},
                        write_unscaled_record: bool=True, write_norm_record: bool=True, write_scaled_records: bool=True):
        """提取地震动数据

        Args:
            RSN (int, optional): 按给定的单个RSN序号提取，默认None
            RSN_list (list, optional): 按RSN列表提取，默认None
            RSN_range (list, optional): 按RSN范围提取，默认None
            files (list, optional): 按地震动文件名提取，默认[]
            file_SF_error (dict, optional): 地震文件-缩放系数字典，默认{}
            write_unscaled_record (bool, optional): 是否写入未缩放地震动，默认True
            write_norm_record (bool, optional): 是否写入归一化地震动，默认True
            write_scaled_records (bool, optional): 是否写入缩放后地震动，默认True
        """
        print('正在提取地震动...')
        records = self.records
        output_dir = self.output_dir
        plt.savefig(output_dir/'反应谱-规范谱对比.jpg', dpi=600)
        img = Image.open(output_dir/'反应谱-规范谱对比.jpg')
        records.img = img
        print('选波完成，请查看反应谱曲线')
        plt.show()
        if write_unscaled_record:
            self._new_folder(output_dir/'未缩放地震动')
        if write_norm_record:
            self._new_folder(output_dir/'归一化地震动')
        if write_scaled_records:
            self._new_folder(output_dir/'缩放后地震动')
        f_info = h5py.File(self.file_info, 'r')
        f_spec = h5py.File(self.file_spec, 'r')
        f_A = h5py.File(self.file_accec, 'r')
        f_V = h5py.File(self.file_vel, 'r')
        f_D = h5py.File(self.file_disp, 'r')
        # 找到地震动文件名
        if RSN:
            files.extend(self._extract_one_RSN(RSN, f_info))
        if RSN_list:
            for RSN_ in RSN_list:
                files.extend(self._extract_one_RSN(RSN_, f_info))
        if RSN_range:
            RSN_list_ = list(range(RSN_range))
            RSN_list_.append(RSN_range[-1])
            for RSN__ in RSN_list_:
                files.extend(self._extract_one_RSN(RSN__, f_info))
        # 读取数据
        df_info_columns = ['No.', 'RSN', 'earthquake_name', 'component', 'Rjb (km)', 'R_rup (km)',
                           'Tp-pluse (s)', 'arias Intensity (m/s)','5-75% Duration (s)',
                           '5-95% Duration (s)', 'duration (s)', 'magnitude', 'mechanism', 'station','Vs30 (m/s)',
                           'year', 'PGA (g)', 'PGV (mm/s)', 'PGD (mm)', 'dt', 'NPTS', 'scale factor', 'Norm. error']
        df_info_columns += [f'error_{i}' for i in self.rules]
        df_info = pd.DataFrame(data=None, columns=df_info_columns)
        # Ta, Tb = min(self.T_targ0), max(self.T_targ0)
        df_spec = pd.DataFrame(data=self.T_spec, columns=['T (s)'])  # 未缩放反应谱
        df_scaled_spec = pd.DataFrame(data=self.T_spec, columns=['T (s)'])  # 缩放后反应谱
        data_spec_sum = np.zeros(len(self.T_spec))
        data_scaled_spec_sum = np.zeros(len(self.T_spec))
        for i, file_stem in enumerate(files):
            ds_A = f_A[file_stem + '.AT2']
            ds_V = f_V[file_stem + '.VT2']
            ds_D = f_D[file_stem + '.DT2']
            RSN = ds_A.attrs['RSN']
            PGA = ds_A.attrs['PGA']
            PGV = ds_V.attrs['PGV']
            PGD = ds_D.attrs['PGD']
            data = ds_A[:]
            dt = ds_A.attrs['dt']
            NPTS = ds_A.attrs['NPTS']
            ds_info = f_info[f'RSN{RSN}']
            Rjb = ds_info.attrs['Rjb']
            Rrup = ds_info.attrs['Rrup']
            Tp = ds_info.attrs['Tp']
            arias = ds_info.attrs['arias']
            D_5_75 = ds_info.attrs['duration_5_75']
            D_5_95 = ds_info.attrs['duration_5_95']
            duration = ds_info.attrs['duration']
            earthquake_name = ds_info.attrs['earthquake_name']
            magnitude = ds_info.attrs['magnitude']
            mechanism = ds_info.attrs['mechanism']
            station = ds_info.attrs['station']
            vs30 = ds_info.attrs['vs30']
            year = ds_info.attrs['year']
            if file_stem in file_SF_error.keys():
                SF, error, error_ls = file_SF_error[file_stem]
                data_scaled = data * SF
            else:
                SF, error, error_ls = '-', '-', []
                data_scaled = data
            if ds_info.attrs['H1_file'] == file_stem:
                component = 'H1'
            elif ds_info.attrs['H2_file'] == file_stem:
                component = 'H2'
            elif ds_info.attrs['V_file'] == file_stem:
                component = 'V'
            else:
                raise ValueError('【Error】1')
            line = [i+1, RSN, earthquake_name, component, Rjb, Rrup, Tp, arias,
                    D_5_75, D_5_95, duration, magnitude, mechanism, station, vs30,
                    year, PGA, PGV, PGD, dt, NPTS, SF, error, *error_ls]
            df_info.loc[len(df_info.index)] = line
            data_spec = f_spec[file_stem][:]
            data_scaled_spec = f_spec[file_stem][:] * SF
            data_spec_sum += data_spec
            data_scaled_spec_sum += data_scaled_spec
            df_spec[f'No. {i+1}'] = data_spec
            df_scaled_spec[f'No. {i+1}'] = data_scaled_spec
            earthquake_name_to_file = earthquake_name.replace('/', '_')  # 文件名不得出现"/"、"\"
            earthquake_name_to_file = earthquake_name_to_file.replace('\\', '_')
            if write_unscaled_record:
                np.savetxt(output_dir/'未缩放地震动'/f'No{i+1}_RSN{RSN}_{earthquake_name_to_file}_{NPTS}_{dt}.txt', data)
            if write_norm_record:
                np.savetxt(output_dir/'归一化地震动'/f'No{i+1}_RSN{RSN}_{earthquake_name_to_file}_{NPTS}_{dt}.txt', self._normalize(data))
            if write_scaled_records:
                np.savetxt(output_dir/'缩放后地震动'/f'No{i+1}_RSN{RSN}_{earthquake_name_to_file}_{NPTS}_{dt}.txt', data_scaled)
            records._add_record(data, data_spec, SF, dt, 'A')
        records.info = df_info
        data_spec_mean = data_spec_sum / len(files)
        data_scaled_spec_mean = data_scaled_spec_sum / len(files)
        df_spec['Mean'] = data_spec_mean
        df_scaled_spec['Mean'] = data_scaled_spec_mean
        df_info.to_csv(output_dir/'地震动信息.csv', index=None)
        df_spec.to_csv(output_dir/'未缩放反应谱.csv', index=False)
        df_scaled_spec.to_csv(output_dir/'缩放后反应谱.csv', index=False)
        file_path = output_dir / f'records.opju'
        with WriteOrigin(op, file_path, 'results') as f_op:
            print('正在写入origin文件...\r', end='')
            f_op.delete_obj('Book1')
            wb1 = op.new_book('w')
            wb1.lname = '选波信息'
            ws: originpro.WSheet = wb1[0]
            ws.from_df(df_info)  # 写入选波信息
            wb2 = op.new_book('w')
            wb2.lname = '反应谱'
            ws: originpro.WSheet = wb2[0]
            ws.from_list(0, records.individual_spec[:, 0], 'T', 's', axis='X')
            for col in range(1, records.individual_spec.shape[1]):
                ws.from_list(col, records.individual_spec[:, col], 'Sa', 'g', axis='Y')
            ws.from_list(col + 1, records.target_spec[:, 0], 'T', 's', axis='X')
            ws.from_list(col + 2, records.target_spec[:, 1], 'Sa', 'g', 'Target', axis='Y')  # 写入目标谱
            ws.from_list(col + 3, records.mean_spec[:, 0], 'T', 's', axis='X')
            ws.from_list(col + 4, records.mean_spec[:, 1], 'Sa', 'g', 'Mean', axis='Y')  # 写入平均谱
        print('已导出origin文件          ')
        with open(output_dir/'选波参数设置.txt', 'w') as f:
            f.write(self.selecting_text)
        records.selecting_text = self.selecting_text  # 记录选波设置文本
        f_info.close()
        f_spec.close()
        f_A.close()
        f_V.close()
        f_D.close()
        records._to_pkl('records', output_dir)
        print('完成！')

    def _extract_one_RSN(self, RSN: int, f_info: h5py.File):
        """提取一组RSN"""
        ds_name = f'RSN{RSN}'
        RSN_files = []
        if ds_name not in f_info:
            self._write(f'【Warning】数据库缺少RSN{RSN}')
        elif f_info[ds_name].attrs['H1_file'] != '-':
            RSN_files.append(f_info[ds_name].attrs['H1_file'])
        elif f_info[ds_name].attrs['H2_file'] != '-':
            RSN_files.append(f_info[ds_name].attrs['H2_file'])
        elif f_info[ds_name].attrs['V_file'] != '-':
            RSN_files.append(f_info[ds_name].attrs['V_file'])
        return RSN_files

    def _write(self, text: str, end='\n'):
        print(text)
        self.selecting_text += text + end


    @classmethod
    def _check_file(cls, file_path: str | Path):
        """检查文件是否存在且为最新版本"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileExistsError(f'无法找到文件：{file_path}')
        with h5py.File(file_path, 'r') as f:
            if 'VERSION' not in f:
                version = '1.0'
            else:
                version = f['VERSION'][()].decode('utf-8')
        if not version == cls.version:
            raise FileExistsError(f'数据库文件版本过旧（{version} < {cls.version}），请使用update.py进行升级')


    @staticmethod
    def _get_y(x: np.ndarray, y: np.ndarray, x0: float | int):
        """求曲线在某点处的值"""
        for i in range(1, len(x)):
            if not x[i] > x[i - 1]:
                raise ValueError('【Error】x序列不是单调递增的')
        if x0 < x[0] or x0 > x[-1]:
            raise ValueError(f'【Error】x0超出范围\nx0 = {x0}, range=[{x[0]}, {x[-1]}]')
        for i in range(1, len(x)):
            xim1 = x[i - 1]
            xi = x[i]
            yim1 = y[i - 1]
            yi = y[i]
            if xim1 <= x0 <= xi:
                k = (yi - yim1) / (xi - xim1)
                y0 = k * (x0 - xi) + yi
                break
        return y0
    
    @staticmethod
    def _geometric_mean(data):
        """计算几何平均数"""
        total = 1
        n = len(data)
        for i in data:
            total *= pow(i, 1 / n)
        return total
        
    @staticmethod
    def _gradient_descent(a, b, init_SF, learning_rate, num_iterations):
        """梯度下降"""
        f = init_SF
        for _ in range(num_iterations):
            error = a * f - b
            gradient = 2 * np.dot(error, a) / len(a)
            f -= learning_rate * gradient
        return f
    
    @staticmethod
    def _RMSE(y1, y2):
        """计算均方根值"""
        result = np.sqrt(sum((y1 - y2) ** 2) / len(y1))
        return result

    @staticmethod
    def _new_folder(folder_path: str | Path):
        """新建文件夹"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            os.makedirs(folder_path)

    @staticmethod
    def _normalize(data: np.ndarray):
        """归一化数组"""
        peak = max(abs(data))
        data_norm = data / peak
        return data_norm
    

class WriteOrigin():
    def __init__(self, op: originpro, opju_file: Path, folder_name:str, set_show: bool=False) -> None:
        """写入origin文件

        Args:
            op (originpro): originpro对象
            opju_file (Path): opju的路径，数据将保存到这个文件，若该opju文件不存在则将创建，若存在则将打开并写入数据
            set_show (bool, optional): 是否显示origin窗口, 默认False
        """
        self.op = op
        self.opju_file = opju_file
        self.folder_name = folder_name
        self.set_show = set_show

    def __enter__(self):
        if self.op and self.op.oext:
            sys.excepthook = self.origin_shutdown_exception_hook
        if self.op.oext:
            self.op.set_show(self.set_show)
        if not self.opju_file.exists():
            # 如果opju文件不存在则创建
            self.op.new(self.opju_file.absolute().as_posix())
            fd = self.op.pe.active_folder()
            fd.name = self.folder_name
        else:
            # 如果存在则打开
            self.op.open(self.opju_file.absolute().as_posix())
            res0 = None
            while True:
                res1 = self.op.pe.cd('..')
                if res0 == res1:
                    break  # 返回上级origin文件夹，直至最顶层
                res0 = res1
            self.op.pe.mkdir(self.folder_name, True)  # 创建origin文件夹，如果已经存在则不会创建
            self.op.pe.cd(self.folder_name)  # 进入origin文件夹
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.op.save(str(self.opju_file.absolute()))
        if self.op.oext:
            self.op.exit()
        return False
    
    def origin_shutdown_exception_hook(self, exctype, value, traceback):
        '''Ensures Origin gets shut down if an uncaught exception'''
        self.op.exit()
        sys.__excepthook__(exctype, value, traceback)

    def delete_obj(self, obj_name: str):
        """删除表格

        Args:
            obj_name (str.WBook): 表格名
        """
        wb = self.op.find_book('w', obj_name)
        if wb:
            wb.destroy()

if __name__ == "__main__":
    file_acc = r'G:\NGAWest2\Acceleration.hdf5'
    file_vel = r'G:\NGAWest2\Velocity.hdf5'
    file_disp = r'G:\NGAWest2\Displacement.hdf5'
    file_spec = r'G:\NGAWest2\Spectra.hdf5'
    file_info = r'G:\NGAWest2\Info.hdf5'
    selector = Selecting()
    selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
    selector.target_spectra('DBE谱J.txt')
    selector.scaling_approach('a', para=1)
    selector.matching_rules(rules=['c'], para=[(0.1, 2)], weight=[1])
    selector.constrain_range(N_events=5, scale_factor=(0.2, 10), component=['H1'])
    selected_records, records_info = selector.run(35)
    selector.extract_records('选波', files=selected_records, file_SF_error=records_info)
    # selector.check_database()






