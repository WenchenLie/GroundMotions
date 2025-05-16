"""
用于升级地震动数据库hdf5文件
"""
from pathlib import Path
import h5py
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from seismicutils.spectrum import spectrum


def _get_version(file_info: str | Path):
    """获取版本号"""
    f = h5py.File(file_info, 'r')
    if not 'VERSION' in f:
        return '1.0'
    version = f['VERSION'][()].decode('utf-8')
    f.close()
    return version

def _check_file_exists(file: Path | str):
    file = Path(file)
    if not file.exists():
        path_ = file.absolute().as_posix()
        raise FileExistsError(f'文件不存在：{path_}')

def _update_10_20(
        file_acc: str | Path,
        file_vel: str | Path,
        file_disp: str | Path,
        file_spec: str | Path,
        file_info: str | Path,
):
    """从1.0版本升级到2.0
    增加地震动时间步长、持时到info文件
    """
    if float(_get_version(file_info)) >= 2.0:
        return
    print('更新中... (1.0 -> 2.0)  ', end='')
    f_acc = h5py.File(file_acc, 'a')
    f_vel = h5py.File(file_vel, 'a')
    f_disp = h5py.File(file_disp, 'a')
    f_spec = h5py.File(file_spec, 'a')
    f_info = h5py.File(file_info, 'a')
    f_info['VERSION'] = '2.0'
    f_vel['VERSION'] = '2.0'
    f_disp['VERSION'] = '2.0'
    f_spec['VERSION'] = '2.0'
    f_acc['VERSION'] = '2.0'
    for name in [f'RSN{i}' for i in range(1, 21541)]:
        if not name in f_info:
            continue
        V_file = f_info[name].attrs['Vertical_file']
        f_info[name].attrs['V_file'] = V_file
        del f_info[name].attrs['Vertical_file']
        H1_file = f_info[name].attrs['H1_file'] + '.AT2'
        NPTS = f_acc[H1_file].attrs['NPTS']
        dt = f_acc[H1_file].attrs['dt']
        duration = (NPTS - 1) * dt
        f_info[name].attrs['dt'] = dt
        f_info[name].attrs['duration'] = round(duration, 8)
    f_acc.close()
    f_vel.close()
    f_disp.close()
    f_spec.close()
    f_info.close()
    print('完成')

def _update_20_21(
        file_acc: str | Path,
        file_vel: str | Path,
        file_disp: str | Path,
        file_spec: str | Path,
        file_info: str | Path,
        *patch):
    """从2.0版本升级到2.1，增加了多条地震动
    """
    if float(_get_version(file_info)) >= 2.1:
        return
    print('更新中... (2.0 -> 2.1)')
    for file in patch:
        if Path(file).name == 'update_20_21.patch':
            break
    else:
        raise FileNotFoundError(f'未找到补丁update_20_21.patch')
    with open(file, 'rb') as f:
        records: dict = pickle.load(f)
    f_acc = h5py.File(file_acc, 'a')
    f_vel = h5py.File(file_vel, 'a')
    f_disp = h5py.File(file_disp, 'a')
    f_spec = h5py.File(file_spec, 'a')
    f_info = h5py.File(file_info, 'a')
    T = np.arange(0, 10.01, 0.01)
    for n in records.keys():
        if n == 'version':
            continue
        RSN_ls = records[n]['RSN']
        Tp_ls = records[n]['Tp']
        D5_75_ls = records[n]['D5_75']
        D5_95_ls = records[n]['D5_95']
        arias_ls = records[n]['arias']
        earthquake_name_ls = records[n]['earthquake_name']
        year_ls = records[n]['year']
        station_ls = records[n]['station']
        magnitude_ls = records[n]['magnitude']
        mechanism_ls = records[n]['mechanism']
        Rjb_ls = records[n]['Rjb']
        Rrub_ls = records[n]['Rrub']
        v30_ls = records[n]['v30']
        H1_file_ls = records[n]['H1_file']
        H2_file_ls = records[n]['H2_file']
        V_file_ls = records[n]['V_file']
        for i in range(len(RSN_ls)):
            RSN = int(RSN_ls[i])
            print(f'  {i+1}/{len(RSN_ls)}, add RSN{RSN}   \r', end='')
            Tp = Tp_ls[i]
            D5_75 = float(D5_75_ls[i])
            D5_95 = float(D5_95_ls[i])
            arias = float(arias_ls[i])
            earthquake_name = earthquake_name_ls[i].strip('"')
            year = int(year_ls[i])
            station = station_ls[i].strip('"')
            magnitude = float(magnitude_ls[i])
            mechanism = mechanism_ls[i]
            Rjb = float(Rjb_ls[i])
            Rrub = float(Rrub_ls[i])
            v30 = float(v30_ls[i])
            H1_file = H1_file_ls[i][:-4]
            H2_file = H2_file_ls[i][:-4]
            V_file = V_file_ls[i][:-4]
            dt =  records[n]['data']['H1'][str(RSN)][0][1]
            NPTS =  records[n]['data']['H1'][str(RSN)][0][0]
            duration = (NPTS - 1) * dt
            f_info.create_dataset(f'RSN{RSN}', dtype=int)
            f_info[f'RSN{RSN}'].attrs['H1_file'] = H1_file
            f_info[f'RSN{RSN}'].attrs['H2_file'] = H2_file
            f_info[f'RSN{RSN}'].attrs['RSN'] = RSN
            f_info[f'RSN{RSN}'].attrs['Rjb'] = Rjb
            f_info[f'RSN{RSN}'].attrs['Rrub'] = Rrub
            f_info[f'RSN{RSN}'].attrs['Tp'] = Tp
            f_info[f'RSN{RSN}'].attrs['V_file'] = V_file
            f_info[f'RSN{RSN}'].attrs['arias'] = arias
            f_info[f'RSN{RSN}'].attrs['dt'] = dt
            f_info[f'RSN{RSN}'].attrs['duration'] = duration
            f_info[f'RSN{RSN}'].attrs['duration_5_75'] = D5_75
            f_info[f'RSN{RSN}'].attrs['duration_5_95'] = D5_95
            f_info[f'RSN{RSN}'].attrs['earthquake_name'] = earthquake_name
            f_info[f'RSN{RSN}'].attrs['magnitude'] = magnitude
            f_info[f'RSN{RSN}'].attrs['mechanism'] = mechanism
            f_info[f'RSN{RSN}'].attrs['station'] = station
            f_info[f'RSN{RSN}'].attrs['v30'] = v30
            f_info[f'RSN{RSN}'].attrs['year'] = year
            NPTS_dt_th_H1 = records[n]['data']['H1'][str(RSN)]
            NPTS_dt_th_H2 = records[n]['data']['H2'][str(RSN)]
            NPTS_dt_th_V = records[n]['data']['V'][str(RSN)]
            if NPTS_dt_th_H1 is not None:
                # H1分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_H1[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_H1[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_H1[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(H1_file, data=RSA)
                f_spec[H1_file].attrs['NPTS'] = NPTS_A
                f_spec[H1_file].attrs['PGA'] = max(abs(th_A))
                f_spec[H1_file].attrs['RSN'] = RSN
                f_spec[H1_file].attrs['dt'] = dt
                f_acc.create_dataset(H1_file+'.AT2', data=th_A)
                f_acc[H1_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[H1_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[H1_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[H1_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(H1_file+'.VT2', data=th_V)
                f_vel[H1_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[H1_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[H1_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[H1_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(H1_file+'.DT2', data=th_D)
                f_disp[H1_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[H1_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[H1_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[H1_file+'.DT2'].attrs['dt'] = dt_D
            if NPTS_dt_th_H2 is not None:
                # H2分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_H2[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_H2[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_H2[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(H2_file, data=RSA)
                f_spec[H2_file].attrs['NPTS'] = NPTS_A
                f_spec[H2_file].attrs['PGA'] = max(abs(th_A))
                f_spec[H2_file].attrs['RSN'] = RSN
                f_spec[H2_file].attrs['dt'] = dt
                f_acc.create_dataset(H2_file+'.AT2', data=th_A)
                f_acc[H2_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[H2_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[H2_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[H2_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(H2_file+'.VT2', data=th_V)
                f_vel[H2_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[H2_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[H2_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[H2_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(H2_file+'.DT2', data=th_D)
                f_disp[H2_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[H2_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[H2_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[H2_file+'.DT2'].attrs['dt'] = dt_D
            if NPTS_dt_th_V is not None:
                # V分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_V[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_V[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_V[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(V_file, data=RSA)
                f_spec[V_file].attrs['NPTS'] = NPTS_A
                f_spec[V_file].attrs['PGA'] = max(abs(th_A))
                f_spec[V_file].attrs['RSN'] = RSN
                f_spec[V_file].attrs['dt'] = dt
                f_acc.create_dataset(V_file+'.AT2', data=th_A)
                f_acc[V_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[V_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[V_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[V_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(V_file+'.VT2', data=th_V)
                f_vel[V_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[V_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[V_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[V_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(V_file+'.DT2', data=th_D)
                f_disp[V_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[V_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[V_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[V_file+'.DT2'].attrs['dt'] = dt_D
        print('')
    del f_info['VERSION']
    del f_spec['VERSION']
    del f_vel['VERSION']
    del f_disp['VERSION']
    del f_acc['VERSION']
    f_info['VERSION'] = '2.1'
    f_spec['VERSION'] = '2.1'
    f_vel['VERSION'] = '2.1'
    f_disp['VERSION'] = '2.1'
    f_acc['VERSION'] = '2.1'
    f_acc.close()
    f_vel.close()
    f_disp.close()
    f_spec.close()
    f_info.close()
    print('完成')
    

def _convention_update(
    old_version: str,
    new_version: str,
    file_acc: str | Path,
    file_vel: str | Path,
    file_disp: str | Path,
    file_spec: str | Path,
    file_info: str | Path,
    *patch: Path,
):
    """常规的添加地震动的升级"""
    if float(_get_version(file_info)) >= float(new_version):
        return
    print(f'更新中... ({old_version} -> {new_version})')
    expected_patch_name = 'update_' + old_version.replace('.', '') + '_' + new_version.replace('.', '') + '.patch'  # patch文件的命名
    for file in patch:
        if Path(file).name == expected_patch_name:
            break
    else:
        raise FileNotFoundError(f'未找到补丁{expected_patch_name}')
    with open(file, 'rb') as f:
        records: dict = pickle.load(f)
    f_acc = h5py.File(file_acc, 'a')
    f_vel = h5py.File(file_vel, 'a')
    f_disp = h5py.File(file_disp, 'a')
    f_spec = h5py.File(file_spec, 'a')
    f_info = h5py.File(file_info, 'a')
    T = np.arange(0, 10.01, 0.01)
    for n in records.keys():
        if n == 'version':
            continue
        RSN_ls = records[n]['RSN']
        Tp_ls = records[n]['Tp']
        D5_75_ls = records[n]['D5_75']
        D5_95_ls = records[n]['D5_95']
        arias_ls = records[n]['arias']
        earthquake_name_ls = records[n]['earthquake_name']
        year_ls = records[n]['year']
        station_ls = records[n]['station']
        magnitude_ls = records[n]['magnitude']
        mechanism_ls = records[n]['mechanism']
        Rjb_ls = records[n]['Rjb']
        Rrub_ls = records[n]['Rrub']
        v30_ls = records[n]['v30']
        H1_file_ls = records[n]['H1_file']
        H2_file_ls = records[n]['H2_file']
        V_file_ls = records[n]['V_file']
        for i in range(len(RSN_ls)):
            RSN = int(RSN_ls[i])
            print(f'  {i+1}/{len(RSN_ls)}, add RSN{RSN}   \r', end='')
            Tp = Tp_ls[i]
            D5_75 = float(D5_75_ls[i])
            D5_95 = float(D5_95_ls[i])
            arias = float(arias_ls[i])
            earthquake_name = earthquake_name_ls[i].strip('"')
            year = int(year_ls[i])
            station = station_ls[i].strip('"')
            magnitude = float(magnitude_ls[i])
            mechanism = mechanism_ls[i]
            Rjb = float(Rjb_ls[i])
            Rrub = float(Rrub_ls[i])
            v30 = float(v30_ls[i])
            H1_file = H1_file_ls[i][:-4]
            H2_file = H2_file_ls[i][:-4]
            V_file = V_file_ls[i][:-4]
            dt =  records[n]['data']['H1'][str(RSN)][0][1]
            NPTS =  records[n]['data']['H1'][str(RSN)][0][0]
            duration = (NPTS - 1) * dt
            f_info.create_dataset(f'RSN{RSN}', dtype=int)
            f_info[f'RSN{RSN}'].attrs['H1_file'] = H1_file
            f_info[f'RSN{RSN}'].attrs['H2_file'] = H2_file
            f_info[f'RSN{RSN}'].attrs['RSN'] = RSN
            f_info[f'RSN{RSN}'].attrs['Rjb'] = Rjb
            f_info[f'RSN{RSN}'].attrs['Rrub'] = Rrub
            f_info[f'RSN{RSN}'].attrs['Tp'] = Tp
            f_info[f'RSN{RSN}'].attrs['V_file'] = V_file
            f_info[f'RSN{RSN}'].attrs['arias'] = arias
            f_info[f'RSN{RSN}'].attrs['dt'] = dt
            f_info[f'RSN{RSN}'].attrs['duration'] = duration
            f_info[f'RSN{RSN}'].attrs['duration_5_75'] = D5_75
            f_info[f'RSN{RSN}'].attrs['duration_5_95'] = D5_95
            f_info[f'RSN{RSN}'].attrs['earthquake_name'] = earthquake_name
            f_info[f'RSN{RSN}'].attrs['magnitude'] = magnitude
            f_info[f'RSN{RSN}'].attrs['mechanism'] = mechanism
            f_info[f'RSN{RSN}'].attrs['station'] = station
            f_info[f'RSN{RSN}'].attrs['v30'] = v30
            f_info[f'RSN{RSN}'].attrs['year'] = year
            NPTS_dt_th_H1 = records[n]['data']['H1'][str(RSN)]
            NPTS_dt_th_H2 = records[n]['data']['H2'][str(RSN)]
            NPTS_dt_th_V = records[n]['data']['V'][str(RSN)]
            if NPTS_dt_th_H1 is not None:
                # H1分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_H1[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_H1[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_H1[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(H1_file, data=RSA)
                f_spec[H1_file].attrs['NPTS'] = NPTS_A
                f_spec[H1_file].attrs['PGA'] = max(abs(th_A))
                f_spec[H1_file].attrs['RSN'] = RSN
                f_spec[H1_file].attrs['dt'] = dt
                f_acc.create_dataset(H1_file+'.AT2', data=th_A)
                f_acc[H1_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[H1_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[H1_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[H1_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(H1_file+'.VT2', data=th_V)
                f_vel[H1_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[H1_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[H1_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[H1_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(H1_file+'.DT2', data=th_D)
                f_disp[H1_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[H1_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[H1_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[H1_file+'.DT2'].attrs['dt'] = dt_D
            if NPTS_dt_th_H2 is not None:
                # H2分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_H2[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_H2[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_H2[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(H2_file, data=RSA)
                f_spec[H2_file].attrs['NPTS'] = NPTS_A
                f_spec[H2_file].attrs['PGA'] = max(abs(th_A))
                f_spec[H2_file].attrs['RSN'] = RSN
                f_spec[H2_file].attrs['dt'] = dt
                f_acc.create_dataset(H2_file+'.AT2', data=th_A)
                f_acc[H2_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[H2_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[H2_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[H2_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(H2_file+'.VT2', data=th_V)
                f_vel[H2_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[H2_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[H2_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[H2_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(H2_file+'.DT2', data=th_D)
                f_disp[H2_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[H2_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[H2_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[H2_file+'.DT2'].attrs['dt'] = dt_D
            if NPTS_dt_th_V is not None:
                # V分量时程
                NPTS_A, dt_A, th_A = NPTS_dt_th_V[0]
                NPTS_V, dt_V, th_V = NPTS_dt_th_V[1]
                NPTS_D, dt_D, th_D = NPTS_dt_th_V[2]
                RSA, _, _ = spectrum(th_A, dt, T)
                f_spec.create_dataset(V_file, data=RSA)
                f_spec[V_file].attrs['NPTS'] = NPTS_A
                f_spec[V_file].attrs['PGA'] = max(abs(th_A))
                f_spec[V_file].attrs['RSN'] = RSN
                f_spec[V_file].attrs['dt'] = dt
                f_acc.create_dataset(V_file+'.AT2', data=th_A)
                f_acc[V_file+'.AT2'].attrs['NPTS'] = NPTS_A
                f_acc[V_file+'.AT2'].attrs['PGA'] = max(abs(th_A))
                f_acc[V_file+'.AT2'].attrs['RSN'] = RSN
                f_acc[V_file+'.AT2'].attrs['dt'] = dt_A
                f_vel.create_dataset(V_file+'.VT2', data=th_V)
                f_vel[V_file+'.VT2'].attrs['NPTS'] = NPTS_V
                f_vel[V_file+'.VT2'].attrs['PGV'] = max(abs(th_V))
                f_vel[V_file+'.VT2'].attrs['RSN'] = RSN
                f_vel[V_file+'.VT2'].attrs['dt'] = dt_V
                f_disp.create_dataset(V_file+'.DT2', data=th_D)
                f_disp[V_file+'.DT2'].attrs['NPTS'] = NPTS_D
                f_disp[V_file+'.DT2'].attrs['PGD'] = max(abs(th_D))
                f_disp[V_file+'.DT2'].attrs['RSN'] = RSN
                f_disp[V_file+'.DT2'].attrs['dt'] = dt_D
        print('')
    del f_info['VERSION']
    del f_spec['VERSION']
    del f_vel['VERSION']
    del f_disp['VERSION']
    del f_acc['VERSION']
    f_info['VERSION'] = new_version
    f_spec['VERSION'] = new_version
    f_vel['VERSION'] = new_version
    f_disp['VERSION'] = new_version
    f_acc['VERSION'] = new_version
    f_acc.close()
    f_vel.close()
    f_disp.close()
    f_spec.close()
    f_info.close()
    print('完成')


def update(
    file_acc: str | Path,
    file_vel: str | Path,
    file_disp: str | Path,
    file_spec: str | Path,
    file_info: str | Path,
    *patch: Path
):
    _check_file_exists(file_acc)
    _check_file_exists(file_vel)
    _check_file_exists(file_disp)
    _check_file_exists(file_spec)
    _check_file_exists(file_info)
    all_files = (file_acc, file_vel, file_disp, file_spec, file_info)
    _update_10_20(*all_files)
    _update_20_21(*all_files, *patch)
    _convention_update('2.1', '2.2', *all_files, *patch)



if __name__ == "__main__":
    file_acc = r'F:\Projects\GroundMotions\temp\Acceleration.hdf5'
    file_vel = r'F:\Projects\GroundMotions\temp\Velocity.hdf5'
    file_disp = r'F:\Projects\GroundMotions\temp\Displacement.hdf5'
    file_spec = r'F:\Projects\GroundMotions\temp\Spectra.hdf5'
    file_info = r'F:\Projects\GroundMotions\temp\Info.hdf5'
    # patch = r'F:\Projects\GroundMotions\developer\update_20_21.patch'
    patch = r'F:\Projects\GroundMotions\developer\update_21_22.patch'
    update(file_acc, file_vel, file_disp, file_spec, file_info, patch)

"""
.patck文件命名规则：
如从2.0升级成2.1版本，则命名为"update_20_21.patch"
"""
