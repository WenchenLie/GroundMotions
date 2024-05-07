"""
用于升级地震动数据库hdf5文件
"""
from pathlib import Path
import h5py


def _get_version(file_info: str | Path):
    """获取版本号"""
    f = h5py.File(file_info, 'r')
    if not 'VERSION' in f:
        return '1.0'
    version = f['VERSION'][()].decode('utf-8')
    f.close()
    return version


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
    print('更新中... (1.0 -> 2.0)  ', end='')
    if float(_get_version(file_info)) >= 2.0:
        print('完成')
        return
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


def update(
        file_acc: str | Path,
        file_vel: str | Path,
        file_disp: str | Path,
        file_spec: str | Path,
        file_info: str | Path,
):
    _update_10_20(file_acc, file_vel, file_disp, file_spec, file_info)




if __name__ == "__main__":
    file_acc = r'G:\NGAWest2\Acceleration.hdf5'
    file_vel = r'G:\NGAWest2\Velocity.hdf5'
    file_disp = r'G:\NGAWest2\Displacement.hdf5'
    file_spec = r'G:\NGAWest2\Spectra.hdf5'
    file_info = r'G:\NGAWest2\Info.hdf5'
    update(file_acc, file_vel, file_disp, file_spec, file_info)
