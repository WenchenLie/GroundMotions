## 基于NGAWest2数据库的选波程序
开发者：列文琛  
日期：2024年2月4日

### 1.实例化选波器

from Selecting import Selecting
selector = Selecting()

### 2.导入地震动数据库文件

# 文件路径
file_acc = r'F:\NGAWest2\Acceleration.hdf5'
file_vel = r'F:\NGAWest2\Velocity.hdf5'
file_disp = r'F:\NGAWest2\Displacement.hdf5'
file_spec = r'F:\NGAWest2\Spectra.hdf5'
file_info = r'F:\NGAWest2\Info.hdf5'
# 导入文件
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)