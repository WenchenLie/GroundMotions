from Selecting import Selecting
import numpy as np


file_acc = r'H:\NGAWest2\V2.2\Acceleration.hdf5'
file_vel = r'H:\NGAWest2\V2.2\Velocity.hdf5'
file_disp = r'H:\NGAWest2\V2.2\Displacement.hdf5'
file_spec = r'H:\NGAWest2\V2.2\Spectra.hdf5'
file_info = r'H:\NGAWest2\V2.2\Info.hdf5'
Selecting.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector = Selecting(r'results_test')
selector.target_spectra(r"F:\Projects\SCBF-RC_modeling\data1\DBE_AS.txt", scale=1.5)
selector.scaling_approach('c', para=(0.25, 3))
selector.matching_rules(rules=['full'], para=[None], weight=[1])
# B-(760, 1525), C-(360, 760), D-(180, 360)
selector.constrain_range(magnitude=(5, 20), component=['H1'], duration=(25, 100), pulse=False, vs30=(180, 360))
selected_records, records_info = selector.run(30)
# selected_records.sort(key=lambda x: int(x.split('_')[0][3:]))
# # with open('3046records.txt', 'w') as f:
# #     f.writelines([i + '\n' for i in selected_records])
selector.get_results(files=selected_records, file_SF_error=records_info)
# # selector.check_database()

# RSN_list = np.loadtxt(r"C:\Users\admin\Desktop\新建 文本文档.txt", dtype=int).tolist()
# Selecting.extract_records(r'C:\Users\admin\Desktop\Results222', RSN_list=RSN_list, components=['H1', 'H2'])




