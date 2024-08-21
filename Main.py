from Selecting import Selecting


file_acc = r'H:\NGAWest2\Acceleration.hdf5'
file_vel = r'H:\NGAWest2\Velocity.hdf5'
file_disp = r'H:\NGAWest2\Displacement.hdf5'
file_spec = r'H:\NGAWest2\Spectra.hdf5'
file_info = r'H:\NGAWest2\Info.hdf5'
Selecting.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector = Selecting(r'G:\OneDrive - e.gzhu.edu.cn\TSSCB_SDOFSimpleAnalysis\选波\结果')
selector.target_spectra(r"G:\OneDrive - e.gzhu.edu.cn\TSSCB_SDOFSimpleAnalysis\选波\DBE_spectrum.out", scale=20)
selector.scaling_approach('c', para=[0, 3])
selector.matching_rules(rules=['full', 'c'], para=[None, (0, 3)], weight=[1, 1])
selector.constrain_range(magnitude=(5, 20), vs30=(180, 360), strike_slip='e', component=['H1', 'H2'])
selected_records, records_info = selector.run(100)
# selected_records.sort(key=lambda x: int(x.split('_')[0][3:]))
# with open('3046records.txt', 'w') as f:
#     f.writelines([i + '\n' for i in selected_records])
selector.get_results(files=selected_records, file_SF_error=records_info)
# selector.check_database()

