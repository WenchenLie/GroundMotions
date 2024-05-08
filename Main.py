from Selecting import Selecting


file_acc = r'G:\NGAWest2\Acceleration.hdf5'
file_vel = r'G:\NGAWest2\Velocity.hdf5'
file_disp = r'G:\NGAWest2\Displacement.hdf5'
file_spec = r'G:\NGAWest2\Spectra.hdf5'
file_info = r'G:\NGAWest2\Info.hdf5'
selector = Selecting()
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('RCF6S_DBE.txt', scale=20)
selector.scaling_approach('a')
selector.matching_rules(rules=['full'], para=[None], weight=[1])
selector.constrain_range(N_events=100, magnitude=(5, 20), vs30=(180, 1000), strike_slip='e', component=['H1', 'H2'])
selected_records, records_info = selector.run(3046)
selected_records.sort(key=lambda x: int(x.split('_')[0][3:]))
with open('3046records.txt', 'w') as f:
    f.writelines([i + '\n' for i in selected_records])
# selector.extract_records('选波', files=selected_records, file_SF_error=records_info)
# selector.check_database()

