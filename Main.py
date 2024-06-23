from Selecting import Selecting


file_acc = r'H:\NGAWest2\Acceleration.hdf5'
file_vel = r'H:\NGAWest2\Velocity.hdf5'
file_disp = r'H:\NGAWest2\Displacement.hdf5'
file_spec = r'H:\NGAWest2\Spectra.hdf5'
file_info = r'H:\NGAWest2\Info.hdf5'
selector = Selecting('results/选波')
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('spec_data/DBE_AS.txt', scale=1)
selector.scaling_approach('c', (0.2*1.0, 1.5*1.2))
selector.matching_rules(rules=['c', 'c'], para=[(0.1, 1), (0.2*1.0, 1.5*1.2)], weight=[3, 5])
selector.constrain_range(N_events=2, magnitude=(5.5, 20), vs30=(260, 510), strike_slip='all', component=['H1'], scale_factor=(0, 3))
selected_records, records_info = selector.run(5)
# selected_records.sort(key=lambda x: int(x.split('_')[0][3:]))
# with open('3046records.txt', 'w') as f:
#     f.writelines([i + '\n' for i in selected_records])
selector.extract_records(files=selected_records, file_SF_error=records_info)
# selector.check_database()

