from Selecting import Selecting


file_acc = r'G:\NGAWest2\Acceleration.hdf5'
file_vel = r'G:\NGAWest2\Velocity.hdf5'
file_disp = r'G:\NGAWest2\Displacement.hdf5'
file_spec = r'G:\NGAWest2\Spectra.hdf5'
file_info = r'G:\NGAWest2\Info.hdf5'
selector = Selecting()
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('DBE谱J.txt')
selector.scaling_approach('c', para=(0.5, 2))
selector.matching_rules(rules=['c'], para=[(0.5, 2)], weight=[1])
selector.constrain_range(N_events=5, scale_factor=(0.2, 10), component=['H1'])
selected_records, records_info = selector.run(35)
selector.extract_records('选波', files=selected_records, file_SF_error=records_info)
# selector.check_database()

