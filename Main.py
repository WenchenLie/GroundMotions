from Selecting import Selecting


file_acc = r'G:\NGAWest2\Acceleration.hdf5'
file_vel = r'G:\NGAWest2\Velocity.hdf5'
file_disp = r'G:\NGAWest2\Displacement.hdf5'
file_spec = r'G:\NGAWest2\Spectra.hdf5'
file_info = r'G:\NGAWest2\Info.hdf5'
selector = Selecting()
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('RCF6S_DBE.txt')
selector.scaling_approach('a')
selector.matching_rules(rules=['b', 'c', 'c'], para=[1.233, (0.1, 1.3), (1.3, 4)], weight=[2, 1, 1])
selector.constrain_range(magnitude=(5, 20), vs30=(180, 1000), strike_slip='e')
selected_records, records_info = selector.run(10000)
# selector.extract_records('选波', files=selected_records, file_SF_error=records_info)
# selector.check_database()

