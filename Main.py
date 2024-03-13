from Selecting import Selecting


file_acc = r'G:\NGAWest2\Acceleration.hdf5'
file_vel = r'G:\NGAWest2\Velocity.hdf5'
file_disp = r'G:\NGAWest2\Displacement.hdf5'
file_spec = r'G:\NGAWest2\Spectra.hdf5'
file_info = r'G:\NGAWest2\Info.hdf5'
selector = Selecting()
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('DBE谱.txt')
selector.scaling_approach('a')
selector.matching_rules(rules=['c', 'b', 'b'], para=[[0.6, 1.8], [4.121], [4.09]], weight=[1.5, 1.5, 1.5])
selector.constrain_range(N_events=3)
selected_records, records_info = selector.run(15)
selector.extract_records(r'C:\Users\admin\Desktop\选波', files=selected_records, file_SF_error=records_info)
# selector.check_database()

