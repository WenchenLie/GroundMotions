from Selecting import Selecting


file_acc = r'G:\NGAWest2\Acceleration.hdf5'
file_vel = r'G:\NGAWest2\Velocity.hdf5'
file_disp = r'G:\NGAWest2\Displacement.hdf5'
file_spec = r'G:\NGAWest2\Spectra.hdf5'
file_info = r'G:\NGAWest2\Info.hdf5'
selector = Selecting()
selector.import_files(file_acc, file_vel, file_disp, file_spec, file_info)
selector.target_spectra('潮州项目DBE谱.txt')
selector.scaling_approach('a')
selector.matching_rules(rules=['full', 'b', 'b', 'b', 'b', 'b', 'b'], para=[None, 4.535, 4.514, 4.062, 2.155, 2.031, 1.931], weight=[1, 1.5, 1.5, 1.5, 1, 1, 1])
selector.constrain_range()
selected_records, records_info = selector.run(3)
selector.extract_records('选波', files=selected_records, file_SF_error=records_info)
# selector.check_database()

