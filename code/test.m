data_path = '../data/';

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

num_train_per_cat = 100; 

[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);

% results = run_experiments('colour histogram', 6, "NA", false, "rgb",  "NA", train_image_paths, test_image_paths);
results = run_experiments('spatial pyramids', 1, "NA", false, "rgb",  "NA", train_image_paths, test_image_paths);
% results = run_experiments('bag_of_sift_col', 150, 25, 2, "rgb",  "none", train_image_paths, test_image_paths);

experiments_svm(results, "Spatial SVM")