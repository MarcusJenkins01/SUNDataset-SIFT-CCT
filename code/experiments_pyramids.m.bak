%% Prerequisits

data_path = '../data/';

categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

num_train_per_cat = 100; 

[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);

% Set all parameters to default values
depth_of_pyramid = 1;
color_spaces = "NA";
pyrmd_normalize = false;
interpolations = "NA";
normalize = "NA";
k_values = 1;
metrics = "euclidean";
knn_normalize = "none";

%% 1. Test affect of quantisation on spatial pyramids
depth_of_pyramid = [0 1 2 3 4];

results = run_experiments('spatial pyramids', depth_of_pyramid, "none", ...
    pyrmd_normalize, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ... 
    knn_normalize, train_labels, test_labels, categories);

plot_results('spatial pyramids', depth_of_pyramid, results, "Depth of Pyramid", ...
    "", [], false, "none", "pyrmd_quantisation_plot")

% Update parameter(s) based on best results
[depth_of_pyramid, color_spaces, pyrmd_normalize, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 2. Test affect of colour_space on spatial pyramids
% color_spaces = ["rgb" "hsv" "lab" "yiq" "ycbcr"];
% 
% results = run_experiments('spatial pyramids', depth_of_pyramid, "none", ...
%     pyrmd_normalize, color_spaces,  "none", train_image_paths, test_image_paths);
% [results, max_accuracy] = test_classifier(results, k_values, metrics, ...
%     knn_normalize, train_labels, test_labels, categories);
% 
% plot_results('spatial pyramids', color_spaces, results, "Colour Space", ...
%     "", [], true, depth_of_pyramid, "pyrmd_colour_space_plot")

% Update parameter(s) based on best results
[depth_of_pyramid, color_spaces, pyrmd_normalize, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of proportions on spatial pyramids
pyrmd_normalize = [false, true];

results = run_experiments('spatial pyramids', depth_of_pyramid, "none", ...
    pyrmd_normalize, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('spatial pyramids', pyrmd_normalize, results, "Normalization", ...
    "quant: %s", [string(depth_of_pyramid(1))], true, color_spaces, "pyrmd_proportions_plot")

% Update parameter(s) based on best results
[depth_of_pyramid, color_spaces, pyrmd_normalize, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 4. Test affect of k on classifier

color_spaces = color_spaces(1);
k_values = 1:2:40;

results = run_experiments('spatial pyramids', depth_of_pyramid, "none", ...
    pyrmd_normalize, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", k_values, results, "K Value", "quant: %s, ..." + ...
    " colour space: %s, proportion: %s", [depth_of_pyramid(1), ...
    color_spaces(1), string(normalize)], false, "none", "pyrmd_k_values_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 5. Test affect of metric on classifier

metrics = ["euclidean", "manhattan", "minkowski"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", metrics, results, "Metric", ...
    "quant: %s, colour space: %s, proportion: %s", [depth_of_pyramid(1), ...
    color_spaces(1), string(normalize)], true, k_values, "pyrmd_metrics_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 6. Test affect of normalize on classifier

knn_normalize = ["none", "standard", "min max"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", knn_normalize, results, "Normalize", ...
    "quant: %s, colour space: %s, proportion: %s", [depth_of_pyramid(1), ...
    color_spaces(1), string(normalize)], true, k_values, "pyrmd_knn_normalize_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% Create final results and confusion matrix

knn_normalize = knn_normalize(1);

results(4:end, 1) = {[]}; % Clear results varriable

predicted_categories = nearest_neighbor_classify (results{2, 1}, ... 
    train_labels, results{3, 1}, k_values, knn_normalize, metrics);      

% Calculate the accuracy of the experiment
[accuracy, confusion_matrix] = calculate_confusion_matrix ...
    (predicted_categories, test_labels, categories);

plot_confusion_matrix(confusion_matrix, categories, ... 
    abbr_categories, "pyrmd_matix", sprintf("Spatial Pyramids Confusion" + ...
    " Matrix with %.1f%% accuracy", accuracy * 100))
