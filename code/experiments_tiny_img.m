1%% Prerequisits

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
sizes = 8;
color_spaces = "rgb";
preserve_aspect = false;
interpolations = "bicubic";
normalize = false;
k_values = 1;
metrics = "euclidean";
knn_normalize = "none";

%% 1. Test affect of dimension on image resizing
sizes = 2.^(0:5);

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('tiny image', sizes, results, "Dimension", ...
    "", [], false, "none", "dimensions_plot")

% Update parameter(s) based on best results
[sizes, color_spaces, preserve_aspect, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 2. Test affect of interpolation on image resizing
interpolations = ["nearest" "bilinear" "bicubic"];

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('tiny image', interpolations, results, "Interpolation", ...
    "", [], true, sizes, "interpolations_plot")

% Update parameter(s) based on best results
[sizes, color_spaces, preserve_aspect, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of colour_space on image resizing
color_spaces = ["rgb" "gray" "hsv" "lab" "yiq" "ycbcr"];

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('tiny image', color_spaces, results, "Colour Space", ...
    "size: %s", [string(sizes(1))], true, interpolations, "colour_space_plot")

% Update parameter(s) based on best results
[sizes, color_spaces, preserve_aspect, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 4. Test affect of preserve_aspect on image resizing
preserve_aspect = [false, true];

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('tiny image', preserve_aspect, results, "Preserve Aspect", ...
    "size: %s, colour space: %s", [sizes(1), color_spaces(1)], true, ...
    interpolations, "preserve_aspect_plot")

% Update parameter(s) based on best results
[sizes, color_spaces, preserve_aspect, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 5. Test affect of normalize on image resizing
normalize = [false, true];

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('tiny image', normalize, results, "Normalize", ...
    "size: %s, interpolation: %s, colour space: %s", [sizes(1), ...
    interpolations(1), color_spaces(1)], true, preserve_aspect, "normalize_plot")

% Update parameter(s) based on best results
[sizes, color_spaces, preserve_aspect, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 6. Test affect of k on classifier

preserve_aspect = preserve_aspect(1);
k_values = 1:2:40;

results = run_experiments('tiny image', sizes, interpolations, ...
    preserve_aspect, color_spaces, normalize, train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", k_values, results, "K Value", "size: %s, " + ...
    "interpolation: %s, colour space: %s, aspect: %s, normalize: %s", ...
    [sizes(1), interpolations(1), color_spaces(1), ...
    string(preserve_aspect(1)), string(normalize(1))], false, "none", "k_values_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 7. Test affect of metric on classifier

metrics = ["euclidean", "manhattan", "minkowski"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", metrics, results, "Metric", "size: %s, " + ...
    "interpolation: %s, colour space: %s, aspect: %s, normalize: %s", ...
    [sizes(1), interpolations(1), color_spaces(1), ...
    string(preserve_aspect(1)), string(normalize(1))], true, k_values, "metrics_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 8. Test affect of normalize on classifier

knn_normalize = ["none", "standard", "min max"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", knn_normalize, results, "Normalize KNN", "size: %s, " + ...
    "interpolation: %s, colour space: %s, aspect: %s, normalize: %s", ...
    [sizes(1), interpolations(1), color_spaces(1), ...
    string(preserve_aspect(1)), string(normalize(1))], true, metrics, "knn_normalize_plot")

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
    abbr_categories, "tiny_image_matix", sprintf("Tiny Image Confusion" + ...
    " Matrix with %.1f%% accuracy", accuracy * 100))