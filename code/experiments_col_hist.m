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
quantisation = 1;
color_spaces = "rgb";
proportions = false;
interpolations = "NA";
normalize = "NA";
k_values = 1;
metrics = "euclidean";
knn_normalize = "none";

%% 1. Test affect of quantisation on colour histogram
quantisation = [4 6 8 12 16];

results = run_experiments('colour histogram', quantisation, "none", ...
    proportions, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ... 
    knn_normalize, train_labels, test_labels, categories);

plot_results('colour histogram', quantisation, results, "Quantisation", ...
    "", [], false, "none", "hist_quantisation_plot")

% Update parameter(s) based on best results
[quantisation, color_spaces, proportions, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 2. Test affect of colour_space on colour histogram
color_spaces = ["rgb" "hsv" "lab" "yiq" "ycbcr"];

results = run_experiments('colour histogram', quantisation, "none", ...
    proportions, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('colour histogram', color_spaces, results, "Colour Space", ...
    "", [], true, quantisation, "hist_colour_space_plot")

% Update parameter(s) based on best results
[quantisation, color_spaces, proportions, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of proportions on colour histogram
proportions = [false, true];

results = run_experiments('colour histogram', quantisation, "none", ...
    proportions, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('colour histogram', proportions, results, "Proportion", ...
    "quant: %s", [string(quantisation(1))], true, color_spaces, "hist_proportions_plot")

% Update parameter(s) based on best results
[quantisation, color_spaces, proportions, interpolations, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 4. Test affect of k on classifier

color_spaces = color_spaces(1);
k_values = 1:2:40;

results = run_experiments('colour histogram', quantisation, "none", ...
    proportions, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", k_values, results, "K Value", "quant: %s, ..." + ...
    " colour space: %s, proportion: %s", [quantisation(1), ...
    color_spaces(1), string(proportions)], false, "none", "hist_k_values_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 5. Test affect of metric on classifier

metrics = ["euclidean", "manhattan", "minkowski"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", metrics, results, "Metric", ...
    "quant: %s, colour space: %s, proportion: %s", [quantisation(1), ...
    color_spaces(1), string(proportions)], true, k_values, "hist_metrics_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 6. Test affect of normalize on classifier

knn_normalize = ["none", "standard", "min max"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", knn_normalize, results, "Normalize", ...
    "quant: %s, colour space: %s, proportion: %s", [quantisation(1), ...
    color_spaces(1), string(proportions)], true, k_values, "hist_knn_normalize_plot")

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
    abbr_categories, "hist_matix", sprintf("Colour Histogram Confusion" + ...
    " Matrix with %.1f%% accuracy", accuracy * 100))
