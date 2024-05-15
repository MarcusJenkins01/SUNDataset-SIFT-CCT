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
size = "NA";
color_spaces = "rgb";
num_of_bins = 2;
num_of_steps = 100;
normalize = "NA";
k_values = 1;
metrics = "euclidean";
knn_normalize = "none";

% Create Vocab with optimum paramters
vocab = build_vocabulary(train_image_paths, 150, 50, 4, "rgb");
save('vocab.mat', 'vocab')


%% 3. Test affect of bin size on bag_of_sift
num_of_bins = [2 3 4 5 6];

% results = run_experiments('bag_of_sift', size, num_of_steps, ...
%     num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
% [results, max_accuracy] = test_classifier(results, k_values, metrics, ...
%     knn_normalize, train_labels, test_labels, categories);

plot_results('bag_sift_odd_split', num_of_bins, results, "Bin Size", ...
    "", [], false, color_spaces, "", "_bin_size_plot")

% Update parameter(s) based on best results
[size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of step size on bag_of_sift
num_of_steps = [25 50 75 100 125 150];
size = "NA";

results = run_experiments('bag_of_sift', size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('bag_sift_odd_split', num_of_steps, results, "Step Size", ...
    "", [], true, num_of_bins, "", "_step_size_plot")

% Update parameter(s) based on best results
[size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 2. Test affect of colour_space on bag_of_sift
color_spaces = ["grayscale", "rgb", "opponent", "rg", "lab"];
size = "NA";
num_of_bins = 2;

% results = run_experiments('bag_of_sift_col', size, num_of_steps, ...
%     num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
% [results, max_accuracy] = test_classifier(results, k_values, metrics, ...
%     knn_normalize, train_labels, test_labels, categories);

plot_results('bag_sift_split', color_spaces, results, "Colour Space", ...
    "Size of Bins: %s", [string(num_of_bins)], true, num_of_steps, "", "bag_of_sift_colour_space_plot")

% Update parameter(s) based on best results
[size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 4. Test affect of k on classifier

color_spaces = color_spaces(1);
num_of_steps = num_of_steps(1);
size = "NA";

k_values = 1:2:40;

results = run_experiments('bag_of_sift_col', size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", k_values, results, "K Value", "Colour: %s, Bin: %s, Step: %s", [ ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], false, "none", "bag_of_sift_k_values_plot", "")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 5. Test affect of metric on classifier

metrics = ["euclidean", "manhattan", "minkowski"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", metrics, results, "Metric", "Colour: %s, Bin: %s, Step: %s", [ ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], true, k_values, "bag_of_sift_metrics_plot", "")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 6. Test affect of normalize on classifier

knn_normalize = ["none", "standard", "min max"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", knn_normalize, results, "Normalise", "Colour: %s, Bin: %s, Step: %s", [ ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], true, k_values, "bag_of_sift_knn_normalize_plot", "")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% Create final results and confusion matrix

k_values = k_values(1);

results(4:end, 1) = {[]}; % Clear results varriable

predicted_categories = nearest_neighbor_classify (results{2, 1}, ... 
    train_labels, results{3, 1}, k_values, knn_normalize, metrics);      

% Calculate the accuracy of the experiment
[accuracy, confusion_matrix] = calculate_confusion_matrix ...
    (predicted_categories, test_labels, categories);

plot_confusion_matrix(confusion_matrix, categories, ... 
    abbr_categories, "bag_of_sift_matix", sprintf("Build Bag of Sift Confusion" + ...
    " Matrix with %.1f%% accuracy", accuracy * 100))
