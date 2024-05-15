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
vocab_size = 150;
color_spaces = "lab";
num_of_bins = 2;
num_of_steps = 100;
normalize = "NA";
k_values = 1;
metrics = "euclidean";
knn_normalize = "none";

%% 1. Test affect of vocabulary size on build vocab
vocab_size = [50 100 150 200 250 300];

results = run_experiments('build_vocab', vocab_size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ... 
    knn_normalize, train_labels, test_labels, categories);

plot_results('Build Vocab', vocab_size, results, "Vocabulary Size", ...
    "", [], false, "none", "vocab_size_plot")

% Update parameter(s) based on best results
[vocab_size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 2. Test affect of colour_space on build vocab
color_spaces = ["grayscale", "rgb", "opponent", "rg", "lab"];
% color_spaces = ["rgb", "lab"];


results = run_experiments('build_vocab', vocab_size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('Build Vocab', color_spaces, results, "Colour Space", ...
    "", [], true, vocab_size, "vocab_colour_space_plot")

% Update parameter(s) based on best results
[vocab_size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of bin size on build vocab
num_of_bins = [2 3 4 5 6];
vocab_size = vocab_size(1);

results = run_experiments('build_vocab', vocab_size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('build_vocab_odd_split', num_of_bins, results, "Bin Size", ...
    "Vocab size: %s", [string(vocab_size(1))], true, color_spaces, "vocab_bin_size_plot")

% Update parameter(s) based on best results
[vocab_size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 3. Test affect of step size on build vocab
num_of_steps = [25 50 75 100 125 150];

results = run_experiments('build_vocab', vocab_size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results('build_vocab_odd_split', num_of_steps, results, "Step Size", ...
    "Size: %s, Colour: %s", [string(vocab_size(1)), color_spaces(1)], true, num_of_bins, "vocab_step_size_plot")

% Update parameter(s) based on best results
[vocab_size, color_spaces, num_of_bins, num_of_steps, normalize] = ...
    update_feature_params(results, max_accuracy);

%% 4. Test affect of k on classifier

num_of_steps = num_of_steps(1);
k_values = 1:2:40;

results = run_experiments('build_vocab', vocab_size, num_of_steps, ...
    num_of_bins, color_spaces,  "none", train_image_paths, test_image_paths);
[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", k_values, results, "K Value", "Size: %s, Colour: %s, Bin: %s, Step: %s", [vocab_size(1), ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], false, "none", "vocab_k_values_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 5. Test affect of metric on classifier

metrics = ["euclidean", "manhattan", "minkowski"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", metrics, results, "Metric", "Size: %s, Colour: %s, Bin: %s, Step: %s", [vocab_size(1), ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], true, k_values, "vocab_metrics_plot")

% Update parameter(s) based on best results
[k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy);

%% 6. Test affect of normalize on classifier

knn_normalize = ["none", "standard", "min max"];

results(4:end, 1) = {[]}; % Clear results varriable

[results, max_accuracy] = test_classifier(results, k_values, metrics, ...
    knn_normalize, train_labels, test_labels, categories);

plot_results("classifier", knn_normalize, results, "Normalize", "Size: %s, Colour: %s, Bin: %s, Step: %s", [vocab_size(1), ...
    color_spaces(1), string(num_of_bins), string(num_of_steps)], true, k_values, "vocab_knn_normalize_plot")

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
    abbr_categories, "vocab_matix", sprintf("Build Vocab Confusion" + ...
    " Matrix with %.1f%% accuracy", accuracy * 100))
