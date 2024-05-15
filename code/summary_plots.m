%% Prerequisits

data_path = '../data/';

categories = {'kitchen', 'store', 'bedroom', 'livingroom', 'house', ...
       'industrial', 'stadium', 'underwater', 'tallbuilding', 'street', ...
       'highway', 'field', 'coast', 'mountain', 'forest'};

abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

num_train_per_cat = 100; 

[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);


%% Colour
vocab = build_vocabulary(train_image_paths, 150, 125, 5, "rgb");
save('vocab.mat', 'vocab');

bag_sift_col = run_experiments('bag_of_sift', "NA", 25, 2, "rgb", "NA", train_image_paths, test_image_paths);
bag_sift_col_knn = nearest_neighbor_classify (bag_sift_col{2, 1}, train_labels, bag_sift_col{3, 1}, 19, "standard", "manhattan");
bag_sift_col_svm = svm_classify(bag_sift_col{2, 1}, train_labels, bag_sift_col{3, 1}, "on", false, "doublelogit", 1000, 1);

[bag_sift_col_knn_accuracy, confusion_matrix] = calculate_confusion_matrix(bag_sift_col_knn, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_sift_col_knn_matix", sprintf("SIFT Colour KNN Confusion Matrix with %.1f%% accuracy", bag_sift_col_knn_accuracy * 100))

[bag_sift_col_svm_accuracy, confusion_matrix] = calculate_confusion_matrix(bag_sift_col_svm, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_sift_col_svm_matix", sprintf("SIFT Colour SVM Confusion Matrix with %.1f%% accuracy", bag_sift_col_svm_accuracy * 100))


pyramids_col = run_experiments('spatial pyramids', 1, "NA", false, "rgb", "NA", train_image_paths, test_image_paths);
pyramids_col_knn = nearest_neighbor_classify (pyramids_col{2, 1}, train_labels, pyramids_col{3, 1}, 13, "standard", "manhattan");
pyramids_col_svm = svm_classify(pyramids_col{2, 1}, train_labels, pyramids_col{3, 1}, "on", false, "none", 500, 10);

[pyramids_col_knn_accuracy, confusion_matrix] = calculate_confusion_matrix(pyramids_col_knn, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_pyrmd_col_knn_matix", sprintf("Pyramid Colour KNN Confusion Matrix with %.1f%% accuracy", pyramids_col_knn_accuracy * 100))
[pyramids_col_svm_accuracy, confusion_matrix] = calculate_confusion_matrix(pyramids_col_svm, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_pyrmd_col_svm_matix", sprintf("Pyramid Colour SVM Confusion Matrix with %.1f%% accuracy", pyramids_col_svm_accuracy * 100))



%% Grey
vocab = build_vocabulary(train_image_paths, 150, 125, 5, "grayscale");
save('vocab.mat', 'vocab');

bag_sift_grey = run_experiments('bag_of_sift', "NA", 25, 2, "grayscale", "NA", train_image_paths, test_image_paths);
bag_sift_grey_knn = nearest_neighbor_classify (bag_sift_grey{2, 1}, train_labels, bag_sift_grey{3, 1}, 19, "standard", "manhattan");
bag_sift_grey_svm = svm_classify(bag_sift_grey{2, 1}, train_labels, bag_sift_grey{3, 1}, "on", false, "doublelogit", 1000, 1);

[bag_sift_grey_knn_accuracy, confusion_matrix] = calculate_confusion_matrix(bag_sift_grey_knn, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_sift_grey_knn_matix", sprintf("SIFT Grey KNN Confusion Matrix with %.1f%% accuracy", bag_sift_grey_knn_accuracy * 100))

[bag_sift_grey_svm_accuracy, confusion_matrix] = calculate_confusion_matrix(bag_sift_grey_svm, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_sift_grey_svm_matix", sprintf("SIFT Grey SVM Confusion Matrix with %.1f%% accuracy", bag_sift_grey_svm_accuracy * 100))


pyramids_grey = run_experiments('spatial pyramids', 1, "NA", false, "grayscale", "NA", train_image_paths, test_image_paths);
pyramids_grey_knn = nearest_neighbor_classify (pyramids_grey{2, 1}, train_labels, pyramids_grey{3, 1}, 13, "standard", "manhattan");
pyramids_grey_svm = svm_classify(pyramids_grey{2, 1}, train_labels, pyramids_grey{3, 1}, "on", false, "none", 500, 10);

[pyramids_grey_knn_accuracy, confusion_matrix] = calculate_confusion_matrix(pyramids_grey_knn, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_pyrmd_grey_knn_matix", sprintf("Pyramid Grey KNN Confusion Matrix with %.1f%% accuracy", pyramids_grey_knn_accuracy * 100))

[pyramids_grey_svm_accuracy, confusion_matrix] = calculate_confusion_matrix(pyramids_grey_svm, test_labels, categories);
plot_confusion_matrix(confusion_matrix, categories, abbr_categories, "final_pyrmd_grey_svm_matix", sprintf("Pyramid Grey SVM Confusion Matrix with %.1f%% accuracy", pyramids_grey_svm_accuracy * 100))


%% Plot

x = [1,2,3,4,5,6,7,8];
y = [bag_sift_col_knn_accuracy, bag_sift_grey_knn_accuracy, bag_sift_col_svm_accuracy, bag_sift_grey_svm_accuracy, pyramids_col_knn_accuracy, pyramids_grey_knn_accuracy, pyramids_col_svm_accuracy, pyramids_grey_svm_accuracy];

x_labels = {"SIFT KNN"; "SIFT SVM"; "Pyramid KNN"; "Pyramid SVM"};

figure
bar(y);

newcolors = [0 0.5 1; 0.7 0.7 0.7];
colororder(newcolors)

xlabel("Feature and Classifier");
ylabel(" Accuracy %");
title("Optimum Paramter Results");

xticklabels(x_labels);

% Save graph within plots folder
filename = fullfile("../plots/", "summary_plot.png");
saveas(gcf, filename);