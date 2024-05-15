function experiments_svm(results, filename)


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

    % results = run_experiments('colour histogram', 6, "NA", false, "rgb",  "NA", train_image_paths, test_image_paths);
    % filename = "col_hist_svm_";

    % full test ranges
    all_crossVals = ["off", "on"];
    all_normalise = [false, true];
    all_transform_func = ["none", "doublelogit", "invlogit", "ismax", "logit", "sign"];
    all_box_constraints = [1, 10, 50, 100, 500, 1000];
    all_kernal_scales = [1, 10, 50, 100, 500, 1000];
    
    % Set all parameters to default values
    crossVals = all_crossVals(1);
    normalise = all_normalise(1);
    transform_func = all_transform_func(1);
    box_constraints = all_box_constraints(1);
    kernal_scales = all_kernal_scales(1);
    
    %% 1. Test affect of kernal size on classifier
    
    kernal_scales = all_kernal_scales;

    [results, max_accuracy] = test_svm_classifier(results, crossVals, normalise, transform_func, box_constraints, kernal_scales, train_labels, test_labels, categories);

    filename_end = ("_kernal_scales_plot");

    plot_results("classifier", kernal_scales, results, "Kernal Scales", "", [], false, "none", filename, filename_end)

    % Update parameter(s) based on best results
    [crossVals, normalise, transform_func, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy);
    
    %% 2. Test affect of box constraint on classifier
    
    box_constraints = all_box_constraints;
    
    results(4:end, 1) = {[]}; % Clear results varriable

    [results, max_accuracy] = test_svm_classifier(results, crossVals, normalise, transform_func, box_constraints, kernal_scales, train_labels, test_labels, categories);

    filename_end = ("_box_const_plot");

    plot_results("SVM", box_constraints, results, "Box Constraits", "", [], true, kernal_scales, filename, filename_end)
    
    % Update parameter(s) based on best results
    [crossVals, normalise, transform_func, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy);
    
    %% 3. Test affect of transofrm function on classifier
    
    transform_func = all_transform_func;
    
    results(4:end, 1) = {[]}; % Clear results varriable
    
    [results, max_accuracy] = test_svm_classifier(results, crossVals, normalise, transform_func, box_constraints, kernal_scales, train_labels, test_labels, categories);
    
    filename_end = ("_trans_func_plot");

    plot_results("SVM", transform_func, results, "Transform Function", "Kernal Scale: %s", [string(kernal_scales(1))], true, box_constraints, filename, filename_end)
    
    % Update parameter(s) based on best results
    [crossVals, normalise, transform_func, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy);

    %% 4. Test affect of normalize on classifier
    
    normalise = all_normalise;

    results(4:end, 1) = {[]}; % Clear results varriable

    [results, max_accuracy] = test_svm_classifier(results, crossVals, normalise, transform_func, box_constraints, kernal_scales, train_labels, test_labels, categories);
    
    filename_end = ("_norm_plot");

    plot_results("SVM", normalise, results, "Normalise", "Kernal Scale: %s, Box Const: %s", [string(kernal_scales(1)), string(box_constraints(1))], true, transform_func, filename, filename_end)
    
    % Update parameter(s) based on best results
    [crossVals, normalise, transform_func, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy);

    %% 5. Test affect of cross validation on classifier
    
    crossVals = all_crossVals;
    
    results(4:end, 1) = {[]}; % Clear results varriable
    
    [results, max_accuracy] = test_svm_classifier(results, crossVals, normalise, transform_func, box_constraints, kernal_scales, train_labels, test_labels, categories);
    
    filename_end = ("_cross_val_plot");

    plot_results("SVM", crossVals, results, "Cross Validation", "Kernal Scale: %s, Box Const: %s Normalise %s", [string(kernal_scales(1)), string(box_constraints(1)), string(normalise)], true, transform_func, filename, filename_end)
    
    % Update parameter(s) based on best results
    [crossVals, normalise, transform_func, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy);
    
    %% Create final results and confusion matrix
        
    crossVals = crossVals(1);
    transform_func = transform_func(1);
        
    predicted_categories = svm_classify(results{2, 1}, train_labels, results{3, 1}, crossVals, normalise, transform_func, box_constraints, kernal_scales);      
    
    % Calculate the accuracy of the experiment
    [accuracy, confusion_matrix] = calculate_confusion_matrix ...
        (predicted_categories, test_labels, categories);
    
    plot_confusion_matrix(confusion_matrix, categories, ... 
        abbr_categories, "vocab_matix", sprintf("%s Confusion" + ...
        " Matrix with %.1f%% accuracy", filename, accuracy * 100))
