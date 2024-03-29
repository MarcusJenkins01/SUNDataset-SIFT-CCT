function [k_values, metrics, knn_normalize] = update_classifier_params(max_accuracy)
    % Update parameter(s) based on best results
    k_values = max_accuracy{1,3};
    metrics = max_accuracy{1, 4};
    knn_normalize = max_accuracy{1, 5};
    
    % Only add second parameter(s) if unique
    if ~ismember(k_values, max_accuracy{2,3})
        k_values = [k_values, max_accuracy{2,3}];
    end
    if ~ismember(metrics, max_accuracy{2,4})
        metrics = [metrics, max_accuracy{2,4}];
    end
    if ~ismember(knn_normalize, max_accuracy{2,5})
        knn_normalize = [knn_normalize, max_accuracy{2,5}];
    end

end