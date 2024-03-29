function [experiments, max_accuracy] = test_classifier(experiments, ...
    k_values, metrics, knn_normalize, train_labels, test_labels, categories)

    max_accuracy = {0 0 0 0 0; 0 0 0 0 0};

    % Loop over each experiment and test value
    for i=1:size(experiments, 2)
        for k_value = k_values
            for metric = metrics
                for knn_norm = knn_normalize
                    % Run experiment based on looped parameters
                    predicted_categories = nearest_neighbor_classify ...
                        (experiments{2, i}, train_labels, ...
                        experiments{3, i}, k_value, knn_norm, metric);      
        
                    % Calculate the accuracy of the experiment
                    [accuracy, ~] = calculate_confusion_matrix ...
                        (predicted_categories, test_labels, categories);
        
                    % Get k_value index => row for result
                    index_of_k_value = find(k_values == k_value) + 3;
                    
                    % Save result in corosponding row
                    experiments{index_of_k_value, i} = ... 
                        [experiments{index_of_k_value, i}, accuracy];
        
                    % Cheack and update max_accuracy if required       
                    if accuracy > max_accuracy{1,2}
                        [max_accuracy{2, 1}, max_accuracy{2, 2}, ...
                            max_accuracy{2, 3}, max_accuracy{2, 4}, ...
                            max_accuracy{2, 5}] = deal(max_accuracy{1, 1}, ...
                            max_accuracy{1, 2}, max_accuracy{1, 3}, ...
                            max_accuracy{1, 4}, max_accuracy{1, 5});
                        [max_accuracy{1, 1}, max_accuracy{1, 2}, ...
                            max_accuracy{1, 3}, max_accuracy{1, 4}, ...
                            max_accuracy{1, 5}] = deal(i, accuracy, ...
                            k_value, metric, knn_norm);
                    elseif accuracy > max_accuracy{2,2}
                        [max_accuracy{2, 1}, max_accuracy{2, 2}, ...
                            max_accuracy{2, 3}, max_accuracy{2, 4}, ...
                            max_accuracy{2, 5}] = deal(i, accuracy, ...
                            k_value, metric, knn_norm);

                    end
    
                    fprintf(['%d. Accuracy (mean of diagonal of ' ...
                        'confusion matrix) is %.3f\n'], i, accuracy)
                end
            end

        end
    end

end