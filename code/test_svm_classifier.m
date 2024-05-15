function [experiments, max_accuracy] = test_svm_classifier(experiments, crossVals, normalises, transform_funcs, box_constraints, kernal_scales, train_labels, test_labels, categories)

    max_accuracy = {0 0 0 0 0 0 0; 0 0 0 0 0 0 0};

    % Loop over each experiment and test value
    for i=1:size(experiments, 2)
        for crossVal = crossVals
            for normalise = normalises
                for transform_func = transform_funcs
                    for box_constraint = box_constraints
                        for kernal_scale = kernal_scales
                            tic
                            % Run experiment based on looped parameters
                            predicted_categories = svm_classify(experiments{2, i}, train_labels, experiments{3, i}, crossVal, normalise, transform_func, box_constraint, kernal_scale);      
                
                            % Calculate the accuracy of the experiment
                            [accuracy, ~] = calculate_confusion_matrix(predicted_categories, test_labels, categories);
                
                            % Get k_value index => row for result
                            % index_of_k_value = find(k_values == k_value) + 3;
                            
                            % Save result in corosponding row
                            experiments{i+3, i} = [experiments{i+3, i}, accuracy];
                
                            % Cheack and update max_accuracy if required       
                            if accuracy > max_accuracy{1,2}
                                [max_accuracy{2, 1}, max_accuracy{2, 2}, ...
                                    max_accuracy{2, 3}, max_accuracy{2, 4}, ...
                                    max_accuracy{2, 5}, max_accuracy{2, 6}, max_accuracy{2, 7}] = deal(max_accuracy{1, 1}, ...
                                    max_accuracy{1, 2}, max_accuracy{1, 3}, ...
                                    max_accuracy{1, 4}, max_accuracy{1, 5}, max_accuracy{1, 6}, max_accuracy{1, 7});
                                [max_accuracy{1, 1}, max_accuracy{1, 2}, ...
                                    max_accuracy{1, 3}, max_accuracy{1, 4}, ...
                                    max_accuracy{1, 5}, max_accuracy{1, 6}, max_accuracy{1, 7}] = deal(i, accuracy, ...
                                    crossVal, normalise, transform_func, box_constraint, kernal_scale);
                            elseif accuracy > max_accuracy{2,2}
                                [max_accuracy{2, 1}, max_accuracy{2, 2}, ...
                                    max_accuracy{2, 3}, max_accuracy{2, 4}, ...
                                    max_accuracy{2, 5}, max_accuracy{2, 6}, max_accuracy{2, 7}] = deal(i, accuracy, ...
                                    crossVal, normalise, transform_func, box_constraint, kernal_scale);
        
                            end
            
                            fprintf(['%d. Accuracy (mean of diagonal of ' ...
                                'confusion matrix) is %.3f\n'], i, accuracy)
                            toc
                        end
                    end
                end
            end

        end
    end

end