function [crossVals, normalises, transform_funcs, box_constraints, kernal_scales] = update_svm_classifier_params(max_accuracy)
    % Update parameter(s) based on best results
    crossVals = max_accuracy{1,3};
    normalises = max_accuracy{1, 4};
    transform_funcs = max_accuracy{1, 5};
    box_constraints = max_accuracy{1, 6};
    kernal_scales = max_accuracy{1, 7};
    
    % Only add second parameter(s) if unique
    if ~ismember(crossVals, max_accuracy{2,3})
        crossVals = [crossVals, max_accuracy{2,3}];
    end
    if ~ismember(normalises, max_accuracy{2,4})
        normalises = [normalises, max_accuracy{2,4}];
    end
    if ~ismember(transform_funcs, max_accuracy{2,5})
        transform_funcs = [transform_funcs, max_accuracy{2,5}];
    end
    if ~ismember(box_constraints, max_accuracy{2,6})
        box_constraints = [box_constraints, max_accuracy{2,6}];
    end
    if ~ismember(kernal_scales, max_accuracy{2,7})
        kernal_scales = [kernal_scales, max_accuracy{2,7}];
    end

end