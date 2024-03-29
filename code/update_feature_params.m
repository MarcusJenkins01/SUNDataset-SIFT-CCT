function [quantisation, color_spaces, proportions, interpolations, normalize] ...
    = update_feature_params(results, max_accuracy)

    % Extract two best accuracies
    best = results{1, (max_accuracy{1, 1})}.split(',');
    best = strtrim(best);
    second = results{1, (max_accuracy{2, 1})}.split(',');
    second = strtrim(second);

    % Update parameter(s) based on best results
    proportions = [strcmp(best(2), 'true')];
    color_spaces = [best(3)];
    quantisation = [str2double(best(4))];
    interpolations = [best(5)];
    normalize = [strcmp(best(6), 'true')];
    
    % Only add second parameter(s) if unique
    if ~ismember(proportions, strcmp(second(2), 'true'))
        proportions = [proportions, strcmp(second(2), 'true')];
    end
    if ~ismember(color_spaces, second(3))
        color_spaces = [color_spaces, second(3)];
    end
    if ~ismember(quantisation, str2double(second(4)))
        quantisation = [quantisation, str2double(second(4))];
    end
    if ~ismember(interpolations, second(5))
        interpolations = [interpolations, second(5)];
    end
    if ~ismember(normalize, strcmp(second(6), 'true'))
        normalize = [normalize, strcmp(second(6), 'true')];
    end

end
