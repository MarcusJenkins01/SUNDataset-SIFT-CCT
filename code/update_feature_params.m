function [quantisation, color_spaces, proportions, interpolations, normalize] ...
    = update_feature_params(results, max_accuracy)

    % Extract two best accuracies
    best = results{1, (max_accuracy{1, 1})}.split(',');
    best = strtrim(best);
    second = results{1, (max_accuracy{2, 1})}.split(',');
    second = strtrim(second);

    % Update parameter(s) based on best results
    if best(1) == "build_vocab" || best(1) == "bag_of_sift" || best(1) == "bag_of_sift_col"
        proportions = [str2double(best(2))];
    else
        proportions = [strcmp(best(2), 'true')];
    end
    color_spaces = [best(3)];
    quantisation = [str2double(best(4))];
    if best(1) == "build_vocab" || best(1) == "bag_of_sift" || best(1) == "bag_of_sift_col"
        interpolations = [str2double(best(5))];
    else
        interpolations = [best(5)];
    end
    normalize = [strcmp(best(6), 'true')];
    
    % Only add second parameter(s) if unique
    if (second(1) == "build_vocab" || second(1) == "bag_of_sift" || second(1) == "bag_of_sift_col") && ~ismember(proportions, str2double(second(2)))
        proportions = [proportions, str2double(second(2))];
    elseif (second(1) ~= "build_vocab" && second(1) ~= "bag_of_sift" && second(1) ~= "bag_of_sift_col") && ~ismember(proportions, strcmp(second(2), 'true'))
        proportions = [proportions, strcmp(second(2), 'true')];
    end

    if ~ismember(color_spaces, second(3))
        color_spaces = [color_spaces, second(3)];
    end
    if ~ismember(quantisation, str2double(second(4)))
        quantisation = [quantisation, str2double(second(4))];
    end
    if (second(1) == "build_vocab" || second(1) == "bag_of_sift" || second(1) == "bag_of_sift_col") && ~ismember(interpolations, str2double(second(5)))
        interpolations = [interpolations, str2double(second(5))];
    elseif (second(1) ~= "build_vocab" && second(1) ~= "bag_of_sift" && second(1) ~= "bag_of_sift_col") && ~ismember(interpolations, second(5))
        interpolations = [interpolations, second(5)];
    end
    if ~ismember(normalize, strcmp(second(6), 'true'))
        normalize = [normalize, strcmp(second(6), 'true')];
    end

end
