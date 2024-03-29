function [accuracy, confusion_matrix] = calculate_confusion_matrix ...
    (predicted_categories, test_labels, categories)

    % Code taken from create_results_webpage.m

    num_categories = length(categories);
    confusion_matrix = zeros(num_categories, num_categories);

    for j=1:length(predicted_categories)
        row = find(strcmp(test_labels{j}, categories));
        column = find(strcmp(predicted_categories{j}, categories));
        confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
    end

    num_test_per_cat = length(test_labels) / num_categories;
    confusion_matrix = confusion_matrix ./ num_test_per_cat;   
    accuracy = mean(diag(confusion_matrix));

end