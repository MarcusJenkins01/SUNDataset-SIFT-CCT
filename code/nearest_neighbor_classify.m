function [labels] = nearest_neighbor_classify(X,G,T,K,normalise,metric)
    % Use optimal if metric is not given or metric is an invalid value
    if nargin < 6
        fprintf("Metric not specified, using optimal: 'euclidean'\n");
        metric = 'euclidean';
    elseif ~any(strcmp({'euclidean', 'manhattan', 'minkowski'}, metric))
        fprintf("Invalid metric specified, using optimal: 'euclidean'\n");
        metric = 'euclidean';
    end
    
    % Optimal value for normalise if not specified
    if nargin < 5
        fprintf("normalise not specified, using default: 'none'");
        normalise = 'none';
    elseif ~any(strcmp({'none', 'standard', 'min_max', 'min max'}, normalise))
        fprintf("Invalid normalise specified, using default: 'none'\n");
        normalise = 'none';
    end

    N_x = size(X, 1);  % number of train cases
    N_t = size(T, 1);  % number of test cases

    % Encode labels from strings to integers
    class_names = unique(G);
    nc = length(class_names);
    % hashmap for constant time access
    class_encodings = containers.Map(class_names, 1:1:nc);
    g_encoded = zeros(N_t, 1);

    for i=1:N_t
        string_label = string(G(i));
        g_encoded(i) = class_encodings(string_label);
    end
    
    if isequal(normalise, 'standard')
        % Normalise the train and test cases on the same standard
        % distribution
        mu_X = sum(X) ./ N_x;  % mean of each feature in X
        % standard deviation of each feature in X
        sigma_X = sqrt(sum((X - mu_X).^2) ./ (N_x - 1));  
        X = (X - mu_X) ./ sigma_X;
        T = (T - mu_X) ./ sigma_X;
        
        % Where the standard deviation is zero we will have NaN, so replace
        % any NaN values with zero
        X(isnan(X)) = 0;
        T(isnan(T)) = 0;
    elseif isequal(normalise, 'min_max') || isequal(normalise, 'min max')
        % Min-max normalisation using the same min and max for both train
        % and test cases
        max_X = max(X(:));
        min_X = min(X(:));
        X = (X - min_X) ./ (max_X - min_X);
        T = (T - min_X) ./ (max_X - min_X);
        
        % If all values are 0 then division by zero will give us NaN, so we
        % change back to 0
        X(isnan(X)) = 0;
        T(isnan(T)) = 0;
    end

    % Calculate top k closest train instances to each test case
    top_k_labels = zeros(N_t, K);
    
    for ti=1:N_t
        t = T(ti, :);  % the current test instance we are looking at
        distances = zeros(N_x, 2);
        
        % Calculate distance of the test case to each train case
        for xi=1:N_x
            x = X(xi, :);  % the feature vector of the train instance
            g = g_encoded(xi);  % the label of the train instance
            
            % Calculate the distance based on the chosen metric
            if isequal(metric, 'euclidean')
                distance = sqrt(sum((t - x).^2));
            elseif isequal(metric, 'manhattan')
                distance = sum(abs(t - x));
            elseif isequal(metric, 'minkowski')
                distance = sum((t - x).^3).^(1/3);
            end
            
            distances(xi, :) = [g distance];
        end
        
        % Get top k sorted by our distance metric
        sorted = sortrows(distances, 2);
        top_k_labels(ti, :, :) = sorted(1:K, 1);
    end
    
    % Return majority label of the top k
    labels_top_k = mode(top_k_labels(:,:), 2);
    
    % Inverse the y label encoding from a number back to the string class
    % label
    labels = cell(N_t, 1);

    for i=1:N_t
        string_label = class_names(labels_top_k(i));
        labels(i) = string_label;
    end
end