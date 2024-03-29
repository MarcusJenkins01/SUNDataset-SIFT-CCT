function experiments = run_experiments(FEATURE, sizes, interpolations, ...
    preserve_aspects, color_spaces, normalize, train_image_paths, test_image_paths)
    % Value of number of experiments to run based on length of passed parameters
    number_of_experiments = length(sizes) * length(interpolations) * ...
        length(preserve_aspects) * length(color_spaces) * length(normalize);
    % Initialise cell to store results
    experiments = cell(24, number_of_experiments);
    % Initialise starting experiment number
    experiment_number = 1;

    % Loop through all passed parameters
    for size = sizes
        for interpolation = interpolations
            for preserve_aspect = preserve_aspects
                for color_space = color_spaces
                    for norm = normalize
                        disp(experiment_number);

                        switch lower(FEATURE)    
                            case 'tiny image'
                                % Get training and tests features for this experiment
                                train_image_feats = get_tiny_images ...
                                    (train_image_paths, preserve_aspect, ...
                                    color_space, size, interpolation, norm);
                                test_image_feats  = get_tiny_images ...
                                    (test_image_paths, preserve_aspect, ...
                                    color_space, size, interpolation, norm);
                            case 'colour histogram'
                                % Get training and tests features for this experiment
                                train_image_feats = get_colour_histograms ...
                                    (train_image_paths, size, ...
                                    color_space, preserve_aspect);
                                test_image_feats  = get_colour_histograms ...
                                    (test_image_paths, size, ... 
                                    color_space, preserve_aspect);
                        end

                        % Store experiment results in corosponding column
                        % Rows are used to store multiple data points for each
                        % experiment
                        experiments{1, experiment_number} = FEATURE + ...
                            ", " + preserve_aspect + ", " + color_space ...
                            + ", " + size + ", " + interpolation + ", " + norm;
                        experiments{2, experiment_number} = train_image_feats;
                        experiments{3, experiment_number} = test_image_feats;
    
                        % Increase experiment number
                        experiment_number = experiment_number + 1;
                    end
                end
            end
        end
    end
end