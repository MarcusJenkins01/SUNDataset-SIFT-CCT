function image_feats = get_spatial_pyramids(image_paths, depth_of_pyramid, normalize, color_space)

    % Use optimal normalize if not given or is invalid
    if nargin < 3
        fprintf("'normalize' not specified, using optimal: false\n");
        normalize = false;
    elseif ~any(strcmp({'true', 'false'}, string(normalize)))
        fprintf("Invalid normalize specified, using optimal: 'false'\n");
        normalize = false;
    end

    % Use optimal depth of pyramid if not given or is invalid
    if nargin < 2
        fprintf("Depth of pyramid not specified, using optimal value: 2\n");
        depth_of_pyramid = 2;
    end


    % Load our dictionary of visual words
    load('vocab.mat', 'vocab');
    
    % Store vocabulary size
    vocab_size = size(vocab, 2);

    % Number of images
    N = length(image_paths);

    % Number of features
    feats = sum(2.^(0:depth_of_pyramid).^2 * vocab_size);

    % Output to store each images features
    image_feats = zeros(N, feats);
   
    % Set weighting for each level
    weighting = calculate_weight(depth_of_pyramid);
    
    % Iterate through each image
    for i=1:N

        % Load the image
        img = imread(image_paths{i});
        
        % Get height and width from image size
        [height, width, ~] = size(img);   

        % Use our get_dsift_features function to get the SIFT
        % descriptors and their location for the image
        [SIFT_features, locations] = get_dsift_features(img, 2, 2, color_space);
        
        % Compute the distance matrix of each DSIFT descriptor
        % to each visual word in the vocabulary
        D = vl_alldist2(single(SIFT_features), vocab);
    
        % Find the closest visual word in the vocabulary 
        % for each descriptor
        [~, closest_visual_words] = min(D, [], 2);
      
        % Overall histogram for image at all levels
        hist_all = [];
        
        % Loop over each level
        for j = 0 : depth_of_pyramid
            % Calculat the number of tiles per image
            tiles = 2^(j);

            % Calculate height and width of a tile
            % Use round to ensure integer value
            new_height = height / tiles;
            new_width = width / tiles;

            % Resize image to match any height or width rounding
            % img = imresize(img, [new_height*tiles, new_width*tiles]);

            % Create the histogram for tile
            hist_tile = zeros(tiles^2, vocab_size);

            % For each closest visual word
            for desc_i=1:size(closest_visual_words, 1)
                % Current closest word
                closest_i = closest_visual_words(desc_i);

                % x and y coordinates of word
                x_val = locations(1, desc_i);
                y_val = locations(2, desc_i);

                % Calculate tile location based off integer division
                % add one to use a 1 based index
                tile_row = fix(x_val / new_width) + 1;
                tile_col = fix(y_val / new_height) + 1;

                % Find 1D index from row, column coordinates
                tile = (tile_col-1) * tiles + tile_row;

                if tile > tiles^2
                    disp(tile);
                end

                % Increment the count for the relevent tile
                hist_tile(tile, closest_i) = hist_tile(tile, closest_i) + 1;
            end
       
            % Average image histogram using weight
            hist_tile = hist_tile * weighting(j+1);

            % Flatten tile histograms into 1D list
            hist_tile = reshape(hist_tile, 1, []);

            if normalize
                % L1 normalise the histogram (so the sum is 1)
                hist_tile = hist_tile ./ norm(hist_tile, 1);
            end

            % Concatinate image histogram to overall historgram
            hist_all = cat(2, hist_all, hist_tile);
                
        end

        % Add the overall historgam to the image features output
        image_feats(i,:) = hist_all(:);
        
    end
    
end
