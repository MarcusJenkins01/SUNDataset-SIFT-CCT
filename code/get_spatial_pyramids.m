function image_feats = get_spatial_pyramids(image_paths, step, ...
    bin_size, color_space)

    % Load our dictionary of visual words
    load('vocab.mat', 'vocab');
    
    % Store vocabulary size
    vocab_size = size(vocab, 2);

    % Number of images
    N = length(image_paths);

    % Output to store each images features
    image_feats = zeros(N, 1050);

    % Set max depth of pyramid
    depth_of_pyramid = 3;
    
    % Set weighting for each level
    weighting = [0.25, 0.25, 0.5];
    
    % Iterate through each image
    for i=1:N

        % Load the image
        img = imread(image_paths{i});
        
        % Get height and width from image size
        [height, width, ~] = size(img);   
        
        % Overall histogram for image at all levels
        hist_all = [];
        
        % Loop over each level
        for j = 1 : depth_of_pyramid
            % Calculat the number of tiles per image
            tiles = 2^(j-1);

            % Calculate height and width of a tile
            % Use round to ensure integer value
            new_height = round(height / tiles);
            new_width = round(width / tiles);

            % Resize image to match any height or width rounding
            img = imresize(img, [new_height*tiles, new_width*tiles]);

            % Create array of length of tiles with each value being either
            % the tiles height or width
            rows = ones(1, tiles) * new_height;
            cols = ones(1, tiles) * new_width;
        
            % Create tiles for each colour dimension
            R_image_tiles = mat2cell(img(:,:,1), rows, cols);
            G_image_tiles = mat2cell(img(:,:,2), rows, cols);
            B_image_tiles = mat2cell(img(:,:,3), rows, cols);
        
            % Create cells to hold all split image tiles
            image_tiles = cell(tiles, tiles);
        
            % Store number of rows and columns from size of tiled image
            [row_num, col_num] = size(image_tiles);
        
            % figure
            % img_num = 1;
            % 
            % total1 = 0;
            % total2 = 0;
            % total3 = 0;
        
            % Create bag of words histogram for overall tiled image
            hist_image = [];
        
            % Loop over each tile in the tiled image
            for row = 1:row_num
                for col = 1:col_num
        
                    % Concatenate the colour dimensions and store within
                    % relevent tile location
                    image_tiles{row, col} = ... 
                        cat(3, R_image_tiles{row, col}, ...
                        G_image_tiles{row, col}, B_image_tiles{row, col});
        
                    % Select tile as temporary image
                    temp_img = image_tiles{row, col};
                    
                    % Use our get_dsift_features function to get the SIFT
                    % descriptors for each tile, based on passed parameters
                    SIFT_features = get_dsift_features(temp_img, step, ...
                        bin_size, color_space);
                    
                    % Compute the distance matrix of each DSIFT descriptor
                    % to each visual word in the vocabulary
                    D = vl_alldist2(single(SIFT_features), vocab);
                
                    % Find the closest visual word in the vocabulary 
                    % for each descriptor
                    [~, closest_visual_words] = min(D, [], 2);
                    
                    % Create the histogram for tile
                    hist_tile = zeros(1, vocab_size);
                
                    % For each closest visual word
                    for desc_i=1:size(closest_visual_words, 1)
                        closest_i = closest_visual_words(desc_i);
                
                        % Increment the count
                        hist_tile(closest_i) = hist_tile(closest_i) + 1;
                    end
        
                    % Concatenate the tile histogram to the overall image
                    % histogram
                    hist_image = cat(2, hist_image, hist_tile);
                    
                    % subplot(row_num, col_num, img_num);
                    % imshow(temp_img)
                    % temp_sum1 = hist_tile(1);
                    % temp_sum2 = hist_tile(2);
                    % temp_sum3 = hist_tile(3);
                    % total1 = total1 + temp_sum1;
                    % total2 = total2 + temp_sum2;
                    % total3 = total3 + temp_sum3;
                    % if row_num * col_num > 1
                    %     title("1: " + num2str(temp_sum1) + ", 2: " + num2str(temp_sum2) + ", 3: " + num2str(temp_sum3));
                    % end
                    % img_num = img_num+1;
                end
            end
        
            % Average image histogram using weight
            hist_image = hist_image * weighting(j);

            % Concatinate image histogram to overall historgram
            hist_all = cat(2, hist_all, hist_image);
        
            % sgtitle("1: " + num2str(total1) + ", 2: " + num2str(total2) + ", 3: " + num2str(total3));
        
        end

        % Add the overall historgam to the image features output
        image_feats(i,:) = hist_all(:);
        
    end
    
end
