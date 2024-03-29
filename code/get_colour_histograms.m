function output = get_colour_histograms(image_paths, quantisation, ...
    color_space, proportion)

    valid_color_spaces = {'rgb', 'hsv', 'lab', 'yiq', 'ycbcr'};

    % Use optimal proportion if not given or is invalid
    if nargin < 4
        fprintf("'proportion' not specified, using optimal: false\n");
        proportion = false;
    elseif ~any(strcmp({'true', 'false'}, string(proportion)))
        fprintf("Invalid proportion specified, using optimal: 'false'\n");
        proportion = false;
    end
    
    % Use optimal color_space if not given or is invalid
    if nargin < 3
        fprintf("Colour space not specified, using optimal: 'rgb'\n");
        color_space = 'rgb';
    elseif ~any(strcmp(valid_color_spaces, color_space))
        fprintf("Invalid colour space specified, using optimal: 'rgb'\n");
        color_space = 'rgb';
    end

    % Use optimal quantisation if not given or is invalid
    if nargin < 2
        fprintf("Quantisation not specified, using optimal value: 6\n");
        quantisation = 6;
    elseif ~any(strcmp(valid_color_spaces, color_space))
        fprintf("Invalid quantisation specified, using optimal value: 6\n");
        quantisation = 6;
    end

    % All of the colour spaces we have chosen to trial have 3 channels
    num_channels = 3;
    
    % The feature vector is the histogram for each colour combination
    % concatenated
    output = zeros(numel(image_paths), quantisation.^num_channels);
    
    % Iterate through each image in the dataset
    for index = 1:numel(image_paths)
        img = imread(image_paths{index});
        img = double(img);
        
        % Convert the input image to the chosen colour space
        if isequal(color_space, 'hsv')
            img = rgb2hsv(img);
        elseif isequal(color_space, 'lab')
            img = rgb2lab(img);
        elseif isequal(color_space, 'yiq')
            img = rgb2ntsc(img);
        elseif isequal(color_space, 'ycbcr')
            img = rgb2ycbcr(img);
        end
        
        % Transform each channel to 0 to 1 for quantisation
        for chan = 1:3
            ch_min = min(img(:, :, chan));
            ch_max = max(img(:, :, chan));
            img(:, :, chan) = (img(:, :, chan) - ch_min) ./ (ch_max - ch_min);

            % Replace NaNs with 0 if any of the channels have all 0 values
            img(isnan(img)) = 0;
        end
        
        % Apply quantisation
        img = round(img*(quantisation-1)) + 1;
        
        % Generate the histogram for this image
        hh = zeros(quantisation, quantisation, quantisation);
        
        img_h = size(img, 1);
        img_w = size(img, 2);
        for i = 1:img_h  % along the height
            for j = 1:img_w  % along the width
                chan1 = img(i,j,1);
                chan2 = img(i,j,2);
                chan3 = img(i,j,3);
    
                hh(chan1, chan2, chan3) = hh(chan1, chan2, chan3) + 1;
            end
        end
        
        % Flatten the histogram into a 1d feature vector
        hh_flat = reshape(hh, 1, []);
        
        % Divide the histogram by the number of pixels
        if proportion
            hh_flat = hh_flat ./ (img_h * img_w);
        end

        output(index, :) = hh_flat;
    end

end

