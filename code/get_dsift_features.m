function SIFT_features = get_dsift_features(img, step, bin_size, color_space)
    % This function provides a modular method of computing the DSIFT
    % features for a given colour space
    
    % The default colour space will be grayscale, if not specified
    if nargin < 4
        color_space = 'grayscale';
    end

    % Convert image to chosen colour space
    if isequal(color_space, 'grayscale')
        img = rgb2gray(img);
    elseif isequal(color_space, 'opponent')
        R  = img(:, :, 1);
        G  = img(:, :, 2);
        B  = img(:, :, 3);
        img(:, :, 1) = (R - G) ./ sqrt(2);
        img(:, :, 2) = (R + G - 2*B) ./ sqrt(6);
        img(:, :, 3) = (R + G + B) ./ sqrt(3);
    elseif isequal(color_space, 'opponent_w')
        R  = img(:, :, 1);
        G  = img(:, :, 2);
        B  = img(:, :, 3);
        O1 = (R - G) ./ sqrt(2);
        O2 = (R + G - 2*B) ./ sqrt(6);
        O3 = (R + G + B) ./ sqrt(3);

        % Normalise the O1 and O2 colour channels by the luminance, O3 
        O1 = O1 ./ O3;
        O2 = O2 ./ O3;

        img = cat(3, O1, O2);
    elseif isequal(color_space, 'rg')
        R  = img(:, :, 1);
        G  = img(:, :, 2);
        B  = img(:, :, 3);
        r = R ./ (R + G + B);
        g = G ./ (R + G + B);
        img = cat(3, r, g);
    end
    
    % Calculate number of colour components (channels)
    if ndims(img) == 1
        n_channels = 1;
    else
        n_channels = size(img, 3);
    end

    % Build a vocabulary using each colour component/channel
    SIFT_features_out = [];

    for ch=1:n_channels
        % Retrieve and convert channel to single precision (for VLFeat)
        channel = single(img(:, :, ch));
        
        % Compute dense SIFT descriptors for the current channel
        [~, SIFT_features_channel] = vl_dsift(channel, 'Step', step, 'Size', bin_size);
        
        % Concatenate the channel SIFT descriptors to form the full
        % descriptors for the image
        if isempty(SIFT_features_out)
            SIFT_features_out = SIFT_features_channel;
        else
            SIFT_features_out = cat(1, SIFT_features_out, SIFT_features_channel);
        end
    end

    SIFT_features = SIFT_features_out;
end