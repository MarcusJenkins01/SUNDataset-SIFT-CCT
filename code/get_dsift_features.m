function [SIFT_features, locations] = get_dsift_features(img, step, bin_size, color_space, color_hist)
    % This function provides a modular method of computing the DSIFT
    % features for a given colour space
    
    quantisation = 6;  % optimal value established in last coursework

    if nargin < 5
        color_hist = false;
    end

    % The default colour space will be grayscale, if not specified
    if nargin < 4
        color_space = 'grayscale';
    end

    % Convert image to chosen colour space
    if ~color_hist
        if isequal(color_space, 'grayscale')
            img = rgb2gray(img);
        elseif isequal(color_space, 'opponent')
            R  = img(:, :, 1);
            G  = img(:, :, 2);
            B  = img(:, :, 3);
    
            % img will be O1, O2, O3 rather than RGB
            img(:, :, 1) = (R - G) ./ sqrt(2);
            img(:, :, 2) = (R + G - 2*B) ./ sqrt(6);
            img(:, :, 3) = (R + G + B) ./ sqrt(3);
        elseif isequal(color_space, 'rg')
            R  = img(:, :, 1);
            G  = img(:, :, 2);
            B  = img(:, :, 3);
    
            % rg colour space are the R and G components normalised using the
            % luminance; B is not needed separately since B is factored into 
            % the denominator of r and g
            r = R ./ (R + G + B);
            g = G ./ (R + G + B);
    
            img = cat(3, r, g);
        elseif isequal(color_space, 'lab')
            img = rgb2lab(img);
        end
    end
    
    % Calculate number of colour components (channels)
    if ndims(img) == 1
        n_channels = 1;
    else
        n_channels = size(img, 3);
    end

    SIFT_features_out = [];
    
    if color_hist
        img_gray = single(rgb2gray(img));
        [locations, SIFT_features_img] = vl_dsift(img_gray, 'Step', ...
                step, 'Size', bin_size, 'Fast');

        % Extend the array to accept a colour histogram
        sift_descriptor_size = size(SIFT_features_img, 1);
        SIFT_features_img(sift_descriptor_size + quantisation.^3, :) = 0;
        
        img_h = size(img, 1);
        img_w = size(img, 2);

        for loc_i=1:size(locations, 1)
            x = locations(loc_i, 1);
            y = locations(loc_i, 2);

            x_min = x - bin_size .* 2 + 1;
            x_max = min(x + bin_size .* 2, img_w);
            y_min = y - bin_size .* 2 + 1;
            y_max = min(y + bin_size .* 2, img_h);
            
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
            img_window = img(y_min:y_max, x_min:x_max, :);
            window_h = size(img_window, 1);
            window_w = size(img_window, 2);
            for i = 1:window_h  % along the height
                for j = 1:window_w  % along the width
                    chan1 = img_window(i, j, 1);
                    chan2 = img_window(i, j, 2);
                    chan3 = img_window(i, j, 3);
        
                    hh(chan1, chan2, chan3) = hh(chan1, chan2, chan3) + 1;
                end
            end
            
            % Flatten the histogram into a 1d feature vector
            hh_flat = reshape(hh, 1, []);
            SIFT_features_img(sift_descriptor_size+1:end, loc_i) = hh_flat(:);
        end

        SIFT_features_out = SIFT_features_img;
    else
        % Build a vocabulary using each colour component/channel
        for ch=1:n_channels
            % Retrieve and convert channel to single precision (for VLFeat)
            channel = single(img(:, :, ch));
            
            % Compute dense SIFT descriptors for the current channel
            [locations, SIFT_features_channel] = vl_dsift(channel, 'Step', ...
                step, 'Size', bin_size, 'Fast');

            % Concatenate the channel SIFT descriptors to form the full
            % descriptors for the image
            if isempty(SIFT_features_out)
                SIFT_features_out = SIFT_features_channel;
            else
                SIFT_features_out = cat(1, SIFT_features_out, SIFT_features_channel);
            end
        end
    end

    SIFT_features = SIFT_features_out;
end