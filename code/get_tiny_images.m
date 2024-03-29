function tiny_images = get_tiny_images(image_paths, ...
    preserve_aspect, color_space, size, interpolation, normalize)
    %TINYFIY_IMAGE Summary of this function goes here
    %   Detailed explanation goes here

    % Use optimal if normalize is not given or invalid value
    if nargin < 6
        fprintf("Normalize not specified, using optimal: 'false'\n");
        normalize = false;
    elseif ~any(strcmp({'true', 'false'}, string(normalize)))
        fprintf("Invalid normalize specified, using optimal: 'false'\n");
        normalize = false;
    end

    % Use optimal if interpolation is not given or invalid value
    if nargin < 5
        fprintf("Interpolation not specified, using optimal: 'bicubic'\n");
        interpolation = 'bicubic';
    elseif ~any(strcmp({'nearest', 'bilinear', 'bicubic'}, interpolation))
        fprintf("Invalid interpolation specified, using optimal: 'bicubic'\n");
        interpolation = 'bicubic';
    end

    % Use optimal if size is not given or invalid value
    if nargin < 4
        fprintf("Size not specified, using optimal: '4'\n");
        size = 4;
    elseif ~any(ismember([1, 2, 4, 8, 16, 32], size))
        fprintf("Invalid size specified, using optimal: '4'\n");
        size = 4;
    end

    % Use optimal if color_space is not given or invalid value
    if nargin < 3
        fprintf("Color_space not specified, using optimal: 'lab'\n");
        color_space = 'lab';
    elseif ~any(strcmp({'rgb', 'gray', 'hsv', 'lab', 'yiq', 'ycbcr'}, color_space))
        fprintf("Invalid color_space specified, using optimal: 'lab'\n");
        color_space = 'lab';
    end

    % Use optimal if preserve_aspect is not given or invalid value
    if nargin < 2
        fprintf("Preserve_aspect not specified, using optimal: 'true'\n");
        preserve_aspect = true;
    elseif ~any(strcmp({'true', 'false'}, string(preserve_aspect)))
        fprintf("Invalid preserve_aspect specified, using optimal: 'true'\n");
        preserve_aspect = true;
    end
    
    N = length(image_paths);
    
    if isequal(color_space, 'gray')
        d = size*size;  % each pixel is a single grayscale value
    else
        d = size*size*3;  % each pixel has three colour values
    end

    tiny_images = zeros(N, d);
    
    for i=1:N
        img = imread(image_paths{i});
        
        % If aspect ratio is to be preserved a central square crop is taken
        % of the image
        if preserve_aspect
            img = crop_image(img);
        end
        
        % Our features can either be grayscale or a features in a number of
        % colour spaces from: RGB, HSV, LAB, YIQ or YCbCr
        if isequal(color_space, 'gray')
            img = rgb2gray(img);
        elseif isequal(color_space, 'hsv')
            img = rgb2hsv(img);
        elseif isequal(color_space, 'lab')
            img = rgb2lab(img);
        elseif isequal(color_space, 'yiq')
            img = rgb2ntsc(img);
        elseif isequal(color_space, 'ycbcr')
            img = rgb2ycbcr(img);
        end
        
        % Resize and flatten image to a feature vector of dimensionality d
        img = imresize(img, [size size], interpolation);
        flat_img = double(reshape(img, [1 d]));
        
        if normalize
            % Normalise image features to zero mean and unit variance
            img_mean = sum(flat_img) ./ d;
            img_std = sqrt(sum((flat_img - img_mean).^2) ./ d);
            result = (flat_img - img_mean) ./ img_std;
            % If results is 'Not a Number' set to zero
            result(isnan(result)) = 0;
            tiny_images(i, :) = result;
        else
            tiny_images(i, :) = flat_img;
        end
        
    end
end
