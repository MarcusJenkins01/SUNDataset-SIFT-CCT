function cropped_image = crop_image(image)     
    % Get the size of the original image
    [height, width, ~] = size(image);
       
    % If the aspect ratio is already 1 cropping will have no effect
    if height == width
        cropped_image = image;
        return;
    end
    
    % Define the desired size of the cropped region
    if height < width
        % If height is smaller, make width equal to height
        crop_size = height;
    else
        % If width is smaller or equal, make height equal to width
        crop_size = width;
    end
    
    % Calculate the coordinates of the top-left corner of the cropped region
    % Add 1 to avoid zero-based indexing
    x = max(floor((width - crop_size) / 2), 1); 
    y = max(floor((height - crop_size) / 2), 1);
    
    % Crop the image
    cropped_image = imcrop(image, [x, y, crop_size, crop_size]);
end


