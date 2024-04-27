
img = imread('peppers.png');

[height, width, depth] = size(img);

depth_of_pyramid = 2;


for i = 0 : depth_of_pyramid
    tiles = 2^i;
    new_height = height / tiles;
    new_width = width / tiles;
    rows = ones(1, tiles) * new_height;
    cols = ones(1, tiles) * new_width;

    R_image_tiles = mat2cell(img(:,:,1), rows, cols);
    G_image_tiles = mat2cell(img(:,:,2), rows, cols);
    B_image_tiles = mat2cell(img(:,:,3), rows, cols);

    image_tiles = cell(tiles, tiles);

    [row_num, col_num] = size(image_tiles);

    figure
    img_num = 1;
    
    total = 0;

    for row = 1:row_num
        for col = 1:col_num

            image_tiles{row, col} = cat(3, R_image_tiles{row, col}, G_image_tiles{row, col}, B_image_tiles{row, col});

            temp_img = image_tiles{row, col};

            SIFT_features = get_dsift_features(temp_img, 4, 4, 'lab');
            
            subplot(row_num, col_num, img_num);
            imshow(temp_img)
            temp_sum = sum(SIFT_features(:));
            total = total + temp_sum;
            if row_num * col_num > 1
                title(num2str(temp_sum));
            end
            img_num = img_num+1;
        end
    end

    sgtitle(num2str(total));

end
