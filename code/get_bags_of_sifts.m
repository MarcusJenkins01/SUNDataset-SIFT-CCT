% Implementated according to the starter code prepared by James Hays, Brown
% University 
% Michal Mackiewicz, UEA

function image_feats = get_bags_of_sifts(image_paths, step, bin_size, color_space)
    % image_paths is an N x 1 cell array of strings where each string is an
    % image path on the file system.
    
    % This function assumes that 'vocab.mat' exists and contains an N x 128
    % matrix 'vocab' where each row is a kmeans centroid or a visual word. This
    % matrix is saved to disk rather than passed in a parameter to avoid
    % recomputing the vocabulary every time at significant expense.
    
    % image_feats is an N x d matrix, where d is the dimensionality of the
    % feature representation. In this case, d will equal the number of clusters
    % or equivalently the number of entries in each image's histogram.
    
    % You will want to construct SIFT features here in the same way you
    % did in build_vocabulary.m (except for possibly changing the sampling
    % rate) and then assign each local feature to its nearest cluster center
    % and build a histogram indicating how many times each cluster was used.
    % Don't forget to normalize the histogram, or else a larger image with more
    % SIFT features will look very different from a smaller version of the same
    % image.
    
    % Default colour space will be grayscale
    if nargin < 4
        color_space = 'grayscale';
    end
    
    % Load our dictionary of visual words
    load('vocab.mat');
    
    vocab_size = size(vocab, 2);
    N = length(image_paths);
    image_feats_out = zeros(N, vocab_size);
    
    % Iterate through each image
    for i=1:N
        img_path = image_paths{i};
        
        % Load the image
        img = imread(img_path);
        
        % Use our get_dsift_features function to get the SIFT descriptors
        % for a given set of parameters
        SIFT_features = get_dsift_features(img, step, bin_size, color_space);
        
        % Compute the distance matrix of each DSIFT descriptor to each
        % visual word in the vocabulary
        D = vl_alldist2(single(SIFT_features), vocab);

        % Find the closest visual word in the vocabulary for each descriptor
        [~, closest_visual_words] = min(D, [], 2);
        
        % Create the bag-of-visual-words histogram
        bovw_hist = zeros(1, vocab_size);

        for desc_i=1:size(closest_visual_words, 1)
            closest_i = closest_visual_words(desc_i);

            % Increment the count of the closest visual word
            bovw_hist(closest_i) = bovw_hist(closest_i) + 1;
        end

        % Normalise the histogram
        bovw_hist = normalize(bovw_hist);
        
        % Assign the feature vector (histogram) to the output matrix
        image_feats_out(i, :) = bovw_hist(:);
    end

    image_feats = image_feats_out;

    %{
    Useful functions:
    [locations, SIFT_features] = vl_dsift(img) 
     http://www.vlfeat.org/matlab/vl_dsift.html
     locations is a 2 x n list list of locations, which can be used for extra
      credit if you are constructing a "spatial pyramid".
     SIFT_features is a 128 x N matrix of SIFT features
      note: there are step, bin size, and smoothing parameters you can
      manipulate for vl_dsift(). We recommend debugging with the 'fast'
      parameter. This approximate version of SIFT is about 20 times faster to
      compute. Also, be sure not to use the default value of step size. It will
      be very slow and you'll see relatively little performance gain from
      extremely dense sampling. You are welcome to use your own SIFT feature
      code! It will probably be slower, though.
    
    D = vl_alldist2(X,Y) 
       http://www.vlfeat.org/matlab/vl_alldist2.html
        returns the pairwise distance matrix D of the columns of X and Y. 
        D(i,j) = sum (X(:,i) - Y(:,j)).^2
        Note that vl_feat represents points as columns vs this code (and Matlab
        in general) represents points as rows. So you probably want to use the
        transpose operator '  You can use this to figure out the closest
        cluster center for every SIFT feature. You could easily code this
        yourself, but vl_alldist2 tends to be much faster.
    %}
end