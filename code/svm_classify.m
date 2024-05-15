% Based on James Hays, Brown University 

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, crossVal, normalise, transform_func, box_constraint, kernal_scale)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.


% Create unique list of categories
categories=unique(train_labels);

% Number of unique categories
num_categories=length(categories);

% Empty cell for each categories model
SVMModels=cell(num_categories,1);

% Train for each category
for idx = 1:num_categories
    % Create temporary binary class labels
    temp_labels=strcmp(train_labels,categories(idx)); % Create binary classes for each classifier
    % Save model by category index
    SVMModels{idx}=fitcsvm(train_image_feats,temp_labels,'Standardize',normalise,...
        'KernelFunction','linear','ScoreTransform',transform_func,'Crossval',crossVal,'BoxConstraint',box_constraint,'KernelScale',kernal_scale);
end

% Size of test set
num_test=size(test_image_feats,1);

% Default predicted scores (number of tests x number of categories)
Scores=zeros(num_test,num_categories);

% Test for each category
for idx=1:num_categories
    if strcmpi(crossVal, 'on')       
        % Make predictions on test data
        [~, score] = kfoldPredict(SVMModels{idx});
    else
        % Predict and record score only
        [~,score]=predict(SVMModels{idx},test_image_feats);
    end
    % Save score by category index
    % Second column contains positive-class scores
    Scores(:,idx)=score(:,2);
end

% Find the maximum score and record the indice
[~,indices]=max(Scores,[],2);

% Match the indices to the corosponding category index
predicted_categories=categories(indices);
