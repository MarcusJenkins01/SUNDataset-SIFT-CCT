function weights = calculate_weight(depth) 

    weights = zeros(1, depth);

    for i = 0:depth
        if i == 0
            weight = 1/2.^depth;
        else
            weight = 1/2.^(depth-i+1);
        end
        weights(1, i+1) = weight;
    end

end