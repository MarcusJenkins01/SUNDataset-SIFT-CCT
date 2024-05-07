
depths = [0 1 2 3 4];

for L = depths
    temp = zeros(1, L);
    for l = 0:L
        if l == 0
            weight = 1/2.^L;
        else
            weight = 1/2.^(L-l+1);
        end
        temp(1, l+1) = weight;
    end
    disp(temp);
end
