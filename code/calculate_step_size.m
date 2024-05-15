function step_size = calculate_step_size(value_number, bin_size)

    window_size = bin_size * 4;

    switch value_number
        case {25, 0.25}
            step_size = window_size * 0.25;
        case {50, 0.5}
            step_size = window_size * 0.50;
        case {75, 0.75}
            step_size = window_size * 0.75;
        case {100, 1}
            step_size = window_size;
        case {125, 1.25}
            step_size = window_size * 1.25;
        case {150, 1.5}
            step_size = window_size * 1.5;
        otherwise
            step_size = 1;

    end
