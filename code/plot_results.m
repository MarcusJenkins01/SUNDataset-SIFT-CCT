function plot_results(test_type, test_parameter, results, x_label, ...
    subtitle_text, subtitle_values, dual_plot, comparison_parameter, filename)
    x = 1:length(test_parameter);
    
    figure
    % Check if a dual plot is needed
    if dual_plot
        % Check if testing type as results returned in different formats
        if test_type == "classifier"
            y1 = results(4, 1);
            y2 = results(5, 1);
            y = [y1{1}, y2{1}];

            % If second result empty, split first result in half
            if isempty(y2{1})
                mid_point = length(y1{1}) / 2;
                y2{1} = y1{1}(mid_point + 1:end);
                y1{1} = y1{1}(1:mid_point);
            end
    
            plot(x, y1{1}, 'LineWidth', 2);
            hold on
    
            plot(x, y2{1}, 'LineWidth', 2);
            hold off

            % Find x-axis index of max accuracy
            [~, max_idx] = max(y);

        else
            % Calculate split posistion and split results for dual plots
            split_size = length([results{4, :}]) / 2;
            y1 = [results{4, 1:split_size}];
            y2 = [results{4, split_size + 1:end}];
            y = [results{4, :}];
            
            plot(x, y1, 'LineWidth', 2);
            hold on
            
            plot(x, y2, 'LineWidth', 2);
            hold off

            % Find x-axis index of max accuracy
            [~, max_idx] = max(y);
    
            % Update x-axis index if outside of range
            if max_idx > split_size
                max_idx = max_idx - split_size;
            end
        end
        
        % Add legend when dual plots
        legend(string(comparison_parameter), 'Location', 'best');

    else
        if test_type == "classifier"
            % Convert cells to array
            y = cell2mat(results(4:end, 1));
        else
            y = [results{4, :}];
        end
            plot(x, y, 'LineWidth', 2);

            % Find x-axis index of max accuracy
            [~, max_idx] = max(y);
    end
    
    % Plot settings
    xticks(x);
    xlim([min(x), max(x)]);
    xticklabels(string(test_parameter));
    grid on; 
    xlabel(x_label);
    ylabel('Accuracy %');
    subtitle = sprintf(subtitle_text, subtitle_values);
    main_title = sprintf("Affect of %s parameter on %s", x_label, test_type);
    title({main_title, subtitle});
       
    % Add vertical dashed line for maximum accuracy
    xline(max_idx, 'Color', 'r', 'LineStyle', '--', 'HandleVisibility', 'off');
    
    % Add label to the vertical line
    label_text = sprintf('%.1f%%', max(y)*100);
    text(max_idx*1.05, max(y), label_text, 'Color', 'r');
      
    % Check filename has correct file extension
    if ~endsWith(filename, '.png')
        filename = strcat(filename, ".png");
    end

    % Save graph within plots folder
    filename = fullfile("../plots/", filename);
    saveas(gcf, filename);

end
