function plot_confusion_matrix(confusion_matrix, categories, ... 
    abbr_categories, filename, title_text)

    % Code adapted from create_results_webpage.m
    fig_handle = figure; 
    imagesc(confusion_matrix, [0 1]); 
    set(fig_handle, 'Color', [.988, .988, .988])
    axis_handle = get(fig_handle, 'CurrentAxes');
    set(axis_handle, 'XTick', 1:15)
    set(axis_handle, 'XTickLabel', abbr_categories)
    set(axis_handle, 'YTick', 1:15)
    set(axis_handle, 'YTickLabel', categories)
    xlabel('Abbr. Categories');
    ylabel('Categories');
    title(title_text);
    colorbar;

    % Check filename has correct file extension
    if ~endsWith(filename, '.png')
        filename = strcat(filename, ".png");
    end

    % Save graph within plots folder
    filename = fullfile("../plots/", filename);
    saveas(gcf, filename);

end