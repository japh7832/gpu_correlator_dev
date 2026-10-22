% timing_plot.m
% Plot GPU correlator loop timing from all timing_*MHz.csv files.
%
% Run this script from the folder that contains the CSV files.
%
% Each CSV has per-reporting-interval rows; relevant columns:
%   elapsed_s        – wall-clock time since run start
%   block_budget_ms  – constant target cycle time (M / sample_rate)
%   total_mean_ms    – mean total loop time over the last 100 blocks
%
% For each sample rate the script draws:
%   solid line   – total_mean_ms vs elapsed_s
%   dotted line  – block_budget_ms (constant horizontal reference)

% ── Find and sort files ───────────────────────────────────────────────────────
files = dir('timing_*MHz.csv');
if isempty(files)
    error('No timing_*MHz.csv files found in the current directory.');
end

nFiles = numel(files);
sr = zeros(1, nFiles);
for k = 1:nFiles
    tok = regexp(files(k).name, 'timing_(\d+)MHz\.csv', 'tokens');
    if ~isempty(tok)
        sr(k) = str2double(tok{1}{1});
    else
        warning('Could not parse sample rate from "%s"; skipping.', files(k).name);
        sr(k) = NaN;
    end
end

% Remove any entries that didn't parse
valid = ~isnan(sr);
files = files(valid);
sr    = sr(valid);
nFiles = numel(files);

% Sort ascending by sample rate
[sr, ord] = sort(sr);
files = files(ord);

% ── Colors ────────────────────────────────────────────────────────────────────
% lines(N) gives perceptually distinct colours (cycles after 7,
% but contrast is still adequate for 10 series at typical display sizes).
cmap = lines(nFiles);

% ── Plot ──────────────────────────────────────────────────────────────────────
figure('Name', 'GPU Correlator Timing', 'NumberTitle', 'off', ...
       'Position', [100 100 900 540]);
hold on;

hLines = gobjects(nFiles, 1);   % one handle per series for the legend

for k = 1:nFiles
    T = readtable(files(k).name);

    elapsed   = T.elapsed_s;
    totalMean = T.total_mean_ms;
    budget    = T.block_budget_ms(1);   % constant within a run

    col = cmap(k, :);

    % Solid: total mean execution time
    hLines(k) = plot(elapsed, totalMean, '-', ...
        'Color', col, 'LineWidth', 1.8);

    % Dotted: block budget (excluded from legend via HandleVisibility)
    plot([elapsed(1), elapsed(end)], [budget, budget], ':', ...
        'Color', col, 'LineWidth', 1.8, 'HandleVisibility', 'off');
end

hold off;

% ── Formatting ────────────────────────────────────────────────────────────────
grid on;
box on;

xlabel('Elapsed time  (s)',                  'FontSize', 12);
ylabel('Execution time  (ms)',               'FontSize', 12);
title( 'GPU Correlator Loop Timing by Sample Rate', 'FontSize', 14, 'FontWeight', 'bold');

legendLabels = arrayfun(@(x) sprintf('%d MHz', x), sr, 'UniformOutput', false);
legend(hLines, legendLabels, 'Location', 'best', 'FontSize', 10);

% Subtitle explaining line styles
subtitle('solid = total\_mean\_ms   ·   dotted = block budget', ...
         'FontSize', 9, 'Color', [0.45 0.45 0.45]);
