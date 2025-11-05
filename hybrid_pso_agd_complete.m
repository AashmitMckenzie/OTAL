%Hybrid PSO + Adaptive Gradient Descent (AGD)

function hybrid_pso_agd_complete()
    clc; clear; close all;
    
    DIM = 30;
    NUM_PARTICLES = 30;
    MAX_ITER = 500;
    RUNS = 30;

    W_MAX = 0.9;
    W_MIN = 0.4;

    C1 = 1.5; %Social constant
    C2 = 1.5; %Cognitive Constant
    VMAX_RATIO = 0.2; %For stabalization
    TOP_K_FRAC = 0.2; % 20% best particles get refinement for each itteration (exploration)
    ETA_0 = 0.1;
    ALPHA = 0.01; %both for learning rate formula

    GRAD_EPS = 1e-5;
    USE_DYNAMIC_COEFF = true;
    C1_START = 2.5;
    C1_END = 0.5; % decays (self-reliance decreases)
    C2_START = 0.5; 
    C2_END = 2.5; % grows (social learning increases)

    DIVERSITY_THRESHOLD = 1e-8; %If normalized positional variance falls below 1e-8, the algorithm reinitializes the worst 10% particles to escape collapse/premature convergence.
    REINIT_FRACTION = 0.1;
    VELOCITY_TRIGGER = false; %were true, AGD would only apply when a particle's velocity norm is tiny (i.e., stuck)
    STAGNATION_TRIGGER = false; % were true, AGD would kick in only after no global improvement for STAGNATION_WINDOW iterations
    STAGNATION_WINDOW = 10;
    
    functions = get_benchmark_functions();
    num_funcs = length(functions);
    all_results = cell(num_funcs, 1);
    
    fprintf(' HYBRID PSO-AGD OPTIMIZATION FRAMEWORK - MATLAB IMPLEMENTATION\n');
    fprintf('\nConfiguration:\n');
    fprintf('  Dimensions: %d\n', DIM);
    fprintf('  Particles: %d\n', NUM_PARTICLES);
    fprintf('  Max Iterations: %d\n', MAX_ITER);
    fprintf('  Independent Runs: %d\n', RUNS);
    fprintf('  Functions: %d\n', num_funcs);
    
    for f = 1:num_funcs
        func_struct = functions{f};
        func_name = func_struct.name;
        func_handle = func_struct.func;
        bounds = func_struct.bounds;
        
        fprintf('Function: %s\n', func_name);
        fprintf('Domain: [%.2f, %.2f], Optimum: %.2f\n', bounds(1), bounds(2), func_struct.optimum);
        
        fprintf('\nRunning Standard PSO (%d runs)...\n', RUNS);
        pso_histories = zeros(RUNS, MAX_ITER + 1);
        pso_final_fitness = zeros(RUNS, 1);
        pso_times = zeros(RUNS, 1);
        
        for run = 1:RUNS
            fprintf('  Run %2d/%d - Standard PSO on %s', run, RUNS, func_name);
            tic;
            [~, best_fit, history] = standard_pso(func_handle, DIM, bounds, NUM_PARTICLES, MAX_ITER, W_MAX, W_MIN, C1, C2, VMAX_RATIO, run);
            elapsed = toc;
            pso_histories(run, :) = history;
            pso_final_fitness(run) = best_fit;
            pso_times(run) = elapsed;
            fprintf(' -> Best: %.6e, Time: %.3fs\n', best_fit, elapsed);
        end
        
        fprintf('\nRunning Hybrid PSO-AGD (%d runs)...\n', RUNS);
        hybrid_histories = zeros(RUNS, MAX_ITER + 1);
        hybrid_final_fitness = zeros(RUNS, 1);
        hybrid_times = zeros(RUNS, 1);
        
        for run = 1:RUNS
            fprintf('  Run %2d/%d - Hybrid PSO-AGD on %s', run, RUNS, func_name);
            tic;
            [~, best_fit, history] = hybrid_pso_agd(func_handle, DIM, bounds, NUM_PARTICLES, MAX_ITER, W_MAX, W_MIN, C1, C2, VMAX_RATIO, TOP_K_FRAC, ETA_0, ALPHA, GRAD_EPS, USE_DYNAMIC_COEFF, C1_START, C1_END, C2_START, C2_END, DIVERSITY_THRESHOLD, REINIT_FRACTION, VELOCITY_TRIGGER, STAGNATION_TRIGGER, STAGNATION_WINDOW, run);
            elapsed = toc;
            hybrid_histories(run, :) = history;
            hybrid_final_fitness(run) = best_fit;
            hybrid_times(run) = elapsed;
            fprintf(' -> Best: %.6e, Time: %.3fs\n', best_fit, elapsed);
        end
        
        pso_stats.best = min(pso_final_fitness);
        pso_stats.mean = mean(pso_final_fitness);
        pso_stats.std = std(pso_final_fitness);
        pso_stats.median = median(pso_final_fitness);
        pso_stats.avg_time = mean(pso_times);
        
        hybrid_stats.best = min(hybrid_final_fitness);
        hybrid_stats.mean = mean(hybrid_final_fitness);
        hybrid_stats.std = std(hybrid_final_fitness);
        hybrid_stats.median = median(hybrid_final_fitness);
        hybrid_stats.avg_time = mean(hybrid_times);
        
        fprintf('\n----------------------------------------------------------------------\n');
        fprintf('Results for %s:\n', func_name);
        fprintf('----------------------------------------------------------------------\n');
        fprintf('%-20s %20s %20s\n', 'Metric', 'Standard PSO', 'Hybrid PSO-AGD');
        fprintf('----------------------------------------------------------------------\n');
        fprintf('%-20s %20.6e %20.6e\n', 'Best', pso_stats.best, hybrid_stats.best);
        fprintf('%-20s %20.6e %20.6e\n', 'Mean', pso_stats.mean, hybrid_stats.mean);
        fprintf('%-20s %20.6e %20.6e\n', 'Std', pso_stats.std, hybrid_stats.std);
        fprintf('%-20s %20.6e %20.6e\n', 'Median', pso_stats.median, hybrid_stats.median);
        fprintf('%-20s %20.3f %20.3f\n', 'Avg Time (s)', pso_stats.avg_time, hybrid_stats.avg_time);
        
        if pso_stats.mean > 0
            improvement = (pso_stats.mean - hybrid_stats.mean) / pso_stats.mean * 100;
            fprintf('\nImprovement in mean fitness: %+.2f%%\n', improvement);
        end
        
        all_results{f}.name = func_name;
        all_results{f}.pso_stats = pso_stats;
        all_results{f}.hybrid_stats = hybrid_stats;
        all_results{f}.pso_histories = pso_histories;
        all_results{f}.hybrid_histories = hybrid_histories;
        
        plot_convergence_comparison(pso_histories, hybrid_histories, func_name);
    end
    
    plot_summary_comparison(all_results);
    
    fprintf(' FINAL SUMMARY - HYBRID PSO-AGD vs STANDARD PSO\n');
    
    improvements = zeros(num_funcs, 1);
    for f = 1:num_funcs
        pso_mean = all_results{f}.pso_stats.mean;
        hybrid_mean = all_results{f}.hybrid_stats.mean;
        if pso_mean > 0
            improvements(f) = (pso_mean - hybrid_mean) / pso_mean * 100;
        end
        status = '✓';
        if improvements(f) <= 0
            status = '✗';
        end
        fprintf('%s %-15s: %+7.2f%% improvement\n', status, all_results{f}.name, improvements(f));
    end
    
    avg_improvement = mean(improvements);
    functions_improved = sum(improvements > 0);
    
    fprintf('\n----------------------------------------------------------------------\n');
    fprintf('Average improvement: %+.2f%%\n', avg_improvement);
    fprintf('Functions improved: %d/%d\n', functions_improved, num_funcs);
    fprintf('----------------------------------------------------------------------\n');
    fprintf('\nEXPERIMENTS COMPLETED SUCCESSFULLY!\n');
end

% Standard PSO 
function [best_pos, best_fit, history] = standard_pso(func, dim, bounds, num_particles, max_iter, w_max, w_min, c1, c2, vmax_ratio, seed)
    rng(seed);
    lower = bounds(1);
    upper = bounds(2);
    vmax = vmax_ratio * (upper - lower);
    
    positions = lower + (upper - lower) * rand(num_particles, dim);
    velocities = -vmax + 2 * vmax * rand(num_particles, dim);
    
    fitness = zeros(num_particles, 1);

    % population initailization
    for i = 1:num_particles
        fitness(i) = func(positions(i, :));
    end
    
    pbest_positions = positions;
    pbest_fitness = fitness;
    
    [gbest_fitness, best_idx] = min(pbest_fitness);
    gbest_position = pbest_positions(best_idx, :);
    
    history = zeros(1, max_iter + 1);
    history(1) = gbest_fitness;
    
    for t = 1:max_iter
        w = w_max - (w_max - w_min) * t / max_iter;
        
        r1 = rand(num_particles, dim);
        r2 = rand(num_particles, dim);
        
        cognitive = c1 * r1 .* (pbest_positions - positions);
        social = c2 * r2 .* (repmat(gbest_position, num_particles, 1) - positions);
        
        velocities = w * velocities + cognitive + social;
        velocities = max(min(velocities, vmax), -vmax);
        
        positions = positions + velocities;
        positions = max(min(positions, upper), lower);
        
        for i = 1:num_particles
            fitness(i) = func(positions(i, :));
        end
        
        improved = fitness < pbest_fitness;
        pbest_positions(improved, :) = positions(improved, :);
        pbest_fitness(improved) = fitness(improved);
        
        [min_fitness, best_idx] = min(pbest_fitness);
        if min_fitness < gbest_fitness
            gbest_fitness = min_fitness;
            gbest_position = pbest_positions(best_idx, :);
        end
        
        history(t + 1) = gbest_fitness;
    end
    
    best_pos = gbest_position;
    best_fit = gbest_fitness;
end

%hybrid PSO
function [best_pos, best_fit, history] = hybrid_pso_agd(func, dim, bounds, num_particles, max_iter, w_max, w_min, c1_base, c2_base, vmax_ratio, top_k_frac, eta_0, alpha, grad_eps, use_dynamic_coeff, c1_start, c1_end, c2_start, c2_end, diversity_threshold, reinit_fraction, velocity_trigger, stagnation_trigger, stagnation_window, seed)
    rng(seed);
    lower = bounds(1);
    upper = bounds(2);
    vmax = vmax_ratio * (upper - lower);
    top_k = max(1, floor(top_k_frac * num_particles));
    
    positions = lower + (upper - lower) * rand(num_particles, dim);
    velocities = -vmax + 2 * vmax * rand(num_particles, dim);
    
    fitness = zeros(num_particles, 1);
    for i = 1:num_particles
        fitness(i) = func(positions(i, :));
    end
    
    pbest_positions = positions;
    pbest_fitness = fitness;
    
    [gbest_fitness, best_idx] = min(pbest_fitness);
    gbest_position = pbest_positions(best_idx, :);
    
    history = zeros(1, max_iter + 1);
    history(1) = gbest_fitness;
    stagnation_counter = 0;
    
    for t = 1:max_iter
        w = w_max - (w_max - w_min) * t / max_iter;
        
        if use_dynamic_coeff
            c1 = c1_start + (c1_end - c1_start) * t / max_iter;
            c2 = c2_start + (c2_end - c2_start) * t / max_iter;
        else
            c1 = c1_base;
            c2 = c2_base;
        end
        
        r1 = rand(num_particles, dim);
        r2 = rand(num_particles, dim);
        
        cognitive = c1 * r1 .* (pbest_positions - positions);
        social = c2 * r2 .* (repmat(gbest_position, num_particles, 1) - positions);
        
        velocities = w * velocities + cognitive + social;
        velocities = max(min(velocities, vmax), -vmax);
        
        positions = positions + velocities;
        positions = max(min(positions, upper), lower);
        
        for i = 1:num_particles
            fitness(i) = func(positions(i, :));
        end
        
        improved = fitness < pbest_fitness;
        pbest_positions(improved, :) = positions(improved, :);
        pbest_fitness(improved) = fitness(improved);
        
        [min_fitness, best_idx] = min(pbest_fitness);
        if min_fitness < gbest_fitness
            gbest_fitness = min_fitness;
            gbest_position = pbest_positions(best_idx, :);
            stagnation_counter = 0;
        else
            stagnation_counter = stagnation_counter + 1;
        end
        
        apply_agd = true;
        if stagnation_trigger
            apply_agd = stagnation_counter >= stagnation_window;
        end
        
        if apply_agd
            [~, top_indices] = sort(pbest_fitness);
            top_indices = top_indices(1:top_k);
            
            eta_t = eta_0 / (1 + alpha * t);
            
            for idx = 1:length(top_indices)
                i = top_indices(idx);
                
                if velocity_trigger
                    vel_norm = norm(velocities(i, :));
                    if vel_norm >= 0.01
                        continue;
                    end
                end
                
                x_current = positions(i, :);
                grad = compute_gradient(func, x_current, grad_eps);
                x_new = x_current - eta_t * grad;
                x_new = max(min(x_new, upper), lower);
                
                fitness_new = func(x_new);
                
                if fitness_new < fitness(i)
                    positions(i, :) = x_new;
                    fitness(i) = fitness_new;
                    
                    if fitness_new < pbest_fitness(i)
                        pbest_positions(i, :) = x_new;
                        pbest_fitness(i) = fitness_new;
                        
                        if fitness_new < gbest_fitness
                            gbest_position = x_new;
                            gbest_fitness = fitness_new;
                            stagnation_counter = 0;
                        end
                    end
                end
            end
        end
        
        diversity = compute_diversity(positions, lower, upper);
        if diversity < diversity_threshold % If too small, the swarm collapsed; reinitialize the worst PBests
            n_reinit = max(1, floor(reinit_fraction * num_particles));
            [~, worst_indices] = sort(pbest_fitness, 'descend');
            worst_indices = worst_indices(1:n_reinit);
            
            for idx = 1:length(worst_indices) % Randomly re-seed those particles (both positions & velocities) & Update PBests where the random re-seed happened to improve them
                i = worst_indices(idx);
                positions(i, :) = lower + (upper - lower) * rand(1, dim);
                velocities(i, :) = -vmax + 2 * vmax * rand(1, dim);
                fitness(i) = func(positions(i, :));
                
                if fitness(i) < pbest_fitness(i)
                    pbest_positions(i, :) = positions(i, :);
                    pbest_fitness(i) = fitness(i);
                end
            end
        end
        
        history(t + 1) = gbest_fitness;
    end
    
    best_pos = gbest_position;
    best_fit = gbest_fitness;
end

function grad = compute_gradient(func, x, eps)
    dim = length(x);
    grad = zeros(1, dim);
    
    for j = 1:dim
        x_plus = x;
        x_minus = x;
        x_plus(j) = x_plus(j) + eps;
        x_minus(j) = x_minus(j) - eps;
        grad(j) = (func(x_plus) - func(x_minus)) / (2 * eps);
    end
end

function diversity = compute_diversity(positions, lower, upper)
    variances = var(positions, 0, 1);
    mean_variance = mean(variances);
    domain_range = upper - lower;
    diversity = mean_variance / (domain_range^2);
end

function functions = get_benchmark_functions()
    functions = cell(10, 1);
    
    functions{1}.name = 'Sphere';
    functions{1}.bounds = [-5.12, 5.12];
    functions{1}.optimum = 0;
    functions{1}.func = @(x) sum(x.^2);
    
    functions{2}.name = 'SumSquares';
    functions{2}.bounds = [-10, 10];
    functions{2}.optimum = 0;
    functions{2}.func = @(x) sum((1:length(x)) .* x.^2);
    
    functions{3}.name = 'Rosenbrock';
    functions{3}.bounds = [-5, 10];
    functions{3}.optimum = 0;
    functions{3}.func = @(x) sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (x(1:end-1) - 1).^2);
    
    functions{4}.name = 'Schwefel222';
    functions{4}.bounds = [-10, 10];
    functions{4}.optimum = 0;
    functions{4}.func = @(x) sum(abs(x)) + prod(abs(x));
    
    functions{5}.name = 'Zakharov';
    functions{5}.bounds = [-5, 5];
    functions{5}.optimum = 0;
    functions{5}.func = @(x) sum(x.^2) + sum(0.5 * (1:length(x)) .* x).^2 + sum(0.5 * (1:length(x)) .* x).^4;
    
    functions{6}.name = 'Rastrigin';
    functions{6}.bounds = [-5.12, 5.12];
    functions{6}.optimum = 0;
    functions{6}.func = @(x) 10 * length(x) + sum(x.^2 - 10 * cos(2 * pi * x));
    
    functions{7}.name = 'Ackley';
    functions{7}.bounds = [-32, 32];
    functions{7}.optimum = 0;
    functions{7}.func = @(x) -20 * exp(-0.2 * sqrt(sum(x.^2) / length(x))) - exp(sum(cos(2 * pi * x)) / length(x)) + 20 + exp(1);
    
    functions{8}.name = 'Griewank';
    functions{8}.bounds = [-600, 600];
    functions{8}.optimum = 0;
    functions{8}.func = @(x) 1 + sum(x.^2) / 4000 - prod(cos(x ./ sqrt(1:length(x))));
    
    functions{9}.name = 'Schwefel';
    functions{9}.bounds = [-500, 500];
    functions{9}.optimum = 0;
    functions{9}.func = @(x) 418.9829 * length(x) - sum(x .* sin(sqrt(abs(x))));
    
    functions{10}.name = 'Michalewicz';
    functions{10}.bounds = [0, pi];
    functions{10}.optimum = -29.6;
    functions{10}.func = @(x) -sum(sin(x) .* sin((1:length(x)) .* x.^2 / pi).^20);
end

function plot_convergence_comparison(pso_histories, hybrid_histories, func_name)
    pso_mean = mean(pso_histories, 1);
    pso_std = std(pso_histories, 0, 1);
    hybrid_mean = mean(hybrid_histories, 1);
    hybrid_std = std(hybrid_histories, 0, 1);
    
    iterations = 0:(length(pso_mean) - 1);
    
    figure('Name', sprintf('Convergence: %s', func_name), 'NumberTitle', 'off');
    
    subplot(1, 2, 1);
    semilogy(iterations, pso_mean, 'b-', 'LineWidth', 2);
    hold on;
    fill([iterations, fliplr(iterations)], [pso_mean - pso_std, fliplr(pso_mean + pso_std)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    grid on;
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Best Fitness', 'FontSize', 12);
    title(sprintf('Standard PSO - %s', func_name), 'FontSize', 13, 'FontWeight', 'bold');
    legend('Mean', '±1 Std', 'Location', 'best');
    
    subplot(1, 2, 2);
    semilogy(iterations, hybrid_mean, 'r-', 'LineWidth', 2);
    hold on;
    fill([iterations, fliplr(iterations)], [hybrid_mean - hybrid_std, fliplr(hybrid_mean + hybrid_std)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    grid on;
    xlabel('Iteration', 'FontSize', 12);
    ylabel('Best Fitness', 'FontSize', 12);
    title(sprintf('Hybrid PSO-AGD - %s', func_name), 'FontSize', 13, 'FontWeight', 'bold');
    legend('Mean', '±1 Std', 'Location', 'best');
    
    figure('Name', sprintf('Overlay: %s', func_name), 'NumberTitle', 'off');
    semilogy(iterations, pso_mean, 'b-', 'LineWidth', 2.5);
    hold on;
    fill([iterations, fliplr(iterations)], [pso_mean - pso_std, fliplr(pso_mean + pso_std)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    semilogy(iterations, hybrid_mean, 'r-', 'LineWidth', 2.5);
    fill([iterations, fliplr(iterations)], [hybrid_mean - hybrid_std, fliplr(hybrid_mean + hybrid_std)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    grid on;
    xlabel('Iteration', 'FontSize', 13);
    ylabel('Best Fitness', 'FontSize', 13);
    title(sprintf('Convergence Comparison - %s', func_name), 'FontSize', 14, 'FontWeight', 'bold');
    legend('Standard PSO', '', 'Hybrid PSO-AGD', '', 'Location', 'best');
end

function plot_summary_comparison(all_results)
    num_funcs = length(all_results);
    func_names = cell(num_funcs, 1);
    pso_means = zeros(num_funcs, 1);
    hybrid_means = zeros(num_funcs, 1);
    
    for f = 1:num_funcs
        func_names{f} = all_results{f}.name;
        pso_means(f) = all_results{f}.pso_stats.mean;
        hybrid_means(f) = all_results{f}.hybrid_stats.mean;
    end
    
    figure('Name', 'Summary Comparison', 'NumberTitle', 'off');
    x = 1:num_funcs;
    width = 0.35;
    
    bar(x - width/2, pso_means, width, 'FaceAlpha', 0.8);
    hold on;
    bar(x + width/2, hybrid_means, width, 'FaceAlpha', 0.8);
    
    set(gca, 'XTick', x, 'XTickLabel', func_names, 'XTickLabelRotation', 45);
    set(gca, 'YScale', 'log');
    xlabel('Benchmark Function', 'FontSize', 12);
    ylabel('Mean Final Fitness', 'FontSize', 12);
    title('Performance Comparison Across All Functions', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Standard PSO', 'Hybrid PSO-AGD', 'Location', 'best');
    grid on;
end

