function [optimal_order_stats,full_stats] = mttm_ordering(idims,odims,pdims)

    N = length(idims);
    
    % for exhaustive search, consider all permutations
    P = perms(1:N);   
    f = zeros(factorial(N),1);
    w = zeros(factorial(N),1);
    m = zeros(factorial(N),1);
    for i = 1:factorial(N)
        f(i) = mttm_flops(idims,odims,P(i,:));
        w(i) = mttm_words(idims,odims,pdims,P(i,:));
        m(i) = mttm_mem(idims,odims,P(i,:));
    end
    
    % find mins of each metric with optimal orders
    [minf,fi] = min(f);
    [minw,wi] = min(w);
    [minm,mi] = min(m);  
    
    % Convert perms array to vector of representative numbers (0-indexed)
    %   * fails if N > 9
    PV = P * logspace(N-1,0,N)';
    
    % Construct table of all permutations, using relative values
    full_stats = table(f,w,m);
    full_stats.Properties.VariableNames = {'Flops','Bandwidth','Memory'};
    full_stats.Properties.RowNames = cellstr(num2str(PV));
    full_stats.Flops = full_stats.Flops / minf;
    full_stats.Bandwidth = full_stats.Bandwidth / minw;
    full_stats.Memory = full_stats.Memory / minm;
    
    % return table with stats of 
    inds = [fi,wi,mi];
    numdups = [length(find(f == minf));length(find(w == minw));length(find(m == minm))];
    optimal_order_stats = table(PV(inds),f(inds),w(inds),m(inds),numdups);
    optimal_order_stats.Properties.RowNames = {'Opt Flops','Opt BW','Opt Mem'};
    optimal_order_stats.Properties.VariableNames = {'Order','Flops','Bandwidth','Memory','NumDups'};

end

function flops = mttm_flops(indims,outdims,order)
% compute number of multi-TTM flops if TTMs performed in specified order
    
    N = length(indims);
    flops = 0;
    for n = 1:N
        flops = flops + prod(outdims(order(1:n)))*prod(indims(order(n:N)));
    end
    
end

function words = mttm_words(indims,outdims,procdims,order)
% compute number of multi-TTM words sent by all procs if TTMs performed in specified order
    
    N = length(indims);
    words = 0;
    for n = 1:N
        words = words + prod(outdims(order(1:n)))*prod(indims(order(n+1:N))) * (procdims(order(n))-1);
    end
    
end

function mem = mttm_mem(indims,outdims,order)
% compute temp mem required for multi-TTM if TTMs performed in specified order
    
    N = length(indims);
    mem = 0;
    for n = 2:N-1
        mem = max(mem,prod(outdims(order(1:n)))*prod(indims(order(n+1:N))));
    end
       
end