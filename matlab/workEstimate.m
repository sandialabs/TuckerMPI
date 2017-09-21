function [min_flop_order, min_mem_order, min_flops, min_mem] = workEstimate(core_dims, reconstruct_dims)

ndims = size(core_dims,2);

% Compute all possible orderings
permutations = perms(1:ndims);

for i=1:size(permutations,1)
    current_dims = core_dims;
    
    % Memory usage: storage cost of current tensor, storage cost of
    % previous tensor, and storage cost of all remaining factor matrices
    
    % Multiplications: number of factor matrix rows * sizeof current tensor
    flops=0;
    mem = sum(core_dims.*reconstruct_dims);
    max_mem = mem;
    for j=1:size(permutations,2)
        flops = flops + reconstruct_dims(:,permutations(i,j))*prod(current_dims);
        mem = mem + prod(current_dims);
        current_dims(:,permutations(i,j)) = reconstruct_dims(:,permutations(i,j));
        mem = mem + prod(current_dims) - reconstruct_dims(:,permutations(i,j))*core_dims(:,permutations(i,j));
        max_mem = max(mem,max_mem);
    end
    
    if i==1
        min_flops = flops;
        min_flop_order = permutations(i,:);
    elseif flops < min_flops
        min_flops = flops;
        min_flop_order = permutations(i,:);
    end
    
    if i==1
        min_mem = max_mem;
        min_mem_order = permutations(i,:);
    elseif max_mem < min_mem
        min_mem = max_mem;
        min_mem_order = permutations(i,:);
    end
end


end