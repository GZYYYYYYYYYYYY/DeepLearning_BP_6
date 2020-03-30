%% cost.m
% Define Cost Function
%% Step 4: Define Cost Function
function [J] = cost(a, x, beta)
    J = 1/2 * sum((a - x).^2) + beta*sum(a);
end


