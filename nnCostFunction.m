function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1), X];
z_2 = X * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(size(a_2, 1), 1), a_2];
h_theta = sigmoid(a_2 * Theta2');

Y = zeros(num_labels, m);
for i=1:num_labels
    Y(i,:) = y==i;
end
J = sum(sum(-Y .* log(h_theta)' - (1-Y).* log(1-h_theta)'));
J = J / m
% disp(J)
% for i=1:m
%     for k=1:num_labels
%         y_k = y == k;
%         J = J + (-y_k(i) * log(h_theta(i, k)) - (1-y_k(i))*log(1-h_theta(i, k)));
%     end
% end
% J = J / m;

Theta1_reg = Theta1(:, 2:end);
Theta2_reg = Theta2(:, 2:end);
J = J + (lambda / (2 * m)) * (sum(sum(Theta1_reg .^ 2)) + sum(sum(Theta2_reg .^ 2)));

for i = 1:m
    a1 = X(i,:)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    d3 = a3 - Y(:, i);
    z2 = [1; z2];
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    
    d2 = d2(2:end);
    Theta2_grad = Theta2_grad + d3 * a2';
    Theta1_grad = Theta1_grad + d2 * a1';
end

Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

Theta2_grad(:,2:end) = Theta2_grad(:, 2:end) + (lambda / m) .* (Theta2(:, 2:end));
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) .* (Theta1(:, 2:end));




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
