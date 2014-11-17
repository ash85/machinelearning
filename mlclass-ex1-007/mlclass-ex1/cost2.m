
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;


function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions = X*theta;
sqrErrors = (predictions-y).^2;

J=1/(2*m)*sum(sqrErrors);


% =========================================================================

end

computeCost(X, y, theta)


function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta
for iter = 1:num_iters
    

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    x = X(:,2);
    hyp = theta(1) + (theta(2)*x);

    t_zero = theta(1) - alpha * (1/m) * sum(hyp-y);
    t_one  = theta(2) - alpha * (1/m) * sum((hyp - y) .* x);

    
    theta = [t_zero; t_one];
    hyp
    theta



    % ============================================================

    % Save the cost J in every iteration   
    fprintf('%f \n',iter); 
    J_history(iter) = computeCost(X, y, theta);
    fprintf('%f \n', J_history(iter));

end


end
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));
