function [J, grad] = costFunction(theta, X, Y, lambda)
  %%    COST FUNCTION - Compute cost and gradient for logistic regression
  %%    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %%    parameter for logistic regression and the gradient of the cost
  %%    w.r.t. to the parameters.
  
  %% number of training examples
  M = length(Y); 
  J    = 0;
  grad = zeros(size(theta));
  
  %% =============================================================
  %%     COST FUNCTION AND GRADIENT CALCULATOR
  %% =============================================================
  
  h = 1./(1+exp(-X*theta));
  
  J = (1/M)*sum(-Y.*log(h)-(1-Y).*log(1-h)) + (lambda/(2*M))*sum(theta(2:end).^2);
  grad(1) = (1/M)*sum((h-Y).*X(:,1));
  for ii = 2:length(theta)
    grad(ii) = (1/M)*sum((h-Y).*X(:,ii))+lambda*theta(ii)/M;
  end
end
