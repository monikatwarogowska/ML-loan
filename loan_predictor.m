%% =============================================================
%%      LOAN PREDICTOR
%% =============================================================
%%
%%   LOAN PREDICTOR - train the algorithm on train.csv data, divided into train and cross-validation sets, and use supervised learning with fminunc minimization method to find optimal parameters. Then, the model is used to predict the loan classifier for data in test.csv. 

%%   Functions:
%%           cost_function - calculates cost function of a multivariate features set
%%           data_preparation - prepares data and checks for missing data

clear all
close all

%% =============================================================
%%                         LOAD TRAIN DATA
%% =============================================================
data_raw = readtable('train.csv');
[X, Y]   = trainData_preparation(data_raw,1);
[M, N]   = size(X);

%% Division into train and cross-validation sets
Mtr = floor(0.7*M);
Mcv = M-Mtr;

%% =============================================================
%%                        OPTIMISATION
%% =============================================================
lambda_vec = [0 0.1 0.5 1 1.5 2 2.5 3 4 5 6 7 8 9 10 15 20 25 30 35 40 45 50 60 70 80 100  110 120 130 140 150 160 170 180 190 200 220 240 260 280 300];
initial_theta = zeros(size(X,2)+1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
error_old = 1000;
for ii = 1:length(lambda_vec)
  lambda = lambda_vec(ii);
  idx_tr = 1:1:Mtr;
  idx_cv = [];
  for ii=1:M
    if ~ismember(ii,idx_tr)
      idx_cv = [idx_cv ii];
    end
  end
  X_ext = [ones(Mtr,1) X(idx_tr,:)];
  %% Training on the train set
  initial_theta = 0*initial_theta;
  [theta_temp, cost] = fminunc(@(t)(costFunction(t, X_ext, Y(idx_tr,1),lambda)), initial_theta, options);

  %% Verification on the cross validation set
  h    = 1./(1+exp(-[ones(Mcv,1) X(idx_cv,:)]*theta_temp));
  h_th = 0.5;
  p    = 1*(h>=h_th);

  error_new = 100*nnz(abs(p-Y(idx_cv,1)))/(Mcv);
  if error_new < error_old
      theta = theta_temp;
      error_old = error_new;
      lambda_final = lambda;
  end
end

%% =============================================================
%%                LOAN PREDICTION OF TEST DATA
%% =============================================================

data_test_raw = readtable('test.csv');
[Xt]    = testData_preparation(data_test_raw,1);
[Mt,Nt] = size(Xt);

h = 1./(1+exp(-[ones(Mt,1) Xt]*theta));
p = 1*(h>=h_th);
Loan_Status = cell(length(p),1);
for ii=1:length(p)
  if p(ii) == 1
    Loan_Status(ii,1) = {'Y'};
  else
    Loan_Status(ii,1) = {'N'};
  end
end

%% Saving data
Loan_ID = data_test_raw{1:end,1};
results = table(Loan_ID,Loan_Status);
writetable(results,'sample_submission.csv')

