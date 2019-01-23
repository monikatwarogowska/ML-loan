function [X,Y,R] = data_preparation_NEW(data_list, train_flag,non_lin_flag)
  %% DATA_PREPARATION checks raw data set for errors and normalizes features
  %%       [X,Y] = data_preparation(data_list,train_check) adjust features and prepares raw data by checking missing entries, Nan entries; normalizes the features and computes the correlation matrix

  %% data_list - structure with first column corresponding to Loan_id and features: X = data_list{:,2:end-1} and classification_vector: Y = data_list(:,end)
  %%
  %%       X = [ x1 = male ;
  %%             x2 = female ;
  %%             x3 = married ;
  %%             x4 = zero_dependents ;
  %%             x5 = one_dependent ;
  %%             x6 = two_dependents ;
  %%             x7 = threemore_dependents ;
  %%             x8 = graduated ;
  %%             x9 = self_employed ;
  %%             x10 = applicant_income ;
  %%             x11 = coapplicant_income ;
  %%             x12 = loan_amount ;
  %%             x13 = loan_amount_term ;
  %%             x14 = credit_history ;
  %%             x15 = rural_area ;
  %%             x16 = semiurban_area ;
  %%             x17 = urban_area ]
  %%
  %% train_flag - set to 1 if data is a training set and contain decision data Y 
  %% non_lin_frag - set to 1 if quadratic nonlinearities are taken into account
  
  [M,N] = size(data_list);

  %% Manually adjusted list of features: numbers of USELESS features have to be delated
  idx_fts = [2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17];
  num_features = length(idx_fts);

  %% =============================================================
  %%                       MISSING DATA 
  %%  -> less significant data (such as gender, marriage status, education): replacement by generic value
  %% -> missing "applicant income" or "loan amount" : data example is deleted
  %% =============================================================
  
  mdvec = ones(M,1);    % vector of  indexes that have missing important data: loan amount, applicant income   
  for ii = 1:M
    for jj = 1:N
      if ismissing(data_list(ii,jj))
	switch jj
	  case 2
            data_list.Gender(ii) = {'Female'};
	  case 3
            data_list.Married(ii) = {'Yes'};
	  case 4
            data_list.Dependents(ii) = {'0'};
	  case 5
            data_list.Education(ii) = {'Graduate'};
	  case 6
            data_list.Self_Employed(ii) = {'No'};
	  case 7
            mdvec(ii,1) = 0;
	    break
	  case 8
            data_list.CoapplicantIncome(ii) = data_list(ii,7);
	  case 9
	    mdvec(ii,1) = 0;
	    break
	  case 10
            data_list.Loan_Amount_Term(ii) = 360;
	  case 11
            data_list.Credit_History(ii) = 1;
	  case 12
            data_list.Property_Area(ii) = {'Urban'};
	  otherwise
            disp('other value')
	    pause
	end
      end
    end
  end
  idx = find(mdvec>0);
  M = length(idx)
  data_list = data_list(idx,:);

  %% =============================================================
  %%               FEATURES
  %% =============================================================
  X_temp = zeros(M,17);
  X_temp(:,1) = ones(M,1).*(grp2idx(data_list{1:end,2})==1);
  X_temp(:,2) = ones(M,1).*(grp2idx(data_list{1:end,2})~=1);
  X_temp(:,3) = ones(M,1).*(grp2idx(data_list{1:end,3})==1);
  X_temp(:,4) = ones(M,1).*(grp2idx(data_list{1:end,4})==2);
  X_temp(:,5) = ones(M,1).*(grp2idx(data_list{1:end,4})==1);
  X_temp(:,6) = ones(M,1).*(grp2idx(data_list{1:end,4})==3);
  X_temp(:,7) = ones(M,1).*(grp2idx(data_list{1:end,4})==4);
  X_temp(:,8) = ones(M,1).*(grp2idx(data_list{1:end,5})==1);
  X_temp(:,9) = ones(M,1).*(grp2idx(data_list{1:end,6})==2);
  X_temp(:,10) = data_list{1:end,7};
  X_temp(:,11) = data_list{1:end,8};
  X_temp(:,12) = data_list{1:end,9};
  X_temp(:,13) = data_list{1:end,10};
  X_temp(:,14) = data_list{1:end,11};
  X_temp(:,15) = ones(M,1).*(grp2idx(data_list{1:end,12})==1);
  X_temp(:,16) = ones(M,1).*(grp2idx(data_list{1:end,12})==3);
  X_temp(:,17) = ones(M,1).*(grp2idx(data_list{1:end,12})==2);

  X = X_temp(:,idx_fts);

  %% =============================================================
  %%                  NON-LINEAR FEATURES
  %% =============================================================
  if non_lin_flag == 1
    for ii = 1:size(X,2)
      for jj = ii:size(X,2)
	X_newF = X(:,ii).*X(:,jj);
	X = [X X_newF];
      end
    end
  end

  %% =============================================================
  %%               NORMALIZATION
  %% =============================================================
  for ii=1:size(X,2)
    mn = mean(X(:,ii));
    st = std(X(:,ii),1);
    X(:,ii) = (X(:,ii)-mn)/st;
  end

  %% =============================================================
  %%         CLASSIFICATION DATA
  %% =============================================================
  if train_flag == 1
    mdvec = ones(M,1);
    for ii = 1:M
      if ismissing(data_list{ii,end})
	fprintf('No data for training set output Y')
	mdvec(ii) = 0; 
      end
    end
    idx = find(mdvec>0);
    M = length(idx);
    data_list = data_list(idx,1:end);
    
    if isequal(data_list{1,end},{'Y'})   
      Y = grp2idx(data_list{1:end,end});
      idx_N = find(Y==2);
      Y(idx_N) = 0;
    else
      Y = grp2idx(data_list{1:end,end})-1;
    end
  else
    Y=[];
  end
  X = X(idx,:);
  ID = data_list{1:end,1}; 
  ID(1:10)
  Y(1:10)
  %% =============================================================
