
def convert_to_numerical(c_f_list, df):
  for feature in c_f_list:
    labels = df[feature].astype('category').cat.categories.tolist()
    replace_dict = {feature : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
    df.replace(replace_dict, inplace=True)

#creating distributions 
def create_distributions(df, f_list):
  d_list = [y for x, y in df.groupby(f_list, as_index=False)]

  return d_list

#creating X_train and y_train for the full dataset
def create_scaled_data(train, test):
  from sklearn.model_selection import train_test_split
  from sklearn import preprocessing
  
  X = train.iloc[:, :-1].values
  y = train.iloc[:, -1].values
  X_train, X_val, ytrain, yval = train_test_split(X, y, test_size=0.2, random_state=1) 

  #creating X_test and y_test
  X_test = test.iloc[:, :-1].values
  ytest = test.iloc[:, -1].values

  #normalizing all values
  min_max_scaler = preprocessing.MinMaxScaler()
  train_scaled = min_max_scaler.fit_transform(X_train)
  val_scaled = min_max_scaler.fit_transform(X_val)
  test_scaled = min_max_scaler.fit_transform(X_test)

  return train_scaled, val_scaled, test_scaled, ytrain, yval, ytest

#create distribution wise training and validation sets
def create_distribution_train_val(dist_list):
  t_v_list = []
  for d in distribution_list_train:
    X = d.iloc[:, :-1].values
    y = d.iloc[:, -1].values
    Xd_train, Xd_val, yd_train, yd_val = train_test_split(X, y, test_size=0.2, random_state=1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    Xd_train_scaled = min_max_scaler.fit_transform(Xd_train)
    Xd_val_scaled = min_max_scaler.fit_transform(Xd_val) 

    t_v_list.append((Xd_train_scaled, Xd_val_scaled, yd_train, yd_val))
  
  return t_v_list


