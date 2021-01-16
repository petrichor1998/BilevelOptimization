#pytorch Logreg model testing on full dataset

def calculate_total_accuracy(X, y, model, L, optimizer):
  with torch.no_grad():
    ypred_test = model(torch.Tensor(X))
    values, indices = torch.max(ypred_test, 1)
    correct = torch.eq(indices, torch.Tensor(y)).sum()
    accuracy = correct.item()/len(indices)

  print("total correct predictions: {} ".format(correct.item()))
  print("total number of items: {}".format(len(indices)))
  print("Accuracy: {}".format(accuracy))

def calculate_dist_accuracy(dist_list, model, L, optimizer):
  with torch.no_grad():
    accuracy_list = []
    min_max_scaler = preprocessing.MinMaxScaler()
    for d in dist_list:
      X_scaled = min_max_scaler.fit_transform(d.values[:, :-1])
      ypred_test = model(torch.Tensor(X_scaled))
      values, indices = torch.max(ypred_test, 1)
      correct = torch.eq(indices, torch.Tensor(d.values[:, -1])).sum()
      accuracy = correct.item()/len(indices)
      accuracy_list.append(accuracy)
    
    for a in accuracy_list:
      print(a)
