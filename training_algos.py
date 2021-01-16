def train_bilevel(epochs, k, tv_list, w_list, model, e1, e2):
  for t in range(epochs):
    train_loss = 0
    Ui_loss_params_list = []
    
    for i in range(k):
      if t == 0 and i == 0:
        pass
      else:
        for param in model.parameters():
          param.grad.data.zero_()
      ypred = model(torch.Tensor(train_val_list[i][0]))
      L_Ui = L(ypred, torch.Tensor(train_val_list[i][2].astype(np.long)).long())
      L_Ui.backward(retain_graph=True)
      
      grads_list_Ui = []
      for param in model.parameters():
        grads_list_Ui.append(param.grad.view(-1))

      Ui_loss_params = torch.cat(grads_list_Ui)
      Ui_loss_params_list.append(Ui_loss_params)


      train_loss = train_loss + (w_list[i] * L_Ui)

    for param in model.parameters():
        param.grad.data.zero_()

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()
    if t % 100 == 0:
      print(train_loss)

    val_loss_list = []
    for j in range(k):
      ypred = model(torch.Tensor((train_val_list[j][1])))
      val_loss_list.append(L(ypred, torch.Tensor(train_val_list[j][3]).long())) 

    i_t = np.argmax(np.array(val_loss_list))

    for param in model.parameters():
      param.grad.data.zero_()

    
    val_loss_list[i_t].backward()

    grads_list_val = []
    for param in model.parameters():
      grads_list_val.append(param.grad.view(-1))

    val_loss_params = torch.cat(grads_list_val)

    for i in range(k):
      w_list[i] = w_list[i] + eta1*eta2*torch.dot(val_loss_params, Ui_loss_params_list[i])
      if w_list[i] < 0:
        w_list[i] = 0
  
  return model, w_list
  
def train_bilevel_qform(epochs, k, tv_list, w_list, model, e1, e2, q):
  for t in range(epochs):
    train_loss = 0
    Ui_loss_params_list = []
    
    for i in range(k):
      if t == 0 and i == 0:
        pass
      else:
        for param in model.parameters():
          param.grad.data.zero_()
      ypred = model(torch.Tensor(tv_list[i][0]))
      L_Ui = L(ypred, torch.Tensor(tv_list[i][2].astype(np.long)).long())
      L_Ui.backward(retain_graph=True)
      
      grads_list_Ui = []
      for param in model.parameters():
        grads_list_Ui.append(param.grad.view(-1))

      Ui_loss_params = torch.cat(grads_list_Ui)
      Ui_loss_params_list.append(Ui_loss_params)


      train_loss = train_loss + (w_list[i] * L_Ui)

    for param in model.parameters():
        param.grad.data.zero_()

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()
    if t % 100 == 0:
      print(train_loss)

    qloss_sum = 0
    for j in range(k):
      ypred = model(torch.Tensor((tv_list[j][1])))
      L_Vi = L(ypred, torch.Tensor(tv_list[j][3]).long())
      qloss_sum = qloss_sum + torch.pow(L_Vi,  q)


    #i_t = np.argmax(np.array(val_loss_list))

    qloss = torch.pow(qloss_sum, 1/q)

    for param in model.parameters():
      param.grad.data.zero_()

    
    # val_loss_list[i_t].backward()
    qloss.backward()

    grads_list_val = []
    for param in model.parameters():
      grads_list_val.append(param.grad.view(-1))

    val_loss_params = torch.cat(grads_list_val)

    for i in range(k):
      w_list[i] = w_list[i] + e1*e2*torch.dot(val_loss_params, Ui_loss_params_list[i])
      if w_list[i] < 0:
        w_list[i] = 0

  return model, w_list
def train_full_data(epochs, X, y, model):
  for t in range(epochs):
    ypred = model(torch.Tensor(X))
    train_loss = L(ypred, torch.Tensor(y).long())
    
    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    if t % 100 == 0:
      print(train_loss)
  return model
  
def train_vanilla_qloss(epochs, k, tv_list, model, q):
  for t in range(epochs):
    train_loss = 0
    Ui_loss_params_list = []
    
    for i in range(k):
      ypred = model(torch.Tensor(tv_list[i][0]))
      L_Ui = L(ypred, torch.Tensor(tv_list[i][2].astype(np.long)).long())
      L_Uiq = torch.pow(L_Ui, q)


      if t % 100 == 0:
        print("crossentropy_loss slice {}: {}".format(i, L_Ui))
        print("q_loss slice {}: {}".format(i, L_Uiq))
        
      
      train_loss = train_loss + (L_Uiq)

    train_lossq = torch.pow(train_loss, (1/q))
    optimizer.zero_grad()

    train_lossq.backward()

    optimizer.step()
    if t % 100 == 0:
      print("crossentropy loss: {}".format(train_loss))
      print("q_loss: {}".format(train_lossq))
  
  return model

