def convert_to_numerical(c_f_list, df):
  for feature in c_f_list:
    labels = df[feature].astype('category').cat.categories.tolist()
    replace_dict = {feature : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
    df.replace(replace_dict, inplace=True)
