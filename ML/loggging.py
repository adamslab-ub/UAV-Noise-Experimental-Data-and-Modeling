import plotly.express as px
from _datetime import datetime
import pandas as pd
import numpy as np
import os
if not os.path.exists('Logs'):
    os.makedirs('Logs')
if not os.path.exists('Logs/imgs'):
    os.makedirs('Logs/imgs')
def log(df):#,df2,df3):
    now = datetime.now()
    #dt_string = now.strftime("%d%m%Y%H:%M:%S")
    dt_string = now.strftime("%d%m%Y%H_%M_%S")
    img_names = ['Logs/imgs/loss'+dt_string+'.png','Logs/imgs/train_vars' + dt_string + '.png']#,'Logs/imgs/train_var2' + dt_string + '.png','Logs/imgs/test_var1' + dt_string + '.png','Logs/imgs/test_var2' + dt_string + '.png']
    title_str ='#Log'+dt_string+'\n'
    vars = open('config1.py','r')
    vars_str = '##Variables: \n' + vars.read() +'\n'
    vars.close()
    print('creating plots')
    fig = px.line(df,y=['train_loss','test_loss'],x=df.index.values)
    fig.write_image(img_names[0])
    imgs =[]
    final = title_str + vars_str
    for i in range(len(img_names)):
        #final =final + '\n![](/media/manaswin/Userfiles/Edu/lab/Pytorch_Surrogate/'+img_names[i]+')\n'
        final =final + '\n![](C://Users/rayha/Desktop/Heuristic Project/Summer Project/Pytorch_DNN/'+img_names[i]+')\n'
    file = open('Logs/log'+dt_string+'.md','w+')
    file.write(final)
    file.close()
    np.save('Logs/loss_datalog_'+dt_string,df)