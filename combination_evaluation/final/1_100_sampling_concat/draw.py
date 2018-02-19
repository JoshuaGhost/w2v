from pandas import DataFrame;                                           
import matplotlib.pyplot as plt                              
data = DataFrame.from_csv('eval.csv')                     
import pylab as pl                                        
for column in (data.loc[data.index!='base_line']).columns[data.columns!=
'nsubs']:                                                 
    pl.figure(figsize=(20, 20))                           
    axes = data[data.nsubs!=0].boxplot(column=column, by='nsubs')
    fig = axes.get_figure()
    fig.suptitle('')
    plt.axhline(y=data.loc[data.index=='base_line'][column][0], color='r', linestyle='-')                                         
    pl.savefig('100_c_{}.png'.format(column)) 
