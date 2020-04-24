# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score,precision_score,recall_score,auc, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def group_by_with_main_metrics(df, gbcolumn, index, target):
    gbdf = (df.groupby(gbcolumn)
            .agg({index:'count',target:'sum'})
            .rename_axis(gbcolumn)
            .reset_index())
    gbdf['share'] = gbdf[index]/gbdf[index].sum()
    gbdf['target_rate'] = gbdf[target]/gbdf[index]
    return gbdf


def show_acc_prec_rec(y, preds):
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    print(f'Accuracy: {acc} \nPrecision: {prec} \nRecall: {rec}')
    
    
def evaluate_model(y, preds, G=0.5, RC=1):
    fpr, tpr, _ = roc_curve(y, preds)

    auc_score = auc(fpr, tpr)

    # clear current figure
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

    # it's helpful to add a diagonal to indicate where chance 
    # scores lie (i.e. just flipping a coin)
    plt.plot([0,1],[0,1],'r--')

    # Cost function

    churn_rate_x = y.sum()/y.count()
    r = churn_rate_x*tpr*G - (1-churn_rate_x)*fpr*RC
    print("test churn_rate_x:"+str(round(churn_rate_x,3)))
    print("max profit:"+str(round(max(r),3)))
    pos = np.argmax(r)
    print("position:"+str(pos))
    print(f"tpr:{round(tpr[pos],3)} fpr:{round(fpr[pos],3)}")
    plt.plot([0,G*churn_rate_x/(1-churn_rate_x)],[0,1],'r--', color='gray', label='Profit zero')
    plt.scatter( fpr[pos],tpr[pos], s=50)
    plt.text(fpr[pos]+0.05,tpr[pos],'Optimal', horizontalalignment='left')

    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc='lower right')
    plt.show()
