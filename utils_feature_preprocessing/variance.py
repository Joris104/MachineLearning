import numpy as np

def getVar(sequence, axis):
    #return variance over all frames
    #-----------------PARAMETER TUNING-----------------
    #Training accuracy 0.9299316642594441 +/- 0.0035055100240696563
    #Cross-validation accuracy: 0.7076612803498793 +/- 0.06295446103947513
    #Best estimator:
    #Pipeline(steps=[('preprocessing',
    #                 Pipeline(steps=[('scaler', StandardScaler()),
    #                                 ('decompose', PCA())])),
    #                ('feature_selection',
    #                 Pipeline(steps=[('selectKBest', SelectKBest(k=1100))])),
    #                ('classifier',
    #                 RidgeClassifier(alpha=1, class_weight='balanced', tol=1e-09))])
    #---------------------------------------------------
    new_sequence = sequence[:,:,axis]
    return np.var(new_sequence,0)

def getVar2(sequence, axis):
    #return variance over all frames of first halve and second halve
    new_sequence = sequence[:,:,axis]
    n = new_sequence.shape[0]
    if n>1:
        new_sequence_1 = new_sequence[:int(n/2)]
        new_sequence_2 = new_sequence[int(n/2):]
        var_1 = np.var(new_sequence_1,0)
        var_2 = np.var(new_sequence_2,0)
        return np.concatenate((var_1,var_2))
    else: #if n==1 return getVar two times
        return np.concatenate((np.var(new_sequence,0),np.var(new_sequence,0)))