import random
import bisect
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import interp
from scipy.stats import mstats
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import ShuffleSplit

def run_cross_validation(_df, _classifier, _features_columns, _id, _target, _prob_name,
                         _n_iter=5, _test_size=.3, _sample_weight=None ,_random_state=0, _early_stopping_rounds=40):

    if str(type(_classifier)).startswith("<class 'xgboost"):
            model_type = "xgboost"
    elif str(type(_classifier)).startswith("<class 'lightgbm"):
            model_type = "lightgbm"          
    else:
        print("Model not supported, used function run_cross_validation_sk instead")
        return False
        
    # cross validation type can be changed here
    ss = ShuffleSplit(n_splits=_n_iter, test_size=_test_size, random_state=_random_state)
    results_cv_targeting = pd.DataFrame([], columns=[_id, _target, 'fold', _prob_name])

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)
    
    mean_lift = 0.0
    mean_lift_decay = 0.0
    mean_cum_pos = 0.0
    
    nb_calls_cv = pd.DataFrame([],columns=['nb_actions', 'total_population', 'total_pos_targets', 
                                           'nb_pos_targets', 'pos_rate', 'Percentage_of_pos_targets_found', 
                                           'Percentage_of_Population', 'Lift'])
    feature_importances = pd.DataFrame([], columns=['feature', 'importance', 'fold'])

    fig = plt.figure(figsize=(6, 12))
    fig.subplots_adjust(bottom=-0.5, left=-0.5, top=0.5, right=1.5)

    print ('cross validation started')
    plt.gcf().clear()
    ax1 = plt.subplot(3,3,1)
    ax2 = plt.subplot(3,3,2)
    ax3 = plt.subplot(3,3,3)
    ax4 = plt.subplot(3,3,4)
    ax5 = plt.subplot(3,3,5)
    ax6 = plt.subplot(3,3,6)
    ax7 = plt.subplot(3,3,7)
    ax8 = plt.subplot(3,3,8)
    ax9 = plt.subplot(3,3,9)
                        

    for i, (train_index, valid_index) in enumerate(ss.split(_df[_id].unique())):
        customer_id = _df[_id].unique().copy()
        shuffled_customer_id = np.array(sorted(customer_id, key=lambda k: random.random()))
        train_customer_id = shuffled_customer_id[train_index]
        valid_customer_id = shuffled_customer_id[valid_index]
        
        train = _df.loc[_df[_id].isin(train_customer_id), 
                        np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)
        valid = _df.loc[_df[_id].isin(valid_customer_id), 
                        np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)
        
        temp = valid[[_id, _target]].copy()
        temp['fold'] = i

        # modeling#
        train_X = train.drop([_id, _target], axis=1)
        valid_X = valid.drop([_id, _target], axis=1)

        train_Y = np.array(train[_target].astype(np.uint8))
        valid_Y = np.array(valid[_target].astype(np.uint8))
        
        if _sample_weight is not None:
            probas_ = _classifier.fit(train_X, train_Y, eval_metric='auc', eval_set=[(valid_X, valid_Y)], 
                                      sample_weight=train_X[_sample_weight],
                                      early_stopping_rounds=_early_stopping_rounds, verbose=0).predict_proba(valid_X)            
        else :
            probas_ = _classifier.fit(train_X, train_Y, eval_metric='auc', eval_set=[(valid_X, valid_Y)],
                                      early_stopping_rounds=_early_stopping_rounds, verbose=0).predict_proba(valid_X)
            
        if model_type=="xgboost":
            evals_result = _classifier.evals_result()['validation_0']['auc']
            
        elif model_type=="lightgbm":
            evals_result = _classifier.evals_result_['valid_0']['auc']
            
        else:
            evals_result = np.zeros(max(_classifier.n_estimators,250))
            
        probabilities = pd.DataFrame(data=probas_[:, 1], index=valid_X.index, columns=[_prob_name])

        temp = temp.join(probabilities, how='left')
        results_cv_targeting = results_cv_targeting.append(temp)


        ###############################################################################
        # Plot probability distribution
        #ax1 = plt.subplot(3, 3, 1)
        ax1.hist(probas_[:, 1], range=(0, 1), bins=100, label="fold %d" % (i), histtype="step")
        
        ###############################################################################
        # plot proba distribution for both class
        target_probs = pd.DataFrame(valid_Y, columns=['target'])
        target_probs['probs'] = probas_[:, 1]
        #ax2 = plt.subplot(3, 3, 2)
        if i==0:
            ax2.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100, 
                     label="Class 1", histtype="step", color='green', density=True)
            ax2.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100, 
                     label="Class 0", histtype="step", color='red', density=True)
        else:
            ax2.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100, 
                     histtype="step", color='green', density=True)
            ax2.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100, 
                     histtype="step", color='red', density=True)


        ###############################################################################
        # Plot calibration plots
        fraction_of_positives, mean_predicted_value = calibration_curve(valid_Y, probas_[:, 1], n_bins=20)
        #ax3 = plt.subplot(3,3,3)
        ax3.plot(mean_predicted_value, fraction_of_positives, "P-", label="fold %d" % (i), lw=1)


        ###############################################################################
        # plot evals_result
        #ax4 = plt.subplot(3, 3, 4)
        ax4.plot(range(len(evals_result)), evals_result, label='Fold %d' %(i), lw=1)


        ###############################################################################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(valid_Y, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        #ax5 = plt.subplot(3, 3, 5)
        ax5.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i, roc_auc), lw=1)

        
        ###############################################################################
        # Compute Precision-Recall curve and area the curve
        precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
        mean_precision += interp(mean_recall, recall[::-1], precision[::-1])
        pr_auc = auc(recall, precision)

        #ax6 = plt.subplot(3, 3, 6)
        ax6.plot(recall, precision, label='PR fold %d (area = %0.2f)' % (i, pr_auc), lw=1)


        ###############################################################################
        # calculate lift related information
        cust_rank = temp[[_target, _prob_name]].copy()
        cust_rank = cust_rank.sort_values(by=_prob_name, ascending=False).reset_index(drop=True)
        cust_rank['rank'] = cust_rank.index + 1
        cust_rank['num_pos_target'] = np.cumsum(cust_rank[_target])
        pos_rate = temp[_target].mean()
        pos_sum = temp[_target].sum()

        lift_cums = []
        lift_decays = []
        cum_poss = []
        
        for q in range(10, 110, 10):
            small_q = (q - 10) / 100.0
            big_q = q / 100.0
            if q == 100:
                lift_cum = cust_rank[_target].mean() / pos_rate
                lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) :][_target].mean() / pos_rate
                cum_pos = cust_rank[_target].sum() / pos_sum
            else:
                lift_cum = cust_rank[: int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
                lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) : int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
                cum_pos = cust_rank[: int(big_q * cust_rank.shape[0])][_target].sum() / pos_sum
            
            lift_cums.append(lift_cum)
            lift_decays.append(lift_decay)
            cum_poss.append(cum_pos)

        print ('shuffle: %i, AUC: %f, lift at 10 percent: %f' % (i, roc_auc, lift_cums[0]))
        mean_lift += np.array(lift_cums)
        mean_lift_decay += np.array(lift_decays)
        mean_cum_pos += np.array(cum_poss)

        ###############################################################################
        # calculate number of calls
        max_positive_in_fold = cust_rank.num_pos_target.max()
        max_possible_actions = min(max_positive_in_fold,3000)

        nb_calls = cust_rank.copy()
        for i in range(100,int(max_possible_actions), 100):
            nb_calls['to_get_%i' %i] = nb_calls.loc[nb_calls.num_pos_target==i,'rank'].min()

        nb_calls['to_get_all'] = nb_calls.loc[nb_calls.num_pos_target==nb_calls.num_pos_target.max(),'rank'].min()

        nb_calls = nb_calls[[col1 for col1 in nb_calls.columns if col1.startswith('to_get_')]].min()
        nb_calls = pd.DataFrame(nb_calls,columns=['nb_actions'])
        nb_calls['total_population'] = cust_rank.shape[0]
        nb_calls['total_pos_targets'] = cust_rank[_target].sum()
        nb_calls['nb_pos_targets']= list(range(100,int(max_possible_actions), 100))+[int(max_possible_actions)]
        nb_calls['pos_rate'] = nb_calls.nb_pos_targets/nb_calls.nb_actions
        nb_calls['Percentage_of_pos_targets_found'] = nb_calls.nb_pos_targets/nb_calls.total_pos_targets
        nb_calls['Percentage_of_Population'] = nb_calls.nb_actions/nb_calls.total_population
        nb_calls['Lift'] = nb_calls.Percentage_of_pos_targets_found/nb_calls.Percentage_of_Population

        nb_calls_cv = nb_calls_cv.append(nb_calls)
        
        ###############################################################################
        feature_importances_data = []
        features = train_X.columns
        if model_type=="xgboost":
            for feature_name, feature_importance in _classifier.get_booster().get_score(fmap='',
                                                                                        importance_type='gain').items():
                feature_importances_data.append({
                        'feature': feature_name,
                        'importance': feature_importance
                    })

        elif model_type=="lightgbm":
            for feature_name, feature_importance in zip(_features_columns,
                                                        _classifier.booster_.feature_importance(importance_type='gain')):
                feature_importances_data.append({
                        'feature': feature_name,
                        'importance': feature_importance
                    })
                
        else :
            for feature_name in _features_columns:
                feature_importances_data.append({
                        'feature': feature_name,
                        'importance': 0
                    })
            
        temp = pd.DataFrame(feature_importances_data)
        temp['fold'] = i
        feature_importances = feature_importances.append(temp)

    #correct nb_calls_cv because sometime index has more rows than others
    nb_calls_cv = nb_calls_cv.reset_index()
    nb_calls_cv = nb_calls_cv.rename(columns={'index':'objective'})
    nb_calls_cv['undesired_index'] = nb_calls_cv.groupby(['objective'])['nb_actions'].transform('count')
    nb_calls_cv = nb_calls_cv.loc[nb_calls_cv.undesired_index==_n_iter].reset_index(drop=True)
    nb_calls_cv = nb_calls_cv.drop('undesired_index', axis=1)
    nb_calls_cv = nb_calls_cv.set_index('objective')
    
    for col in nb_calls_cv.columns:
        nb_calls_cv[col] = pd.to_numeric(nb_calls_cv[col])
    nb_calls_cv = nb_calls_cv.reset_index().groupby('objective').mean().sort_values(by='nb_pos_targets')
    
    results_cv_targeting = results_cv_targeting.reset_index(drop=True)
    
    feature_importances = feature_importances.groupby('feature')['importance'].agg([np.mean, np.std])
    feature_importances = feature_importances.sort_values(by='mean')
    feature_importances = feature_importances.reset_index()


    # plot probas for probas
    #ax1.subplot(3, 3, 1)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Predicted probas', fontsize=12)
    ax1.legend(loc="lower right")


    # plot probas for both classes
    #ax2.subplot(3, 3, 2)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Predicted probas for different classes', fontsize=12)
    ax2.legend(['Class 1', 'Class 0'], loc="lower right")


    # plot the perfectly calibrated curve
    #ax3.subplot(3,3,3)
    ax3.plot([0, 1], [0, 1], "k--")
    ax3.set_ylabel("Fraction of positives", fontsize=10)
    ax3.set_xlabel("Mean predicted value", fontsize=10)
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc="lower right")
    ax3.set_title('Calibration plots  (reliability curve)', fontsize=12)


    # plot evals_result
    #ax4.subplot(3, 3, 4)
    ax4.set_xlabel('n estimators', fontsize=10)
    ax4.set_ylabel('ROC AUC', fontsize=10)
    ax4.set_title('ROC through n_estimators', fontsize=12)
    ax4.legend(loc="lower right")

    # plot the averaged ROC curve
    #ax5.subplot(3, 3, 5)
    mean_tpr /= ss.get_n_splits()
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax5.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc)
    ax5.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax5.set_xlim([-0.05, 1.05])
    ax5.set_ylim([-0.05, 1.05])
    ax5.set_xlabel('False Positive Rate', fontsize=10)
    ax5.set_ylabel('True Positive Rate', fontsize=10)
    ax5.set_title('ROC', fontsize=12)
    ax5.legend(loc="lower right")


    # plot averaged PR curve
    #ax6.subplot(3, 3, 6)
    mean_precision /= ss.get_n_splits()
    mean_pr_auc = auc(mean_recall, mean_precision)
    ax6.plot(mean_recall, mean_precision, 'k--', label='Mean PR (area = %0.2f)' % mean_pr_auc, lw=2)
    ax6.set_xlabel('Recall', fontsize=10)
    ax6.set_ylabel('Precision', fontsize=10)
    ax6.set_title('Precision-recall', fontsize=12)
    ax6.legend(loc="lower right")
    
    # plot lift cumulative
    #ax7 = plt.subplot(3, 3, 7)
    mean_lift /= ss.get_n_splits()
    mean_cum_pos /= ss.get_n_splits()
    ax7.bar(range(10), mean_lift)
    ax7.set_xticks(range(10))
    ax7.set_xticklabels(['0-%d' %(num) for num in range(10, 110, 10)])
    for i, v in enumerate(zip(mean_lift,mean_cum_pos)):
        ax7.text(i-0.5, v[0]+.05, '%.2f' %(v[0]), color='black')
        ax7.text(i-0.5, v[0]+.4, '(%.0f%%)' %(v[1]*100), color='red', fontsize=8)
    ax7.set_xlabel('Rank percentage interval', fontsize=10)
    ax7.set_ylabel('Lift', fontsize=10)
    ax7.set_title('Lift cumulative plot', fontsize=12)
    
    # plot lift decay
    #ax8 = plt.subplot(3, 3, 8)
    mean_lift_decay /= ss.get_n_splits()
    ax8.bar(range(10), mean_lift_decay)
    ax8.set_xticks(range(10))
    ax8.set_xticklabels(['%d-%d' %(num-10, num) for num in range(10, 110, 10)])
    for i, v in enumerate(mean_lift_decay):
        ax8.text(i-0.5, v+.05, '%.2f' %v, color='black')
    ax8.set_xlabel('Rank percentage interval', fontsize=10)
    ax8.set_ylabel('Lift', fontsize=10)
    ax8.set_title('Lift decay plot', fontsize=12)
    
    # plot number of calls
    #ax9 = plt.subplot(3, 3, 9)
    ax9.bar(range(len(nb_calls_cv['nb_actions'].values)), nb_calls_cv['nb_actions'].values)
    ax9.set_xticks(range(len(nb_calls_cv['nb_actions'].values)))
    ax9.set_xticklabels(nb_calls_cv['nb_pos_targets'].values)
    for i, v in enumerate(nb_calls_cv['nb_actions'].values):
        ax9.text(i-0.1, v*1.01, '%.0f' %v, color='black')
    ax9.set_xlabel('Number of target to get', fontsize=10)
    ax9.set_ylabel('Number of contacts', fontsize=10)
    ax9.set_title('Number of actions', fontsize=12)
    
    plt.show();
    plt.gcf().clear();
    
    cv_result = {
        'results_cv_targeting': results_cv_targeting,
        'feature_importances': feature_importances,
        'nb_calls_cv': nb_calls_cv
    }
    
    return cv_result


def model_evaluation(_df, _classifier, _features_columns, _id, _target, _prob_name, _ntree_limit=0):
    
    if str(type(_classifier)).startswith("<class 'xgboost"):
            model_type = "xgboost"
    elif str(type(_classifier)).startswith("<class 'lightgbm"):
            model_type = "lightgbm"          
    else:
        print("Model not supported, used function model_evaluation_sk instead")
        return False
    
    nb_calls = pd.DataFrame([],columns=['nb_actions', 'total_population', 'total_pos_targets',
                                        'nb_pos_targets', 'pos_rate', 'Percentage_of_pos_targets_found',
                                        'Percentage_of_Population', 'Lift'])
    
    fig = plt.figure(figsize=(6, 12))
    fig.subplots_adjust(bottom=-0.5, left=-0.5, top=0.5, right=1.5)

    print ('evaluation started')
    plt.gcf().clear()
    ax1 = plt.subplot(3,3,1)
    ax2 = plt.subplot(3,3,2)
    ax3 = plt.subplot(3,3,3)
    ax4 = plt.subplot(3,3,4)
    ax5 = plt.subplot(3,3,5)
    ax6 = plt.subplot(3,3,6)
    ax7 = plt.subplot(3,3,7)
    ax8 = plt.subplot(3,3,8)
    ax9 = plt.subplot(3,3,9)

    valid = _df[np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)

    # modeling#

    valid_X = valid.drop([_id, _target], axis=1)
    valid_Y = np.array(valid[_target].astype(np.uint8))
    
    if hasattr(_classifier,'ntree_limit'):
        probas_ = _classifier.predict_proba(valid_X, ntree_limit=_ntree_limit)
        
    else:
        probas_ = _classifier.predict_proba(valid_X)
    probabilities = pd.DataFrame(data=probas_[:, 1], index=valid_X.index, columns=[_prob_name])
    
    results = valid[[_id, _target]].copy()
    results = results.join(probabilities)
    
    ###############################################################################
    # Plot probability distribution
    #ax1.subplot(3,3,1)
    ax1.hist(probas_[:, 1], range=(0, 1), bins=100, histtype="step")
    ###############################################################################
    # plot proba distribution for both class
    target_probs = pd.DataFrame(valid_Y, columns=['target'])
    target_probs['probs'] = probas_[:, 1]
    #ax2.subplot(3, 3, 2)
    ax2.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100,
             label="Class 1", histtype="step", color='green', density=True)
    ax2.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100,
             label="Class 0", histtype="step", color='red', density=True)
        
    ###############################################################################
    # Plot calibration plots
    fraction_of_positives, mean_predicted_value = calibration_curve(valid_Y, probas_[:, 1], n_bins=20)
    #ax3.subplot(3,3,3)
    ax3.plot(mean_predicted_value, fraction_of_positives, "P-", lw=1)
    
    ###############################################################################
    # plot evals_result
    #ax4.subplot(3, 3, 4)
    ax4.set_xlabel('n estimators', fontsize=10)
    ax4.set_ylabel('ROC AUC', fontsize=10)
    ax4.set_title('ROC/PR through n_estimators', fontsize=12)
    evals_roc = []
    evals_pr = []
    
    for j in range(1, _classifier.n_estimators+1):
        if model_type=="xgboost":
            probas_limit = _classifier.predict_proba(valid_X, ntree_limit=j)
        elif model_type=="lightgbm":
            probas_limit = _classifier.predict_proba(valid_X, num_iteration=j)
        
        fpr_limit, tpr_limit, thresholds_limit = roc_curve(valid_Y, probas_limit[:, 1])
        roc_auc_limit = auc(fpr_limit, tpr_limit)
        evals_roc.append(roc_auc_limit)
        
        precision_limit, recall_limit, thresholds_limit = precision_recall_curve(valid_Y, probas_limit[:, 1])
        pr_auc_limit = auc(recall_limit, precision_limit)
        evals_pr.append(pr_auc_limit)
        
    ax4.plot(range(len(evals_roc)), evals_roc, lw=1, label='ROC AUC')
    ax4.set_ylabel('ROC', color='b')
    ax4b = ax4.twinx()
    ax4b.plot(range(len(evals_pr)), evals_pr, lw=1 , label='PR AUC', color='r')
    ax4b.set_ylabel('PR', color='r')
    
    ###############################################################################
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(valid_Y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    #ax5.subplot(3, 3, 5)
    ax5.plot(fpr, tpr, label='ROC (area = %0.2f)' % (roc_auc), lw=1)
    
    ###############################################################################
    # Compute Precision-Recall curve and area the curve
    precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
    pr_auc = auc(recall, precision)
    #ax6.subplot(3, 3, 6)
    ax6.plot(recall, precision, label='PR (area = %0.2f)' % (pr_auc), lw=1)
    
    ###############################################################################
    # calculate lift related information
    cust_rank = results[[_target, _prob_name]].copy()
    cust_rank = cust_rank.sort_values(by=_prob_name, ascending=False).reset_index(drop=True)
    cust_rank['rank'] = cust_rank.index + 1
    cust_rank['num_pos_target'] = np.cumsum(cust_rank[_target])
    pos_rate = results[_target].mean()
    pos_sum = results[_target].sum()
    
    lift_cums = []
    lift_decays = []
    cum_poss = []
    
    for q in range(10, 110, 10):
        small_q = (q - 10) / 100.0
        big_q = q / 100.0
        if q == 100:
            lift_cum = cust_rank[_target].mean() / pos_rate
            lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) :][_target].mean() / pos_rate
            cum_pos = cust_rank[_target].sum() / pos_sum
        else:
            lift_cum = cust_rank[: int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
            lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) : int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
            cum_pos = cust_rank[: int(big_q * cust_rank.shape[0])][_target].sum() / pos_sum
            
        lift_cums.append(lift_cum)
        lift_decays.append(lift_decay)
        cum_poss.append(cum_pos)

    print ('AUC: %f, lift at 10 percent: %f' % (roc_auc, lift_cums[0]))

    ###############################################################################
    # calculate number of calls
    max_positive_in_fold = cust_rank.num_pos_target.max()
    max_possible_actions = min(max_positive_in_fold,3000)

    nb_calls = cust_rank.copy()
    for i in range(100,int(max_possible_actions), 100):
        nb_calls['to_get_%i' %i] = nb_calls.loc[nb_calls.num_pos_target==i,'rank'].min()

    nb_calls['to_get_all'] = nb_calls.loc[nb_calls.num_pos_target==nb_calls.num_pos_target.max(),'rank'].min()

    nb_calls = nb_calls[[col1 for col1 in nb_calls.columns if col1.startswith('to_get_')]].min()
    nb_calls = pd.DataFrame(nb_calls,columns=['nb_actions'])
    nb_calls['total_population'] = cust_rank.shape[0]
    nb_calls['total_pos_targets'] = cust_rank[_target].sum()
    nb_calls['nb_pos_targets']= list(range(100,int(max_possible_actions), 100))+[int(max_possible_actions)]
    nb_calls['pos_rate'] = nb_calls.nb_pos_targets/nb_calls.nb_actions
    nb_calls['Percentage_of_pos_targets_found'] = nb_calls.nb_pos_targets/nb_calls.total_pos_targets
    nb_calls['Percentage_of_Population'] = nb_calls.nb_actions/nb_calls.total_population
    nb_calls['Lift'] = nb_calls.Percentage_of_pos_targets_found/nb_calls.Percentage_of_Population
    
    for col in nb_calls.columns:
        nb_calls[col] = pd.to_numeric(nb_calls[col])
    nb_calls = nb_calls.reset_index().rename(columns={'index':'objective'})
    nb_calls = nb_calls.groupby('objective').mean().sort_values(by='nb_pos_targets')

    # plot probas for probas
    #plt.subplot(3, 3, 1)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_title('Predicted probas', fontsize=12)
    #plt.legend(loc="lower right")

    # plot probas for both classes
    #plt.subplot(3, 3, 2)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.set_title('Predicted probas for different classes', fontsize=12)
    ax2.legend(loc="lower right")

    # plot the perfectly calibrated curve
    #plt.subplot(3,3,3)
    ax3.plot([0, 1], [0, 1], "k--")
    ax3.set_ylabel("Fraction of positives", fontsize=10)
    ax3.set_xlabel("Mean predicted value", fontsize=10)
    ax3.set_ylim([-0.05, 1.05])
    #plt.legend(loc="lower right")
    ax3.set_title('Calibration plots  (reliability curve)', fontsize=12)

    # plot the averaged ROC curve
    #plt.subplot(3, 3, 5)
    ax5.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax5.set_xlim([-0.05, 1.05])
    ax5.set_ylim([-0.05, 1.05])
    ax5.set_xlabel('FPR', fontsize=10)
    ax5.set_ylabel('TPR', fontsize=10)
    ax5.set_title('ROC', fontsize=12)
    ax5.legend(loc="lower right")
    
    # plot averaged PR curve
    #plt.subplot(3, 3, 6)
    ax6.set_xlabel('Recall', fontsize=10)
    ax6.set_ylabel('Precision', fontsize=10)
    ax6.set_title('Precision-recall', fontsize=12)
    ax6.legend(loc="lower right")
    
    # plot lift cumulative
    #plt.subplot(3, 3, 7)
    ax7.bar(range(10), lift_cums)
    ax7.set_xticks(range(10))
    ax7.set_xticklabels(['0-%d' %(num) for num in range(10, 110, 10)], rotation=-45)
    for i, v in enumerate(zip(lift_cums,cum_poss)):
        ax7.text(i-0.5, v[0]+.05, '%.2f' %(v[0]), color='black')
        ax7.text(i-0.5, v[0]+.4, '(%.0f%%)' %(v[1]*100), color='red', fontsize=8)
    ax7.set_xlabel('Rank percentage interval', fontsize=10)
    ax7.set_ylabel('Lift', fontsize=10)
    ax7.set_title('Lift cumulative plot', fontsize=12)
    
    # plot lift decay
    #plt.subplot(3, 3, 8)
    ax8.bar(range(10), lift_decays)
    ax8.set_xticks(range(10))
    ax8.set_xticklabels(['%d-%d' %(num-10, num) for num in range(10, 110, 10)], rotation=-45)
    for i, v in enumerate(lift_decays):
        ax8.text(i-0.5, v+.05, '%.2f' %v, color='black')
    ax8.set_xlabel('Rank percentage interval', fontsize=10)
    ax8.set_ylabel('Lift', fontsize=10)
    ax8.set_title('Lift decay plot', fontsize=12)
    
    # plot number of calls
    #plt.subplot(3, 3, 9)
    ax9.bar(range(len(nb_calls['nb_actions'].values)), nb_calls['nb_actions'].values)
    ax9.set_xticks(range(len(nb_calls['nb_actions'].values)))
    ax9.set_xticklabels(nb_calls['nb_pos_targets'].values, rotation=90)
    for i, v in enumerate(nb_calls['nb_actions'].values):
        ax9.text(i, v*1.02, '%.0f' %v, color='black', rotation=90, horizontalalignment='center', verticalalignment='bottom')
    ax9.set_xlabel('Number of target to get', fontsize=10)
    ax9.set_ylabel('Number of contacts', fontsize=10)
    ax9.set_title('Number of actions', fontsize=12)
    
    plt.show();
    plt.gcf().clear();
    
    result_resume = {
        'results_targeting': results,
        'nb_calls': nb_calls
    }
    return result_resume


def plot_imp(feature_importances, top_n):
    feature_importances['abs_imp'] = feature_importances['mean'].apply(lambda x: abs(x))
    feature_importances_sort = feature_importances.sort_values(by='abs_imp',ascending=False)
    feature_importances_sort['relative_imp'] = 100.0 * (feature_importances_sort['abs_imp'] / feature_importances_sort['abs_imp'].max())
    feature_importances_sort = feature_importances_sort[::-1].reset_index(drop=True)
    feature_importances_sort = feature_importances_sort.tail(top_n).reset_index(drop=True)

    plt.figure(figsize=(10, int(0.4 * top_n)))
    plt.title("Feature importances for Model")
    plt.barh(feature_importances_sort.index, feature_importances_sort['relative_imp'],
             color='#348ABD', align="center", lw='3', edgecolor='#348ABD', alpha=0.6)
    plt.yticks(feature_importances_sort.index, feature_importances_sort['feature'], fontsize=12,)
    plt.ylim([-1, feature_importances_sort.index.max()+1])
    plt.xlim([0, feature_importances_sort['relative_imp'].max()*1.1])
    plt.show()
    

def plot_imp_model(model, top_n=20, weight='gain'):
    
    if str(type(model)).startswith("<class 'xgboost"):
            model_type = "xgboost"
    elif str(type(model)).startswith("<class 'lightgbm"):
            model_type = "lightgbm"          
    else:
        print("Model not supported, used function plot_imp_model_sk instead")
        return False
    
    feature_importances_data = []
    
    if model_type == "xgboost":
        for feature_name, feature_importance in model.get_booster().get_score(fmap='',importance_type=weight).items():
            feature_importances_data.append({
                    'feature': feature_name,
                    'importance': feature_importance
                })

        feature_importances = pd.DataFrame(feature_importances_data)

    if model_type == "lightgbm":
        for feature_name, feature_importance in zip(model.booster_.feature_name(),
                                                    model.booster_.feature_importance(importance_type=weight)):
            feature_importances_data.append({
                    'feature': feature_name,
                    'importance': feature_importance
                })

        feature_importances = pd.DataFrame(feature_importances_data)

    feature_importances['abs_imp'] = feature_importances['importance'].apply(lambda x: abs(x))
    feature_importances_sort = feature_importances.sort_values(by='abs_imp',ascending=False)
    feature_importances_out = feature_importances_sort.copy()
    feature_importances_sort['relative_imp'] = 100.0 * (feature_importances_sort['abs_imp'] / feature_importances_sort['abs_imp'].max())
    feature_importances_sort = feature_importances_sort[::-1].reset_index(drop=True)
    feature_importances_sort = feature_importances_sort.tail(top_n).reset_index(drop=True)

    plt.figure(figsize=(10, int(0.4 * top_n)))
    plt.title("Feature importances for Model")
    plt.barh(feature_importances_sort.index, feature_importances_sort['relative_imp'],
             color='#348ABD', align="center", lw='3', edgecolor='#348ABD', alpha=0.6)
    plt.yticks(feature_importances_sort.index, feature_importances_sort['feature'], fontsize=10)
    plt.ylim([-1, feature_importances_sort.index.max()+1])
    plt.xlim([0, feature_importances_sort['relative_imp'].max()*1.1])
    plt.show()
    
    return feature_importances_out


def roc_lift(y_true, y_pred):
    results_roc=pd.DataFrame([])
    results_lift=pd.DataFrame(range(0,101), columns=['quantiles'])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Compute Lift curve
    sorted_proba = np.array(list(reversed(np.argsort(y_pred))))
    xtestshape0=y_true.count().astype(int)
    y_test=y_true
    centile = xtestshape0//100
    positives = sum(y_test)
    lift = [0]
    for q in xrange(1,101):
        if q == 100:
            tp = sum(np.array(y_test)[sorted_proba[(q-1)*centile:xtestshape0]])
        else:
            tp = sum(np.array(y_test)[sorted_proba[(q-1)*centile:q*centile]])
        lift.append(lift[q-1]+100*tp/float(positives))
    quantiles = range(0,101)

    print("Model auc: %f, lift at 10: %f" %(roc_auc, lift[10]/10.))
    

def estimate_benefit(_results, _additional_info, _cost_matrix, _gain_name, _p, _id_name, _target_name, _proba_name, _ntrials=100):
    
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_
    
    def cost_attribution(x):
        if (x[0]==1 and x[1]==0):
            return _cost_matrix[0]
        else:
            return _cost_matrix[1]
    
    benefit = _results['results_targeting'].copy()
    benefit = benefit.rename(columns={_target_name:'RESPONSE',_proba_name:'PROBABILITY'})
    benefit = benefit.sort_values(by='PROBABILITY', ascending=False).reset_index(drop=True)
    benefit['SCORE'] = np.round(benefit['PROBABILITY'],2)
    benefit = benefit[[_id_name,'SCORE','PROBABILITY','RESPONSE']]
    benefit = benefit.merge(_additional_info[[_id_name,_gain_name]], on=_id_name, how='left')
    benefit = benefit.rename(columns={_gain_name:'PREMIUM'})
    
    #APE LOST IF WE DO NOTHING
    benefit['APE_LOST'] = benefit['PREMIUM']* benefit['RESPONSE']
    
    benefit_all = pd.DataFrame([])
    
    for n in range(_ntrials):
        
        for bound in ['LOWER','MEAN','UPPER']:
            benefit['CONVINCED_%s' %bound] = 0

        for score in benefit.SCORE.unique():
            impacted_pol = benefit.loc[benefit.SCORE==score].shape[0]
            for i, bound in enumerate(['LOWER','MEAN','UPPER']):
                benefit.loc[benefit.SCORE==score,'CONVINCED_%s' %bound] = np.random.choice([0, 1], size=(impacted_pol,),
                                                                                           p=[1-_p[i], _p[i]])

        for bound in ['LOWER','MEAN','UPPER']:
            benefit['COST_%s' %bound] = benefit[['RESPONSE','CONVINCED_%s' %bound]].apply(lambda x : cost_attribution(x), axis=1)
            benefit['EARNING_%s' %bound] = (benefit['RESPONSE']*benefit['CONVINCED_%s' %bound]*benefit['PREMIUM'])
            benefit['PAYOUT_%s' %bound] = (benefit['RESPONSE']*benefit['CONVINCED_%s' %bound]*benefit['PREMIUM'])-benefit['COST_%s' %bound]

        #AGGREGATE RESULTS AT PERCENTILE LEVEL
        benefit2 = benefit.copy()
        benefit2['RANK'] = benefit2.index+1
        benefit2['PERCENTILE'] = pd.qcut(benefit2.RANK, q=100, labels=False)+1
        
        agg_function = {'ID_POLICY_NO':'count','RANK':'max','SCORE':'min','PROBABILITY':'min',
                        'RESPONSE':'sum','PREMIUM':'sum','APE_LOST':'sum',
                        'CONVINCED_LOWER':'sum','CONVINCED_MEAN':'sum','CONVINCED_UPPER':'sum',
                        'COST_LOWER':'sum','COST_MEAN':'sum','COST_UPPER':'sum',
                        'EARNING_LOWER':'sum','EARNING_MEAN':'sum','EARNING_UPPER':'sum',
                        'PAYOUT_LOWER':'sum','PAYOUT_MEAN':'sum','PAYOUT_UPPER':'sum'
                       }
        
        benefit2 = benefit2.groupby('PERCENTILE', as_index=False).agg(agg_function)
        
        for bound in ['LOWER','MEAN','UPPER']:
            benefit2['CUM_PAYOUT_%s' %bound] = np.cumsum(benefit2['PAYOUT_%s' %bound])

        benefit2['PREMIUM'] = np.round(benefit2['PREMIUM'], 0)
        benefit2['APE_LOST'] = np.round(benefit2['APE_LOST'], 0)

        benefit2 = benefit2.rename(columns={'ID_POLICY_NO':'NB_POLICIES','RANK':'CUMULATIVE_POLICIES',
                                            'SCORE':'THRESHOLD','PREMIUM':'TOTAL_APE',
                                           })
        benefit2['TRIAL'] = n+1
        
        #COLLECT RESULTS OF EACH TRIAL
        benefit_all = benefit_all.append(benefit2)
    
    #KEEP THIS DATA FOR OPTIMUM TABLE ADDITIONAL INFO
    benefit_all_raw = benefit_all.copy()
    
    agg_benefit_all = {col:[percentile(10),percentile(50),percentile(90)] 
                       for col in benefit_all.columns if any(x in col for x in ['_LOWER','_MEAN','_UPPER'])}
    
    for col in benefit_all.columns:
        if (col!='PERCENTILE') & (col not in agg_benefit_all.keys()):
            agg_benefit_all[col] = 'mean'
            
    benefit_all = benefit_all.groupby('PERCENTILE', as_index=False).agg(agg_benefit_all)
    
    #for bound in ['LOWER','MEAN','UPPER']:
    #    for p in [10,50,90]:
    #        benefit_all[('CUM_PAYOUT_%s' %bound, 'percentile_%i' %p)] = np.cumsum(benefit_all[('PAYOUT_%s' %bound, 'percentile_%i' %p)])
    
    benefit_all = benefit_all[['PERCENTILE','NB_POLICIES','CUMULATIVE_POLICIES','THRESHOLD','PROBABILITY','RESPONSE','TOTAL_APE',
                               'APE_LOST','CONVINCED_LOWER','CONVINCED_MEAN','CONVINCED_UPPER','COST_LOWER','COST_MEAN','COST_UPPER',
                               'EARNING_LOWER','EARNING_MEAN','EARNING_UPPER',
                               'PAYOUT_LOWER','PAYOUT_MEAN','PAYOUT_UPPER','CUM_PAYOUT_LOWER','CUM_PAYOUT_MEAN','CUM_PAYOUT_UPPER']]
    
    optimum_lower = benefit_all.iloc[benefit_all['CUM_PAYOUT_LOWER']['percentile_50'].idxmax(),0]
    optimum_mean = benefit_all.iloc[benefit_all['CUM_PAYOUT_MEAN']['percentile_50'].idxmax(),0]
    optimum_upper = benefit_all.iloc[benefit_all['CUM_PAYOUT_UPPER']['percentile_50'].idxmax(),0]
    
    optimum_lower_po = benefit_all['CUM_PAYOUT_LOWER']['percentile_50'].max()/1000
    optimum_mean_po = benefit_all['CUM_PAYOUT_MEAN']['percentile_50'].max()/1000
    optimum_upper_po = benefit_all['CUM_PAYOUT_UPPER']['percentile_50'].max()/1000

    optimum_points = {'LOWER':optimum_lower,'MEAN':optimum_mean,'UPPER':optimum_upper}

    optimum = pd.DataFrame([])
    
    agg_optimum = {col:'sum' for col in benefit_all if col[0] not in ['CONVINCED_LOWER','CONVINCED_MEAN','CONVINCED_UPPER',
                                                                      'COST_LOWER','COST_MEAN','COST_UPPER',
                                                                      'EARNING_LOWER','EARNING_MEAN','EARNING_UPPER',
                                                                      'CUMULATIVE_POLICIES','PERCENTILE','PROBABILITY','THRESHOLD']}
    agg_optimum[('THRESHOLD', 'mean')]='min'
    agg_optimum[('PROBABILITY', 'mean')]='min'
    agg_optimum[('PERCENTILE', '')]='max'
    agg_optimum[('CUM_PAYOUT_LOWER', 'percentile_10')]='last'
    agg_optimum[('CUM_PAYOUT_LOWER', 'percentile_50')]='last'
    agg_optimum[('CUM_PAYOUT_LOWER', 'percentile_90')]='last'
    agg_optimum[('CUM_PAYOUT_MEAN', 'percentile_10')]='last'
    agg_optimum[('CUM_PAYOUT_MEAN', 'percentile_50')]='last'
    agg_optimum[('CUM_PAYOUT_MEAN', 'percentile_90')]='last'
    agg_optimum[('CUM_PAYOUT_UPPER', 'percentile_10')]='last'
    agg_optimum[('CUM_PAYOUT_UPPER', 'percentile_50')]='last'
    agg_optimum[('CUM_PAYOUT_UPPER', 'percentile_90')]='last'
    
    for bound in ['LOWER','MEAN','UPPER']:

        temp = benefit_all.loc[benefit_all.PERCENTILE<=optimum_points[bound]].copy().reset_index(drop=True)
        temp['SCENARIO'] = bound
        temp = temp.groupby(['SCENARIO'], as_index=False).agg(agg_optimum)
        temp = temp[[('SCENARIO', ''), ('PERCENTILE', ''), ('NB_POLICIES', 'mean'), ('THRESHOLD', 'mean'), 
                     ('PROBABILITY', 'mean'), ('RESPONSE', 'mean'), ('TOTAL_APE', 'mean'), ('APE_LOST', 'mean')]+
                    sorted([col for col in temp.columns if bound in col[0]])
                   ]
        temp = temp[[('SCENARIO', '')]+[col for col in benefit_all.columns if col in temp.columns]]
        
        addtional_info = benefit_all_raw.loc[benefit_all_raw.PERCENTILE<=optimum_points[bound]].copy().reset_index(drop=True)
        addtional_info['SCENARIO'] = bound
        addtional_info = addtional_info.groupby(['SCENARIO','TRIAL'], as_index=False).sum()
        addtional_info = addtional_info[['SCENARIO']+
                                        [col for col in addtional_info if any(x in col for x in ['CONVINCED','COST','EARNING'])]]

        agg_addtional_info = {col:[percentile(10),percentile(50),percentile(90)]
                              for col in addtional_info.columns if any(x in col for x in ['_LOWER','_MEAN','_UPPER'])}

        addtional_info = addtional_info.groupby(['SCENARIO'], as_index=False).agg(agg_addtional_info)
        addtional_info = addtional_info[[('SCENARIO', '')]+sorted([col for col in addtional_info.columns if bound in col[0]])]
        
        temp = temp.merge(addtional_info, on=[('SCENARIO','')])
        temp = temp.rename(columns={col[0]:col[0].replace('_'+bound,'') for col in temp.columns})
        optimum = optimum.append(temp)
    
    optimum = optimum.drop(['PAYOUT'], axis=1, level=0)
    #return benefit_all, optimum, temp
    
    #PLOT FIGURE
    fig = plt.figure(figsize=(4, 8))
    fig.subplots_adjust(bottom=-0.5, left=-0.5, top=0.5, right=1.5)
    
    plt.subplot(2,1,1)
    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_LOWER']['percentile_50'], color=sns.color_palette()[5])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_LOWER']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_LOWER']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[5])

    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_MEAN']['percentile_50'], color=sns.color_palette()[4])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_MEAN']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_MEAN']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[4]) 

    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_UPPER']['percentile_50'], color=sns.color_palette()[3])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_UPPER']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_UPPER']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[3]) 

    plt.vlines(optimum_lower, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_LOWER.max().max(), linestyles='--',color=sns.color_palette()[5])
    plt.vlines(optimum_mean, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_MEAN.max().max(), linestyles='--',color=sns.color_palette()[4])
    plt.vlines(optimum_upper, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_UPPER.max().max(), linestyles='--',color=sns.color_palette()[3])
    leg = plt.legend(('Payout %i%% convinced, max: %.1f K$ @ %i %% of pop. ' %(_p[0]*100,optimum_lower_po,optimum_lower), 
                      'Payout %i%% convinced, max: %.1f K$ @ %i %% of pop. ' %(_p[1]*100,optimum_mean_po,optimum_mean), 
                      'Payout %i%% convinced, max: %.1f K$ @ %i %% of pop. ' %(_p[2]*100,optimum_upper_po,optimum_upper)),
                      frameon=True)
    leg.get_frame().set_edgecolor('k') 
    plt.xlabel('Percentage of Population') 
    plt.ylabel('$') 
    plt.title("Payout as a Function of Threshold and Chances to Retain")
    
    #ZOOM IN INTERESTING ZONE
    plt.subplot(2,1,2)
    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_LOWER']['percentile_50'], color=sns.color_palette()[5])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_LOWER']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_LOWER']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[5])

    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_MEAN']['percentile_50'], color=sns.color_palette()[4])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_MEAN']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_MEAN']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[4]) 

    plt.plot(benefit_all.PERCENTILE, benefit_all['CUM_PAYOUT_UPPER']['percentile_50'], color=sns.color_palette()[3])
    plt.fill_between(benefit_all.PERCENTILE, 
                     benefit_all['CUM_PAYOUT_UPPER']['percentile_10'], 
                     benefit_all['CUM_PAYOUT_UPPER']['percentile_90'], 
                     alpha=0.5, linewidth=0, color=sns.color_palette()[3]) 

    plt.vlines(optimum_lower, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_LOWER.max().max(), linestyles='--',color=sns.color_palette()[5])
    plt.vlines(optimum_mean, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_MEAN.max().max(), linestyles='--',color=sns.color_palette()[4])
    plt.vlines(optimum_upper, benefit_all['CUM_PAYOUT_LOWER'].min().min(), 
               benefit_all.CUM_PAYOUT_UPPER.max().max(), linestyles='--',color=sns.color_palette()[3])
    
    po_low_p10 = optimum.loc[optimum['SCENARIO']=='LOWER','CUM_PAYOUT'].values[0][0]/1000
    po_low_p90 = optimum.loc[optimum['SCENARIO']=='LOWER','CUM_PAYOUT'].values[0][2]/1000
    po_med_p10 = optimum.loc[optimum['SCENARIO']=='MEAN','CUM_PAYOUT'].values[0][0]/1000
    po_med_p90 = optimum.loc[optimum['SCENARIO']=='MEAN','CUM_PAYOUT'].values[0][2]/1000
    po_upp_p10 = optimum.loc[optimum['SCENARIO']=='UPPER','CUM_PAYOUT'].values[0][0]/1000
    po_upp_p90 = optimum.loc[optimum['SCENARIO']=='UPPER','CUM_PAYOUT'].values[0][2]/1000
    
    leg = plt.legend(('Payout %i%% convinced, max: %.1f K$ [%.1f K$-%.1f K$]' %(_p[0]*100,optimum_lower_po,po_low_p10,po_low_p90), 
                      'Payout %i%% convinced, max: %.1f K$ [%.1f K$-%.1f K$]' %(_p[1]*100,optimum_mean_po,po_med_p10,po_med_p90), 
                      'Payout %i%% convinced, max: %.1f K$ [%.1f K$-%.1f K$]' %(_p[2]*100,optimum_upper_po,po_upp_p10,po_upp_p90)),
                      frameon=True)
    leg.get_frame().set_edgecolor('k')
    plt.ylim(0, benefit_all.CUM_PAYOUT_UPPER.max().max())
    plt.xlim(0, benefit_all.loc[benefit_all['CUM_PAYOUT_UPPER']['percentile_90']>0,'PERCENTILE'].values[-1])
    plt.xlabel('Percentage of Population') 
    plt.ylabel('$') 
    plt.title("Zoom in Payout")    
    plt.show();        
    
    return benefit_all, optimum

#for scikit learn models

def run_cross_validation_sk(_df, _classifier, _features_columns, _id, _target, _prob_name,
                            _n_iter=5, _test_size=.3 ,_random_state=0):
    
    # cross validation type can be changed here
    ss = ShuffleSplit(n_splits=_n_iter, test_size=_test_size, random_state=_random_state)
    results_cv_targeting = pd.DataFrame([], columns=[_id, _target, 'fold', _prob_name])

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    mean_precision = 0.0
    mean_recall = np.linspace(0, 1, 100)
    
    mean_lift = 0.0
    mean_lift_decay = 0.0
    mean_cum_pos = 0.0
    
    nb_calls_cv = pd.DataFrame([],columns=['nb_actions', 'total_population', 'total_pos_targets', 
                                           'nb_pos_targets', 'pos_rate', 'Percentage_of_pos_targets_found', 
                                           'Percentage_of_Population', 'Lift'])
    feature_importances = pd.DataFrame([], columns=['feature', 'importance', 'fold'])

    fig = plt.figure(figsize=(6, 12))
    fig.subplots_adjust(bottom=-0.5, left=-0.5, top=0.5, right=1.5)

    print ('cross validation started')
    plt.gcf().clear()
    
    trials_results = {}

    for i, (train_index, valid_index) in enumerate(ss.split(_df[_id].unique())):
        customer_id = _df[_id].unique().copy()
        shuffled_customer_id = np.array(sorted(customer_id, key=lambda k: random.random()))
        train_customer_id = shuffled_customer_id[train_index]
        valid_customer_id = shuffled_customer_id[valid_index]
        
        train = _df.loc[_df[_id].isin(train_customer_id), 
                        np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)
        valid = _df.loc[_df[_id].isin(valid_customer_id), 
                        np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)
        
        temp = valid[[_id, _target]].copy()
        temp['fold'] = i

        # modeling#
        train_X = train.drop([_id, _target], axis=1)
        valid_X = valid.drop([_id, _target], axis=1)

        train_Y = np.array(train[_target].astype(np.uint8))
        valid_Y = np.array(valid[_target].astype(np.uint8))
        
        probas_ = _classifier.fit(train_X, train_Y).predict_proba(valid_X)
        
        probabilities = pd.DataFrame(data=probas_[:, 1], index=valid_X.index, columns=[_prob_name])

        temp = temp.join(probabilities, how='left')
        results_cv_targeting = results_cv_targeting.append(temp)


        ###############################################################################
        # Plot probability distribution
        plt.subplot(3,3,1)
        plt.hist(probas_[:, 1], range=(0, 1), bins=100, label="fold %d" % (i), histtype="step")
        
        
        ###############################################################################
        # plot proba distribution for both class
        target_probs = pd.DataFrame(valid_Y, columns=['target'])
        target_probs['probs'] = probas_[:, 1]
        plt.subplot(3, 3, 2)
        if i==0:
            plt.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100, 
                     label="Class 1", histtype="step", color='green', density=True)
            plt.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100, 
                     label="Class 0", histtype="step", color='red', density=True)
        else:
            plt.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100, 
                     histtype="step", color='green', density=True)
            plt.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100, 
                     histtype="step", color='red', density=True)


        ###############################################################################
        # Plot calibration plots
        fraction_of_positives, mean_predicted_value = calibration_curve(valid_Y, probas_[:, 1], n_bins=20)
        plt.subplot(3,3,3)
        plt.plot(mean_predicted_value, fraction_of_positives, "P-", label="fold %d" % (i), lw=1)


        ###############################################################################
        # Compute the curve metrics and thresholds
        precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
        # Compute the F1 score from precision and recall
        # Don't need to warn for F, precision/recall would have warned
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = 1 ** 2
            f_score = ((1 + beta) * precision * recall /(beta * precision + recall))

        # Ensure thresholds ends at 1
        thresholds = np.append(thresholds, 1)
        
        # Compute the queue rate
        queue_rate = np.array([(probas_[:, 1] >= threshold).mean() for threshold in thresholds])

        trials_results[i] = {
            'thresholds': thresholds,
            'precision': precision,
            'recall': recall,
            'fscore': f_score,
            'queue_rate': queue_rate
        }
        

        ###############################################################################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(valid_Y, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)

        plt.subplot(3, 3, 5)
        plt.plot(fpr, tpr, label='ROC fold %d (area = %0.2f)' % (i, roc_auc), lw=1)

        
        ###############################################################################
        # Compute Precision-Recall curve and area the curve
        precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
        mean_precision += interp(mean_recall, recall[::-1], precision[::-1])
        pr_auc = auc(recall, precision)

        plt.subplot(3, 3, 6)
        plt.plot(recall, precision, label='PR fold %d (area = %0.2f)' % (i, pr_auc), lw=1)


        ###############################################################################
        # calculate lift related information
        cust_rank = temp[[_target, _prob_name]].copy()
        cust_rank = cust_rank.sort_values(by=_prob_name, ascending=False).reset_index(drop=True)
        cust_rank['rank'] = cust_rank.index + 1
        cust_rank['num_pos_target'] = np.cumsum(cust_rank[_target])
        pos_rate = temp[_target].mean()
        pos_sum = temp[_target].sum()

        lift_cums = []
        lift_decays = []
        cum_poss = []
        
        for q in range(10, 110, 10):
            small_q = (q - 10) / 100.0
            big_q = q / 100.0
            if q == 100:
                lift_cum = cust_rank[_target].mean() / pos_rate
                lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) :][_target].mean() / pos_rate
                cum_pos = cust_rank[_target].sum() / pos_sum
            else:
                lift_cum = cust_rank[: int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
                lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) : int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
                cum_pos = cust_rank[: int(big_q * cust_rank.shape[0])][_target].sum() / pos_sum
            
            lift_cums.append(lift_cum)
            lift_decays.append(lift_decay)
            cum_poss.append(cum_pos)

        print ('shuffle: %i, AUC: %f, lift at 10 percent: %f' % (i, roc_auc, lift_cums[0]))
        mean_lift += np.array(lift_cums)
        mean_lift_decay += np.array(lift_decays)
        mean_cum_pos += np.array(cum_poss)

        ###############################################################################
        # calculate number of calls
        max_positive_in_fold = cust_rank.num_pos_target.max()
        max_possible_actions = min(max_positive_in_fold,3000)

        nb_calls = cust_rank.copy()
        for i in range(100,int(max_possible_actions), 100):
            nb_calls['to_get_%i' %i] = nb_calls.loc[nb_calls.num_pos_target==i,'rank'].min()

        nb_calls['to_get_all'] = nb_calls.loc[nb_calls.num_pos_target==nb_calls.num_pos_target.max(),'rank'].min()

        nb_calls = nb_calls[[col1 for col1 in nb_calls.columns if col1.startswith('to_get_')]].min()
        nb_calls = pd.DataFrame(nb_calls,columns=['nb_actions'])
        nb_calls['total_population'] = cust_rank.shape[0]
        nb_calls['total_pos_targets'] = cust_rank[_target].sum()
        nb_calls['nb_pos_targets']= range(100,int(max_possible_actions), 100)+[int(max_possible_actions)]
        nb_calls['pos_rate'] = nb_calls.nb_pos_targets/nb_calls.nb_actions
        nb_calls['Percentage_of_pos_targets_found'] = nb_calls.nb_pos_targets/nb_calls.total_pos_targets
        nb_calls['Percentage_of_Population'] = nb_calls.nb_actions/nb_calls.total_population
        nb_calls['Lift'] = nb_calls.Percentage_of_pos_targets_found/nb_calls.Percentage_of_Population

        nb_calls_cv = nb_calls_cv.append(nb_calls)
        
        ###############################################################################
        feature_importances_data = []
        features = train_X.columns
        if str(type(_classifier))=="<class 'sklearn.ensemble.forest.RandomForestClassifier'>":
            for feature_name, feature_importance in zip(_features_columns,_classifier.feature_importances_):
                feature_importances_data.append({
                        'feature': feature_name,
                        'importance': feature_importance
                    })
        else :
            for feature_name in _features_columns:
                feature_importances_data.append({
                        'feature': feature_name,
                        'importance': 0
                    })
            
        temp = pd.DataFrame(feature_importances_data)
        temp['fold'] = i
        feature_importances = feature_importances.append(temp)

    #correct nb_calls_cv because sometime index has more rows than others
    nb_calls_cv = nb_calls_cv.reset_index()
    nb_calls_cv = nb_calls_cv.rename(columns={'index':'objective'})
    nb_calls_cv['undesired_index'] = nb_calls_cv.groupby(['objective'])['nb_actions'].transform('count')
    nb_calls_cv = nb_calls_cv.loc[nb_calls_cv.undesired_index==_n_iter].reset_index(drop=True)
    nb_calls_cv = nb_calls_cv.drop('undesired_index', axis=1)
    nb_calls_cv = nb_calls_cv.set_index('objective')
    
    for col in nb_calls_cv.columns:
        nb_calls_cv[col] = pd.to_numeric(nb_calls_cv[col])
    nb_calls_cv = nb_calls_cv.reset_index().groupby('objective').mean().sort_values(by='nb_pos_targets')
    
    results_cv_targeting = results_cv_targeting.reset_index(drop=True)
    
    feature_importances = feature_importances.groupby('feature')['importance'].agg([np.mean, np.std])
    feature_importances = feature_importances.sort_values(by='mean')
    feature_importances = feature_importances.reset_index()
    
    # Discrimination threshold measures
    # Compute maximum number of uniform thresholds across all trials
    n_thresholds = np.array([len(trials_results[t]['thresholds']) for t in trials_results]).min()
    thresholds_ = np.linspace(0.0, 1.0, num=n_thresholds)

    metrics = [metric for metric in trials_results[0].keys() if metric!='thresholds']
    uniform_metrics = defaultdict(list)

    for trial in trials_results:
        rows = defaultdict(list)
        for t in thresholds_:
            idx = bisect.bisect_left(trials_results[trial]['thresholds'], t)
            for metric in metrics:
                rows[metric].append(trials_results[trial][metric][idx])

        for metric, row in rows.items():
            uniform_metrics[metric].append(row)


    # Convert metrics to metric arrays
    uniform_metrics = {
        metric: np.array(values)
        for metric, values in uniform_metrics.items()
        }

    # Perform aggregation and store cv_scores_
    quantiles = [0.1,0.5,0.9]
    cv_scores_ = {}
    for metric, values in uniform_metrics.items():
        # Compute the lower, median, and upper plots
        lower, median, upper = mstats.mquantiles(values, prob=quantiles, axis=0)
        # Store the aggregates in cv scores
        cv_scores_[metric] = median
        cv_scores_["{}_lower".format(metric)] = lower
        cv_scores_["{}_upper".format(metric)] = upper
    
    
    # plot probas for probas
    plt.subplot(3, 3, 1)
    plt.ylabel('Count', fontsize=10)
    plt.title('Predicted probas', fontsize=12)
    plt.legend(loc="lower right")


    # plot probas for both classes
    plt.subplot(3, 3, 2)
    plt.ylabel('Density', fontsize=10)
    plt.title('Predicted probas for different classes', fontsize=12)
    plt.legend(loc="lower right")


    # plot the perfectly calibrated curve
    plt.subplot(3,3,3)
    plt.plot([0, 1], [0, 1], "k--")
    plt.ylabel("Fraction of positives", fontsize=10)
    plt.xlabel("Mean predicted value", fontsize=10)
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title('Calibration plots  (reliability curve)', fontsize=12)

    # plot discrimination threshold
    plt.subplot(3, 3, 4)
    color_values = ['g','mediumvioletred','b','r']
    for idx, metric in enumerate(metrics):
        color = color_values[idx]
        # Make the label pretty
        label = metric.replace("_", " ")
        # Draw the metric values
        plt.plot(thresholds_, cv_scores_[metric], color=color, label=label)
        # Draw the upper and lower bounds
        lower = cv_scores_["{}_lower".format(metric)]
        upper = cv_scores_["{}_upper".format(metric)]
        plt.fill_between(thresholds_, upper, lower, alpha=0.35, linewidth=0, color=color)
        # Annotate the graph with the maximizing value
        if metric == 'fscore':
            argmax = cv_scores_[metric].argmax()
            threshold = thresholds_[argmax]
            plt.axvline(threshold, ls='--', c='k', lw=1,
                        label="$t_{}={:0.2f}$".format(metric[0], threshold))

    # Set the title of the threshold visualization
    plt.title("Threshold Plot")
    plt.legend(frameon=True, loc='best')
    plt.xlabel('Discrimination Threshold')
    plt.ylabel('Score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # plot the averaged ROC curve
    plt.subplot(3, 3, 5)
    mean_tpr /= ss.get_n_splits()
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('ROC', fontsize=12)
    plt.legend(loc="lower right")


    # plot averaged PR curve
    plt.subplot(3, 3, 6)
    mean_precision /= ss.get_n_splits()
    mean_pr_auc = auc(mean_recall, mean_precision)
    plt.plot(mean_recall, mean_precision, 'k--', label='Mean PR (area = %0.2f)' % mean_pr_auc, lw=2)
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title('Precision-recall', fontsize=12)
    plt.legend(loc="lower right")
    
    # plot lift cumulative
    plt.subplot(3, 3, 7)
    mean_lift /= ss.get_n_splits()
    mean_cum_pos /= ss.get_n_splits()
    plt.bar(range(10), mean_lift)
    plt.xticks(range(10), ['0-%d' %(num) for num in range(10, 110, 10)])
    for i, v in enumerate(zip(mean_lift,mean_cum_pos)):
        plt.text(i-0.5, v[0]+.05, '%.2f' %(v[0]), color='black')
        plt.text(i-0.5, v[0]+.4, '(%.0f%%)' %(v[1]*100), color='red', fontsize=8)
    plt.xlabel('Rank percentage interval', fontsize=10)
    plt.ylabel('Lift', fontsize=10)
    plt.title('Lift cumulative plot', fontsize=12)
    
    # plot lift decay
    plt.subplot(3, 3, 8)
    mean_lift_decay /= ss.get_n_splits()
    plt.bar(range(10), mean_lift_decay)
    plt.xticks(range(10), ['%d-%d' %(num-10, num) for num in range(10, 110, 10)])
    for i, v in enumerate(mean_lift_decay):
        plt.text(i-0.5, v+.05, '%.2f' %v, color='black')
    plt.xlabel('Rank percentage interval', fontsize=10)
    plt.ylabel('Lift', fontsize=10)
    plt.title('Lift decay plot', fontsize=12)
    
    # plot number of calls
    plt.subplot(3, 3, 9)
    plt.bar(range(len(nb_calls_cv['nb_actions'].values)), nb_calls_cv['nb_actions'].values)
    plt.xticks(range(len(nb_calls_cv['nb_actions'].values)), nb_calls_cv['nb_pos_targets'].values)
    for i, v in enumerate(nb_calls_cv['nb_actions'].values):
        plt.text(i-0.1, v*1.01, '%.0f' %v, color='black')
    plt.xlabel('Number of target to get', fontsize=10)
    plt.ylabel('Number of contacts', fontsize=10)
    plt.title('Number of actions', fontsize=12)
    
    plt.show();
    plt.gcf().clear();
    
    cv_result = {
        'results_cv_targeting': results_cv_targeting,
        'feature_importances': feature_importances,
        'nb_calls_cv': nb_calls_cv
    }
    
    return cv_result

def plot_imp_model_sk(model,features_columns, top_n=20, weight='gain'):
    
    feature_importances_data = []
    for feature_name, feature_importance in zip(features_columns,model.feature_importances_):
        feature_importances_data.append({
                'feature': feature_name,
                'importance': feature_importance
            })

    feature_importances = pd.DataFrame(feature_importances_data)

    feature_importances['abs_imp'] = feature_importances['importance'].apply(lambda x: abs(x))
    feature_importances_sort = feature_importances.sort_values(by='abs_imp',ascending=False)
    feature_importances_out = feature_importances_sort.copy()
    feature_importances_sort['relative_imp'] = 100.0 * (feature_importances_sort['abs_imp'] / feature_importances_sort['abs_imp'].max())
    feature_importances_sort = feature_importances_sort[::-1].reset_index(drop=True)
    feature_importances_sort = feature_importances_sort.tail(top_n).reset_index(drop=True)

    plt.figure(figsize=(10, int(0.4 * top_n)))
    plt.title("Feature importances for Model")
    plt.barh(feature_importances_sort.index, feature_importances_sort['relative_imp'],
             color='#348ABD', align="center", lw='3', edgecolor='#348ABD', alpha=0.6)
    plt.yticks(feature_importances_sort.index, feature_importances_sort['feature'], fontsize=10)
    plt.ylim([-1, feature_importances_sort.index.max()+1])
    plt.xlim([0, feature_importances_sort['relative_imp'].max()*1.1])
    plt.show()
    
    return feature_importances_out

def model_evaluation_sk(_df, _classifier, _features_columns, _id, _target, _prob_name):
    
    nb_calls = pd.DataFrame([],columns=['nb_actions', 'total_population', 'total_pos_targets',
                                        'nb_pos_targets', 'pos_rate', 'Percentage_of_pos_targets_found',
                                        'Percentage_of_Population', 'Lift'])
    
    fig = plt.figure(figsize=(6, 12))
    fig.subplots_adjust(bottom=-0.5, left=-0.5, top=0.5, right=1.5)

    print ('evaluation started')
    plt.gcf().clear()

    valid = _df[np.concatenate([[_id], _features_columns, [_target]],axis=0)].copy().reset_index(drop=True)

    # modeling#

    valid_X = valid.drop([_id, _target], axis=1)
    valid_Y = np.array(valid[_target].astype(np.uint8))
    
    probas_ = _classifier.predict_proba(valid_X)
    probabilities = pd.DataFrame(data=probas_[:, 1], index=valid_X.index, columns=[_prob_name])
    
    results = valid[[_id, _target]].copy()
    results = results.join(probabilities)
    
    ###############################################################################
    # Plot probability distribution
    plt.subplot(3,3,1)
    plt.hist(probas_[:, 1], range=(0, 1), bins=100, histtype="step")
    ###############################################################################
    # plot proba distribution for both class
    target_probs = pd.DataFrame(valid_Y, columns=['target'])
    target_probs['probs'] = probas_[:, 1]
    plt.subplot(3, 3, 2)
    plt.hist(target_probs[target_probs['target']==1]['probs'], range=(0, 1), bins=100,
             label="Class 1", histtype="step", color='green', density=True)
    plt.hist(target_probs[target_probs['target']==0]['probs'], range=(0, 1), bins=100,
             label="Class 0", histtype="step", color='red', density=True)
        
    ###############################################################################
    # Plot calibration plots
    fraction_of_positives, mean_predicted_value = calibration_curve(valid_Y, probas_[:, 1], n_bins=20)
    plt.subplot(3,3,3)
    plt.plot(mean_predicted_value, fraction_of_positives, "P-", lw=1)
    
    ###############################################################################
    # Compute the curve metrics and thresholds
    precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
    # Compute the F1 score from precision and recall
    beta = 1 ** 2
    f_score = ((1 + beta) * precision * recall /(beta * precision + recall))

    # Ensure thresholds ends at 1
    thresholds = np.append(thresholds, 1)
        
    # Compute the queue rate
    queue_rate = np.array([(probas_[:, 1] >= threshold).mean() for threshold in thresholds])
    
    trial_result = {
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'fscore': f_score,
        'queue_rate': queue_rate
        }
    
    ###############################################################################
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(valid_Y, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.subplot(3, 3, 5)
    plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % (roc_auc), lw=1)
    
    ###############################################################################
    # Compute Precision-Recall curve and area the curve
    precision, recall, thresholds = precision_recall_curve(valid_Y, probas_[:, 1])
    pr_auc = auc(recall, precision)
    plt.subplot(3, 3, 6)
    plt.plot(recall, precision, label='PR (area = %0.2f)' % (pr_auc), lw=1)
    
    ###############################################################################
    # calculate lift related information
    cust_rank = results[[_target, _prob_name]].copy()
    cust_rank = cust_rank.sort_values(by=_prob_name, ascending=False).reset_index(drop=True)
    cust_rank['rank'] = cust_rank.index + 1
    cust_rank['num_pos_target'] = np.cumsum(cust_rank[_target])
    pos_rate = results[_target].mean()
    pos_sum = results[_target].sum()
    
    lift_cums = []
    lift_decays = []
    cum_poss = []
    
    for q in range(10, 110, 10):
        small_q = (q - 10) / 100.0
        big_q = q / 100.0
        if q == 100:
            lift_cum = cust_rank[_target].mean() / pos_rate
            lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) :][_target].mean() / pos_rate
            cum_pos = cust_rank[_target].sum() / pos_sum
        else:
            lift_cum = cust_rank[: int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
            lift_decay = cust_rank[int(small_q * cust_rank.shape[0]) : int(big_q * cust_rank.shape[0])][_target].mean() / pos_rate
            cum_pos = cust_rank[: int(big_q * cust_rank.shape[0])][_target].sum() / pos_sum
            
        lift_cums.append(lift_cum)
        lift_decays.append(lift_decay)
        cum_poss.append(cum_pos)

    print ('AUC: %f, lift at 10 percent: %f' % (roc_auc, lift_cums[0]))

    ###############################################################################
    # calculate number of calls
    max_positive_in_fold = cust_rank.num_pos_target.max()
    max_possible_actions = min(max_positive_in_fold,3000)

    nb_calls = cust_rank.copy()
    for i in range(100,int(max_possible_actions), 100):
        nb_calls['to_get_%i' %i] = nb_calls.loc[nb_calls.num_pos_target==i,'rank'].min()

    nb_calls['to_get_all'] = nb_calls.loc[nb_calls.num_pos_target==nb_calls.num_pos_target.max(),'rank'].min()

    nb_calls = nb_calls[[col1 for col1 in nb_calls.columns if col1.startswith('to_get_')]].min()
    nb_calls = pd.DataFrame(nb_calls,columns=['nb_actions'])
    nb_calls['total_population'] = cust_rank.shape[0]
    nb_calls['total_pos_targets'] = cust_rank[_target].sum()
    nb_calls['nb_pos_targets']= range(100,int(max_possible_actions), 100)+[int(max_possible_actions)]
    nb_calls['pos_rate'] = nb_calls.nb_pos_targets/nb_calls.nb_actions
    nb_calls['Percentage_of_pos_targets_found'] = nb_calls.nb_pos_targets/nb_calls.total_pos_targets
    nb_calls['Percentage_of_Population'] = nb_calls.nb_actions/nb_calls.total_population
    nb_calls['Lift'] = nb_calls.Percentage_of_pos_targets_found/nb_calls.Percentage_of_Population
    
    for col in nb_calls.columns:
        nb_calls[col] = pd.to_numeric(nb_calls[col])
    nb_calls = nb_calls.reset_index().rename(columns={'index':'objective'})
    nb_calls = nb_calls.groupby('objective').mean().sort_values(by='nb_pos_targets')
    
    ###############################################################################
    # Discrimination threshold measures
    # Compute maximum number of uniform thresholds across all trials
    n_thresholds = len(trial_result['thresholds'])
    thresholds_ = np.linspace(0.0, 1.0, num=n_thresholds)

    metrics = [metric for metric in trial_result.keys() if metric!='thresholds']
    uniform_metrics = defaultdict(list)
    
    rows = defaultdict(list)
    for t in thresholds_:
        idx = bisect.bisect_left(trial_result['thresholds'], t)
        for metric in metrics:
            rows[metric].append(trial_result[metric][idx])

    for metric, row in rows.items():
        uniform_metrics[metric].append(row)
    
    # Convert metrics to metric arrays
    uniform_metrics = {metric: np.array(values) for metric, values in uniform_metrics.items()}

    # plot probas for probas
    plt.subplot(3, 3, 1)
    plt.ylabel('Count', fontsize=10)
    plt.title('Predicted probas', fontsize=12)
    #plt.legend(loc="lower right")

    # plot probas for both classes
    plt.subplot(3, 3, 2)
    plt.ylabel('Density', fontsize=10)
    plt.title('Predicted probas for different classes', fontsize=12)
    plt.legend(loc="lower right")

    # plot the perfectly calibrated curve
    plt.subplot(3,3,3)
    plt.plot([0, 1], [0, 1], "k--")
    plt.ylabel("Fraction of positives", fontsize=10)
    plt.xlabel("Mean predicted value", fontsize=10)
    plt.ylim([-0.05, 1.05])
    #plt.legend(loc="lower right")
    plt.title('Calibration plots  (reliability curve)', fontsize=12)
    
    # plot discrimination threshold
    plt.subplot(3, 3, 4)
    color_values = ['g','mediumvioletred','b','r']
    for idx, metric in enumerate(metrics):
        color = color_values[idx]
        # Make the label pretty
        label = metric.replace("_", " ")
        # Draw the metric values
        plt.plot(thresholds_, uniform_metrics[metric].T, color=color, label=label)
        # Annotate the graph with the maximizing value
        if metric == 'fscore':
            argmax = uniform_metrics[metric].argmax()
            threshold = thresholds_[argmax]
            plt.axvline(threshold, ls='--', c='k', lw=1,
                        label="$t_{}={:0.2f}$".format(metric[0], threshold))

    # Set the title of the threshold visualization
    plt.title("Threshold Plot")
    plt.legend(frameon=True, loc='best')
    plt.xlabel('Discrimination Threshold')
    plt.ylabel('Score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # plot the averaged ROC curve
    plt.subplot(3, 3, 5)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    plt.title('ROC', fontsize=12)
    plt.legend(loc="lower right")
    
    # plot averaged PR curve
    plt.subplot(3, 3, 6)
    plt.xlabel('Recall', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.title('Precision-recall', fontsize=12)
    plt.legend(loc="lower right")
    
    # plot lift cumulative
    plt.subplot(3, 3, 7)
    plt.bar(range(10), lift_cums)
    plt.xticks(range(10), ['0-%d' %(num) for num in range(10, 110, 10)], rotation=-45)
    for i, v in enumerate(zip(lift_cums,cum_poss)):
        plt.text(i-0.5, v[0]+.05, '%.2f' %(v[0]), color='black')
        plt.text(i-0.5, v[0]+.4, '(%.0f%%)' %(v[1]*100), color='red', fontsize=8)
    plt.xlabel('Rank percentage interval', fontsize=10)
    plt.ylabel('Lift', fontsize=10)
    plt.title('Lift cumulative plot', fontsize=12)
    
    # plot lift decay
    plt.subplot(3, 3, 8)
    plt.bar(range(10), lift_decays)
    plt.xticks(range(10), ['%d-%d' %(num-10, num) for num in range(10, 110, 10)], rotation=-45)
    for i, v in enumerate(lift_decays):
        plt.text(i-0.5, v+.05, '%.2f' %v, color='black')
    plt.xlabel('Rank percentage interval', fontsize=10)
    plt.ylabel('Lift', fontsize=10)
    plt.title('Lift decay plot', fontsize=12)
    
    # plot number of calls
    plt.subplot(3, 3, 9)
    plt.bar(range(len(nb_calls['nb_actions'].values)), nb_calls['nb_actions'].values)
    plt.xticks(range(len(nb_calls['nb_actions'].values)), nb_calls['nb_pos_targets'].values, rotation=-45)
    for i, v in enumerate(nb_calls['nb_actions'].values):
        plt.text(i-0.5, v*1.02, '%.0f' %v, color='black')
    plt.xlabel('Number of target to get', fontsize=10)
    plt.ylabel('Number of contacts', fontsize=10)
    plt.title('Number of actions', fontsize=12)
    
    plt.show();
    plt.gcf().clear();
    
    result_resume = {
        'results_targeting': results,
        'nb_calls': nb_calls
    }
    return result_resume
