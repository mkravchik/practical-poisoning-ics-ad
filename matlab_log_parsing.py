import re
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import keras
import pickle

def process_matlab_log(log_file, silent=True):
    f = open(log_file, "rt+")
    lines = f.readlines()
    f.close()
    
    att_idx = 31
    #tests starts with 
    # temporary misspelled variant
    # Att_level 23.200000 att_delay;0.000000

    #Run starts
    att_run_start = "Poison loaded. num_poisons (\d+), att_idx (\d+)"
    
    test_start = "^Att_level (\d+\.\d+) att_delay (\d+\.\d+).*$"
    
    last_sep_val = "^Last value of sep level (\d+\.\d+) time (\d+\.\d+).*$"

    #Attack generation with Sep level u(12) 53.075893, Sep flow u(14) 25.648470, xmv7 37.603183, Strip level u(15) 49.684285, strip flow u(17) 22.817279 xmv8 46.265101 at time 7.550500
    pattern = " with Sep level u\(12\) (\d+\.\d+), Sep flow u\(14\) (\d+\.\d+), xmv7 (\d+\.\d+), Strip level u\(15\) (\d+\.\d+), strip flow u\(17\) (\d+\.\d+) xmv8 (\d+\.\d+) at time (\d+\.\d+)"
    pattern_att_1 = " with A Feed Rate u\(1\) (-?\d+\.\d+), Reactor Pressure u\(7\) (\d+\.\d+), Reactor Level u8 (\d+\.\d+), xmv3 (\d+\.\d+) at time (\d+\.\d+)"

    pred_input = "^\[\[.*\]\]$"
    pred_input_first = "^\[\[.*\s+$"

    # can be also   5.36676888e+01 5.95000000e+00]]
    pred_input_last = "^\s+.*\s+(\d+\.\d+)\s+\]\]$|^\s+.*\s+(\d+\.\d+e[+-]\d{2}).*\]\]$"
    
    pattern_attack_start = "^Attack generation" + pattern + ".*$"
    pattern_attack_start_att_1 = "^Attack generation" + pattern_att_1 + ".*$"
    #ends with
    #Stop attack generation with Sep level u(12) 4.030612, Sep flow u(14) 24.897414, xmv7 36.681489, Strip level u(15) 105.718569, strip flow u(17) 23.568345 xmv8 49.438867 at time 9.525500
    pattern_attack_end = "^Stop attack generation" + pattern + ".*$"
    pattern_attack_end_att_1 = "^Stop attack generation" + pattern_att_1 + ".*$"
    pattern_shutdown = "Shutting down"
    sim_stops = "^Simulator completed!.*$"
    sim_starts = "^Running simulator!$"

    # patterns from the posioning with attack
    poison_starts = "^Applying poison with .*$"
    poison_stops = "^Stop applying poison .*$"
    att_starts = "^Attacking" + pattern + ".*$"
    att_starts_att_1 = "^Attacking" + pattern_att_1 + ".*$"
    
    att_stops = "^Stop attacking poison" + pattern + ".*$"
    att_stops_att_1 = "^Stop attacking poison" + pattern_att_1 + ".*$"

    state = "before_test"
    cols = ["time", "Delay", "Att level", "Sep level u(12)", "Sep flow u(14)", 'xmv7', 'Strip level u(15)',
            "strip flow u(17)", "xmv8", "Final Sep level u(12)", "Shutdown", "Att time", "Att len", "num_poisons"]
    
    cols_att_1 = ["time", "Delay", "Att level", "A Feed Rate u(1)", "Reactor Pressure u(7)", 'Reactor Level u8',
            "xmv3", "Final Sep level u(12)", "Shutdown", "Att time", "Att len", "num_poisons"]
    df_results = None
    rec = None
    num_poisons = 0
    att_len = None
    for line in lines:
#         if not silent: print("Processing", line)
        if state == "before_test":
            #looking for the test run start
            match = re.findall(att_run_start, line)
            if len(match):
                if not silent: print("Attack test starts", line)
                state = "before_test"
                num_poisons = int(match[0][0])
                att_idx = int(match[0][1])
                print("att_idx", att_idx)
                if att_idx == 1:
                    pattern_attack_start = pattern_attack_start_att_1
                    pattern_attack_end = pattern_attack_end_att_1
                    att_starts = att_starts_att_1
                    att_stops = att_stops_att_1
                    cols = cols_att_1
                if df_results is None:
                    df_results = pd.DataFrame(columns=cols)
                continue
            #looking for the test start
            match = re.findall(test_start, line)
            if len(match):
                if not silent: print("Test starts", line)
                state = "before_sim"
                level = match[0][0]
                delay = match[0][1]
                continue
        if state == "before_sim":
            #looking for the test start
            match = re.findall(sim_starts, line)
            if len(match):
                if not silent: print("Sim starts", line)
                state = "before_att"
                continue
        if state == "in_pred_poison":
            if not silent: print("Prediction", line)
            state = "before_att"
            continue
        if state == "in_pred_poison_first":
            match = re.findall(pred_input_last, line)
            if len(match):
                if not silent: print("Prediction input ends", line, match[0])
                att_len = float(match[0][0]) if len(match[0][0]) else float(match[0][1])
                state = "in_pred_poison"
                continue
        if state == "before_att":
            # in the older files, there was no prediction here
            # in the new ones - there is
            match = re.findall(pred_input, line)
            if len(match):
                if not silent: print("Prediction input", line)
                state = "in_pred_poison"
                continue
            match = re.findall(pred_input_first, line)
            if len(match):
                if not silent: print("Prediction input starts", line)
                state = "in_pred_poison_first"
                continue            
            #looking for the attack start
            match = re.findall(pattern_attack_start, line)
            if 0 == len(match):
                match =  re.findall(att_starts, line)
            if len(match):
                if not silent: print("Attack starts", line)
                state = "started"
                rec = {
                    "time":match[0][-1],
                    "Final Sep level u(12)":2.0,
                    "Shutdown":False,
                    "Att time":0,
                    "Delay": delay,
                    "Att level":level,
                    "num_poisons":num_poisons
                }
                if att_idx == 1:
                    rec["A Feed Rate u(1)"] = match[0][0]
                    rec["Reactor Pressure u(7)"] = match[0][1]
                    rec['Reactor Level u8'] = match[0][2]
                    rec['xmv3'] = match[0][3]
                else:
                    rec["Sep level u(12)"] = match[0][0]
                    rec["Sep flow u(14)"] = match[0][1]
                    rec['xmv7'] = match[0][2]
                    rec['Strip level u(15)'] = match[0][3]
                    rec["strip flow u(17)"] = match[0][4]
                    rec["xmv8"] = match[0][5]
#                 print(match, rec)
                continue
            match = re.findall(last_sep_val, line)
            if len(match):
                if not silent: print("Last sep level", line)
                # if we got here without seeing an attack - do not add anything
                rec = None
                state = "before_test"
                continue                       
        if state == "started":
            #looking for the attack end
            match = re.findall(pattern_attack_end, line)
            if 0 == len(match):
                match =  re.findall(att_stops, line)
            if len(match):
                if not silent: print("Attack ends", line)
                if att_idx != 1:
                    rec["Final Sep level u(12)"] = match[0][0]
                rec["Att time"] = float(match[0][-1]) - float(rec["time"])
                rec["Shutdown"] = False
                state = "after_att"
                continue
            #looking for the shutdown
            match = re.findall(pattern_shutdown, line)
            if len(match):
                if not silent: print("Shutdown", line)
                rec["Shutdown"] = True
                # not changing the state
                continue
            #looking for the Last values
            match = re.findall(last_sep_val, line)
            if len(match):
                if not silent: print("Last sep level", line)
                rec["Final Sep level u(12)"] = match[0][0]
                rec["Att time"] = float(match[0][1]) - float(rec["time"])
                # the only place to add the record is after the Last sep level record
                if att_len is None and not rec["Shutdown"]:
                    att_len = rec["Att time"]
                if att_len is None:
                    rec["Att len"] = rec["Att time"]
                else:
                    rec["Att len"] = att_len

                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_test"
                continue
        if state == "after_att":
            #looking for the shutdown
            match = re.findall(pattern_shutdown, line)
            if len(match):
                if not silent: print("Shutdown", line)
                rec["Shutdown"] = True
                continue
            #looking for the Last values
            match = re.findall(last_sep_val, line)
            if len(match):
                if not silent: print("Last sep level", line)
                # the only place to add the record is after the Last sep level record
                if rec["Shutdown"]: # if there was a shutdown after the attack has ended - update the level and time
                    rec["Final Sep level u(12)"] = match[0][0]
                    rec["Att time"] = float(match[0][1]) - float(rec["time"])
                if att_len is None and not rec["Shutdown"]:
                    att_len = rec["Att time"]
                if att_len is None:
                    rec["Att len"] = rec["Att time"]
                else:
                    rec["Att len"] = att_len
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_test"
                continue
            
                
    if rec is not None:
        df_results = df_results.append(rec, ignore_index=True) 
        
    if not silent: print(df_results)
        
    
    return df_results


def process_matlab_log_ce_loss(log_file, silent=True):
    f = open(log_file, "rt+")
    lines = f.readlines()
    f.close()
    
    #tests starts with 
    # temporary misspelled variant
    # Att_level 23.200000 att_delay;0.000000

    pred_input = "^\[\[.*\]\]$"
    pred_input_first = "^\[\[.*\s+$"
    pred_input_last = "^\s+.*\]\]$"
    poison_starts = "^Applying poison with .*$"
    poison_stops = "^Stop applying poison .*$"
    att_starts = "^Attacking with .*$|^Attack generation .*$"
    att_stops = "^Stop attacking .*$|^Stop attack generation .*$"

    sim_stops = "^Simulator completed!.*$"
    sim_starts = "^Running simulator!$"
    pattern_shutdown = "Shutting down"


    state = "before_sim"
    cols = ["y_true", "y_pred"]
    
    df_results = pd.DataFrame(columns=cols)
    rec = None
    for line in lines:
#         if not silent: print("Processing", line)
        if state == "before_sim":
            #looking for the test start
            match = re.findall(sim_starts, line)
            if len(match):
                if not silent: print("Sim starts", line)
                state = "before_poison_or_att"
                continue
        if state == "before_poison_or_att":
            if rec is not None:
                match = re.findall(pattern_shutdown, line)
                if len(match):
                    if not silent: print("Shutdown", line)
                    rec["y_true"] = 1
                else:
                    rec["y_true"] = 0
                # add the record and keep on processing
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
            match = re.findall(pred_input, line)
            if len(match):
                if not silent: print("Prediction input", line)
                state = "in_pred_poison"
                continue
            match = re.findall(pred_input_first, line)
            if len(match):
                if not silent: print("Prediction input starts", line)
                state = "in_pred_poison_first"
                continue
        if state == "in_pred_poison_first":
            match = re.findall(pred_input_last, line)
            if len(match):
                if not silent: print("Prediction input ends", line)
                state = "before_pred_poison_or_attack"
                continue
        if state == "before_pred_poison_or_attack":
            if not silent: print("Prediction", line)
            rec = {
                "y_pred":float(line.strip())
            }
            state = "after_pred_poison_or_attack"
            continue
        if state == "after_pred_poison_or_attack":               
            match = re.findall(poison_starts, line)
            if len(match):
                if not silent: print("Poison starts", line)
                state = "in_poison"
                continue
            match = re.findall(att_starts, line)
            if len(match):
                if not silent: print("Attack starts", line)
                state = "in_attack"
                continue            
            match = re.findall(pred_input, line)
            if len(match):
                if not silent: print("Prediction input", line)
                state = "in_pred_poison"
                continue
            match = re.findall(pred_input_first, line)
            if len(match):
                if not silent: print("Prediction input starts", line)
                state = "in_pred_poison_first"
                continue
        if state == "in_poison":
            match = re.findall(poison_stops, line)
            if len(match):
                if not silent: print("Poison stops", line)
                rec["y_true"] = 0
                # wait - don't add it yet
                # df_results = df_results.append(rec, ignore_index=True) 
                # rec = None
                state = "before_poison_or_att"
                continue
            match = re.findall(sim_stops, line)
            if len(match):
                if not silent: print("Simulation stops (shutdown)", line)
                rec["y_true"] = 1
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_sim"
                continue

        if state == "in_attack":
            match = re.findall(att_stops, line)
            if len(match):
                if not silent: print("Attack stops", line)
                rec["y_true"] = 0
                # don't add yet - give it chance to shutdown after the attack
                # df_results = df_results.append(rec, ignore_index=True) 
                # rec = None
                state = "after_att"
                continue
            match = re.findall(sim_stops, line)
            if len(match):
                if not silent: print("Simulation stops (shutdown)", line)
                rec["y_true"] = 1
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_sim"
                continue
        
        if state == "after_att":
            match = re.findall(sim_stops, line)
            if len(match):
                if not silent: print("Simulation stops (NO shutdown)", line)
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_sim"
                continue
            match = re.findall(pattern_shutdown, line)
            if len(match):
                if not silent: print("Shutdown", line)
                rec["y_true"] = 1
                df_results = df_results.append(rec, ignore_index=True) 
                rec = None
                state = "before_sim"
                continue

            
            
                
    if rec is not None:
        df_results = df_results.append(rec, ignore_index=True) 
        
    if not silent: print(df_results)
        
    
    return df_results

def train_shutdown_predictor(df, clf_cols, save=False, load_prev=False, name_suffix=""):
    X = df[clf_cols]
    y = df.Shutdown.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

    def estimate(c, X_train, X_test, y_train, y_test):
            print(c)
            print("train accuracy", metrics.accuracy_score(c.predict(X_train), y_train))
            print("test accuracy", metrics.accuracy_score(c.predict(X_test), y_test))
            print("confusion matrix")
            print(metrics.confusion_matrix(y_test, c.predict(X_test)))
            print("f1 score", metrics.f1_score(y_test, c.predict(X_test), pos_label = 1.0))

    for clf in [LogisticRegression(random_state=0, max_iter=500, class_weight='balanced'), 
               RandomForestClassifier(class_weight='balanced'),
               LogisticRegressionCV(random_state=0, max_iter=500, class_weight='balanced')]:
        if load_prev:
            clf_name = repr(clf).split('(')[0]
            clf_old = pickle.load(open(clf_name + name_suffix + ".mdl", 'rb'))
            estimate(clf_old, X_train, X_test, y_train, y_test)

        clf.fit(X_train, y_train)
        if save:
            clf_name = repr(clf).split('(')[0]
            pickle.dump(clf, open(clf_name + name_suffix + ".mdl", 'wb'))
        estimate(clf, X_train, X_test, y_train, y_test)

def train_NN_shutdown_predictor(df, clf_cols, save=False, name_suffix=""):
    X = df[clf_cols]
    y = df.Shutdown.astype(int)

    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['mse'])

    earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                 verbose=1,
                                 min_delta=1e-4, mode='auto')
    lr_reduced = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3,
                                   verbose=1,
                                   min_delta=1e-4, mode='min')

    scaler = StandardScaler()
    X = X.values
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2, random_state=42)
    X_train, X_val, y_train, Y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train),
                                                     y_train)
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
    model.fit(X_train, y_train, epochs=500, batch_size=32,
              shuffle=True,
              verbose=2,
              callbacks = [earlyStopping, lr_reduced],
              validation_data=(X_val, Y_val),
              class_weight=class_weights
    )
    print(model)
    # print(model.predict(X_test), y_test)
    print("train accuracy", metrics.accuracy_score(model.predict(X_train)>0.5, y_train))
    print("test accuracy", metrics.accuracy_score(model.predict(X_test)>0.5, y_test))
    print("confusion matrix")
    print(metrics.confusion_matrix(y_test, model.predict(X_test)>0.5))
    print("f1 score", metrics.f1_score(y_test, model.predict(X_test)>0.5, pos_label = 1.0))
    
    if save:
        model.save("NN" + name_suffix + ".h5")
    
    return model, scaler

if __name__ == '__main__':
    log_file_name = "temexd_mod/poison_data_for_shutdown_predictor_att_1.txt"#"temexd_mod/poison_num_points_4_att_31_2.txt"#"attack31_with_poison_upd.txt"#"matlab_log.txt"
    # df_results = process_matlab_log_ce_loss("temexd_mod/att_1_shutdown_3.txt", silent=True).astype(float)
    # print(df_results)
    # print(df_results[df_results["y_true"]==1])
    # print(df_results[df_results["y_pred"]>0.6])
    # print(log_loss(df_results["y_true"], df_results["y_pred"]))
    # exit()
    att_idx = 1
    # log_file_name = "temexd_mod/poison_data_for_shutdown_predictor_att_3.txt"#"temexd_mod/poison_num_points_3_att_31.txt"#"attack31_4_poison_generation.txt"
    df_results = process_matlab_log(log_file_name).astype(float)
    print(df_results)
    print(df_results.Shutdown.sum())
    print(df_results["Att len"].unique())

    if att_idx == 1:
        clf_cols = ["Att level", "A Feed Rate u(1)", "Reactor Pressure u(7)", 'Reactor Level u8',
                "xmv3", "Att len"]
    else:
        clf_cols = ["Att level", "Sep level u(12)", "Sep flow u(14)", 'xmv7', 'Strip level u(15)',
                    "strip flow u(17)", "xmv8", "Att len"]

    df_results_subs = df_results[(df_results["Att len"] < 3.4)]
    print("subset length", len(df_results_subs))
    df_results_subs.to_csv("".join(log_file_name.split('.')[:-1])+".csv")
    train_shutdown_predictor(df_results_subs, clf_cols, save=True, load_prev=False, name_suffix = "_te_attack_%d" % att_idx)
    train_NN_shutdown_predictor(df_results_subs, clf_cols, save=True, name_suffix="_te_attack_%d" % att_idx)
    # print("Current predictions")
    # clf = pickle.load(open("RandomForestClassifier.mdl", 'rb'))
    # df_shutdown = df_results[df_results.Shutdown == 1][clf_cols]
    # df_shutdown.to_csv("shutdown.csv")
    # print(df_shutdown)
    # print(clf.predict(df_shutdown))
    # print(clf.predict_proba(df_shutdown))
    # print("Updated predictions")
    # clf = pickle.load(open("RandomForestClassifier1.mdl", 'rb'))
    # print(clf.predict(df_shutdown))
    # print(clf.predict_proba(df_shutdown))
