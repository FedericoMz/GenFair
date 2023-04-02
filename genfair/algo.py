
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.naive_bayes import GaussianNB


from deap import base
from deap import creator
from deap import tools


import lightgbm as ltb

from scipy.spatial import distance



import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 









def ranker(df, attributes, class_name):
    X = df[attributes].values
    y = df[class_name].to_list()
    clf = GaussianNB()

    clf.fit(X, y)
    X_proba = [None] * len(X)
    probability = []
    for k in range(len(X)):
        X_proba[k] = np.append(X[k], (y[k], clf.predict_proba(X)[k][1]))

    X_proba = np.array(X_proba, dtype=float).tolist()
    
    X_proba_list = []
    
    for arr in (X_proba):
        X_proba_list.append(arr[:len(attributes)+2])
        
    X_proba = pd.DataFrame.from_records(X_proba_list)
    X_proba.columns = attributes + [class_name] + ['proba']
    
    return X_proba

def get_index (attribute, attributes):
    for i in range(len(attributes)):
        if attributes[i] == attribute:
            return i


def get_closest_value (val, value_list):
    
    return value_list[np.argmin(np.abs(np.array(value_list)-val))]


def get_causal_regressor (data, index_X, index_y):
    
    X = data.iloc[:, index_X].values
    y = data.iloc[:, [index_y]].values

    regr = ltb.LGBMRegressor().fit(X, y)

    return regr

def get_causal_classifier (data, index_X, index_y):

    X = data.iloc[:, index_X].values
    y = data.iloc[:, [index_y]].values

    classifier = ltb.LGBMClassifier().fit(X, y)    

    return classifier


def get_discrimination (df, sensitive_attributes, class_name):


    
    sensitive_dict = {}
    
    tot_disc = 0
    tot_pos = 0
        
    for attr in sensitive_attributes:
        print ()
        print ("Analizing", attr, "...")
        sensitive_dict[attr] = {}
        sensitive_dict[attr]['D'] = {}
        sensitive_dict[attr]['P'] = {}
        sensitive_dict[attr]['D']['values_list'] = []
        sensitive_dict[attr]['P']['values_list'] = []
        values = df[attr].unique()
        for val in values:
            PP = df[(df[attr] != val) & (df[class_name] == 1)].values.tolist()
            PN = df[(df[attr] != val) & (df[class_name] == 0)].values.tolist()
            DP = df[(df[attr] == val) & (df[class_name] == 1)].values.tolist()
            DN = df[(df[attr] == val) & (df[class_name] == 0)].values.tolist()

            disc = len(DN) + len(DP)
            priv = len(PN) + len(PP)
            pos = len(PP) + len(DP)
            neg = len(PN) + len(DN)
            
            DP_exp = round(disc * pos / len(df))
            PP_exp = round(priv * pos / len(df))
            DN_exp = round(disc * neg / len(df))
            PN_exp = round(priv * neg / len(df))
            
            discrimination = len(PP) / (len(PP) + len(PN)) - len(DP) / (len(DP) + len(DN))
       
            if discrimination >= 0:
                status = 'D'
                sensitive_dict[attr][status][val] = {}
                print(val, "is discriminated:", discrimination)
                
                sensitive_dict[attr][status][val]['P'] = sorted(DP, key=lambda x:x[len(DP[0])-1])
                sensitive_dict[attr][status][val]['P_exp'] = DP_exp
                sensitive_dict[attr][status][val]['P_curr'] = 0
                
                for i in range(len(sensitive_dict[attr][status][val]['P'])):
                    del sensitive_dict[attr][status][val]['P'][i][-1]
                                
                sensitive_dict[attr][status][val]['N'] = sorted(DN, key=lambda x:x[len(DN[0])-1], reverse = True)
                sensitive_dict[attr][status][val]['N_exp'] = DN_exp
                sensitive_dict[attr][status][val]['N_curr'] = 0

                for i in range(len(sensitive_dict[attr][status][val]['N'])):
                    del sensitive_dict[attr][status][val]['N'][i][-1]
                    
                print("- DP:", len(sensitive_dict[attr][status][val]['P']), '· Expected:', DP_exp, 
                      '· To be added:', abs(len(DP) - DP_exp))
                print("- DN:", len(sensitive_dict[attr][status][val]['N']), '· Expected:', DN_exp, 
                      '· To be removed:', abs(len(DN) - DN_exp))
                
                tot_disc = tot_disc + abs(len(DP) - DP_exp)
                
            else:
                status = 'P'
                sensitive_dict[attr][status][val] = {}
                print("")
                print(val, "is privileged:", discrimination)   
                
                sensitive_dict[attr][status][val]['P'] = sorted(DP, key=lambda x:x[len(DP[0])-1])
                sensitive_dict[attr][status][val]['P_exp'] = DP_exp
                sensitive_dict[attr][status][val]['P_curr'] = 0

                for i in range(len(sensitive_dict[attr][status][val]['P'])):
                    del sensitive_dict[attr][status][val]['P'][i][-1]
                    
                sensitive_dict[attr][status][val]['N'] = sorted(DN, key=lambda x:x[len(DN[0])-1], reverse = True)
                sensitive_dict[attr][status][val]['N_exp'] = DN_exp
                sensitive_dict[attr][status][val]['N_curr'] = 0

                for i in range(len(sensitive_dict[attr][status][val]['N'])):
                    del sensitive_dict[attr][status][val]['N'][i][-1]
                
                print("- PP:", len(sensitive_dict[attr][status][val]['P']), '· Expected:', DP_exp, 
                      '· To be removed:', abs(len(DP) - DP_exp))
                print("- PN:", len(sensitive_dict[attr][status][val]['N']), '· Expected:', DN_exp, 
                      '· To be added:', abs(len(DN) - DN_exp))
                
                tot_pos = tot_pos + abs(len(DP) - DP_exp)
            
            sensitive_dict[attr][status]['values_list'].append(val)
        
    round_error = abs(tot_disc - tot_pos)
    
    if round_error > 0:
        print ("")
        print ("Due to a rounding error, the final dataset might be slightly smaller")
                                   
    return sensitive_dict            


def random_individual(values, const, 
                      weighted_indexes, random_indexes, integer_indexes, regular_indexes, 
                      causal_reg, causal_class):
    
    df = pd.DataFrame(values)
    
    ind = [None] * (len(df.columns)-1)
    
    for tup in const:
        ind[tup[0]] = tup[1]

    for i in regular_indexes:
        val = df.iloc[:, i].to_list()
        
        if i in weighted_indexes: #if the feat can only assume values already in the dataset, randomly
            ind[i] = random.choice(val)
        elif i in random_indexes:
            ind[i] = random.choice(list(set(val))) #... or weighted w.r.t. the distribution
        elif i in integer_indexes: #if the feat can only assume a random value in a int range
            ind[i] = random.randint(min(val), max(val))            
        else: #if the feat can assume a float value in a range
            ind[i] = random.uniform(min(val), max(val))


    for e in causal_reg + causal_class:
        
        X_indexes = e[0]
        y = e[1]
        pred = e[2]
        
        X_val = []
        for index in X_indexes:
            X_val.append(ind[index])
             
        predicted = pred.predict([X_val])
        
        if y in weighted_indexes + random_indexes:
            value_list = df.iloc[:, y].to_list()
            ind[y] = get_closest_value(predicted[0], value_list)        
        elif y in integer_indexes:
            ind[y] = int(predicted[0])        
        else: 
            ind[y] = predicted[0]
    
    return ind

def evaluate(individual, forest, medoid, scaler):
 
    individual = np.array(individual[0])

    individual = scaler.transform(individual.reshape(1, -1))[0]


    score = distance.cosine(individual, medoid) / 2


    
    return score,


def mate(ind1, ind2, values, 
         weighted_indexes, random_indexes, integer_indexes, regular_indexes, 
         causal_reg, causal_class):
    
    #custom crossover
    
    indpb = 0.34
    
    for i in regular_indexes:
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]
            for e in causal_reg + causal_class:
                if i in e[0]:
                    df = pd.DataFrame(values)
                    
                    X_indexes = e[0]
                    y = e[1]
                    pred = e[2]

                    X_val1 = []
                    X_val2 = []
                    for index in X_indexes:
                        X_val1.append(ind1[index])
                        X_val2.append(ind2[index])

                    #ind1
                    predicted1 = pred.predict([X_val1])

                    if y in weighted_indexes + random_indexes:
                        value_list = df.iloc[:, y].to_list()
                        ind1[y] = get_closest_value(predicted1[0], value_list)        
                    elif y in integer_indexes:
                        ind1[y] = int(predicted1[0])        
                    else: 
                        ind1[y] = predicted1[0]
                        
                    #ind2
                    predicted2 = pred.predict([X_val2])

                    if y in weighted_indexes + random_indexes:
                        value_list = df.iloc[:, y].to_list()
                        ind2[y] = get_closest_value(predicted2[0], value_list)        
                    elif y in integer_indexes:
                        ind2[y] = int(predicted2[0])        
                    else: 
                        ind2[y] = predicted2[0]
                                      
                
    return ind1, ind2


def mutate(individual, values, 
           weighted_indexes, random_indexes, integer_indexes, regular_indexes, 
           causal_reg, causal_class):
    
#custom mutation

    df = pd.DataFrame(values)
    
    i = random.choice(regular_indexes) #we select a random feature to mutate
    

    if i in weighted_indexes: 
        val = df.iloc[:, i].to_list()
        individual[i] = random.choice(val)
    elif i in random_indexes: 
        val = df.iloc[:, i].to_list()
        individual[i] = random.choice(list(set(val)))
    elif i in integer_indexes: #if the feat can only assume a random value in a int range
        val = [x for x in df.iloc[:, i].to_list() if x != individual[i]]
        individual[i] = random.randint(min(val), max(val))                        
    else: #if the feat can assume a float value in a range
        val = [x for x in df.iloc[:, i].to_list() if x != individual[i]]
        individual[i] = random.uniform(min(val), max(val))

    
    for e in causal_reg + causal_class: 
        if i in e[0]:

            X_indexes = e[0]
            y = e[1]
            pred = e[2]

            X_val = []
            for index in X_indexes:
                X_val.append(individual[index])

            predicted = pred.predict([X_val])

            if y in weighted_indexes + random_indexes:
                value_list = df.iloc[:, y].to_list()
                individual[y] = get_closest_value(predicted[0], value_list)        
            elif y in integer_indexes:
                individual[y] = int(predicted[0])        
            else: 
                individual[y] = predicted[0]

    return individual,


def GA(values, const, n_HOF, forest, medoid, 
       weighted_indexes, integer_indexes, random_indexes, regular_indexes, 
       causal_reg, causal_class, scaler):
    
    print ("GA started,", n_HOF, "individual(s) will be generated")
        
    NUM_GENERATIONS = 50
    POPULATION_SIZE = 150
    
    CXPB, MUTPB = 0.5, 0.15
    
    creator.create('Fitness', base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()

    toolbox.register("random_individual", random_individual, values, const, weighted_indexes, 
                     random_indexes, integer_indexes, regular_indexes, causal_reg, causal_class)

    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.random_individual, n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, forest=forest, medoid=medoid, scaler=scaler)
        
    toolbox.register("mate", mate, values=values, weighted_indexes=weighted_indexes, random_indexes=random_indexes,
                     integer_indexes=integer_indexes, regular_indexes=regular_indexes, 
                     causal_reg=causal_reg, causal_class=causal_class)
    
    toolbox.register("mutate", mutate, values=values, weighted_indexes=weighted_indexes, random_indexes=random_indexes,
                     integer_indexes=integer_indexes, regular_indexes=regular_indexes, 
                     causal_reg=causal_reg, causal_class=causal_class)    
    
    toolbox.register("select", tools.selNSGA2)


    pop = toolbox.population(n=POPULATION_SIZE)

    hof = tools.HallOfFame(n_HOF)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)   
    #stats.register('max', np.max, axis = 0)
    stats.register('min', np.min) #, axis = 0)
    stats.register('avg', np.mean) #, axis = 0)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields
    
    invalid_individuals = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit
        
    hof.update(pop)
    hof_size = len(hof.items)

    record = stats.compile(pop)
    logbook.record(gen=0, best="-", nevals=len(invalid_individuals), **record)
    print(logbook.stream)

    
    for gen in range(1, NUM_GENERATIONS + 1):
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values
            
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        
        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
        
        
    hof.update(pop)
    
    counter = 0
    for e in hof.items:
        counter = counter + 1
        print("")
        print("#"+str(counter))
        print("Constraints:", const)
        print('Individual:', e)
        print('Fitness:', e.fitness)


    plt.figure(1)

    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='blue')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Avg and Min Fitness')
    plt.show()
    
    
    return hof.items
