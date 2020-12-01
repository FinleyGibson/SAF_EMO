import pickle
from shutil import copyfile



with open("./pkl_data/results.pkl", 'rb') as infile:
    results = pickle.load(infile)

names = [r['name'] for r in results]
if len(names)<len(list(set(names))):
    ## not been modified
    copyfile('./pkl_data/results.pkl', './pkl_data/results_backup.pkl')

for i, r in enumerate(results):
    name = r['name']
    dir_split = [ i.lower() for i in r['log_dir'][0].split('_')]

    assert name.lower() in dir_split
    print(name)
    
    if 'ei' in dir_split:
        name+=' ei'
    elif  'mean' in dir_split:
        name+=' $\mu$'

    if name =='lhs':
        name = 'LHS'

    print(r['log_dir'][0], "\t----->", name)
    results[i]['name'] = name


# assert set(names) == set(['Mpoi', 'ParEgo', 'Saf \$mu$', 'Saf ei', 'SmsEgo \$mu$', 'SmsEgo ei', 'LHS'])

with open('./pkl_data/results.pkl', 'wb') as outfile:
    pickle.dump(results, outfile)

names = [r['name'] for r in results], 
print('Done!')
