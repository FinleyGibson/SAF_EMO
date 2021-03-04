import os
import sys
import numpy as np
import pickle as pkl
import copy

pkl_dir = os.path.join(sys.argv[1], 'pkl_data/')
pkl_path = os.path.join(pkl_dir, 'results__newsms.pkl')
copy_path = os.path.join(pkl_dir, 'results_sorted_newsms.pkl')

with open(pkl_path, 'rb') as infile:
    optimisers = pkl.load(infile)


fixed = copy.deepcopy(optimisers)

print(fixed.keys())
for opt, data in optimisers.items():
    seeds = np.array(data['seed'])
    print(seeds)
    print(data.keys())
    for seed in range(31):
        seed_ind = np.where(seeds==seed)[0][0]
        fixed[opt]['x'][seed] = data['x'][seed_ind] 
        fixed[opt]['y'][seed] = data['y'][seed_ind] 
        fixed[opt]['seed'][seed] = data['seed'][seed_ind] 
        fixed[opt]['hypervolume'][seed] = data['hypervolume'][seed_ind] 
        fixed[opt]['igd+'][seed] = data['igd+'][seed_ind] 


# ref = False
# for opt, data  in fixed.items():
#     if opt != 'lhs':
#         print(np.shape(data['x']))
#         print(np.shape(data['hypervolume']))
# 
#         
#         if ref is False:
#             ref = True
#             ref_x = np.array([data['x'][i][:10, :] for i in range(31)])
#             ref_y = np.array([data['y'][i][:10, :] for i in range(31)])
#             ref_hv = np.array([data['hypervolume'][i][:10] for i in range(31)])
#             ref_igd = np.array([data['igd+'][i][:10] for i in range(31)])
# 
#         x = np.array([data['x'][i][:10, :] for i in range(31)])
#         y = np.array([data['y'][i][:10, :] for i in range(31)])
#         hv = np.array([data['hypervolume'][i][:10] for i in range(31)])
#         igd = np.array([data['igd+'][i][:10] for i in range(31)])
# 
#         try:
#             np.testing.assert_array_almost_equal(x, ref_x, 2)
#             np.testing.assert_array_almost_equal(y, ref_y, 2)
#             np.testing.assert_array_almost_equal(hv, ref_hv, 2)
#             np.testing.assert_array_almost_equal(igd, ref_igd, 2)
#         except AssertionError:
#             print("assert equal not found in {}".format(opt))
#             print('=='*80)
#             print(y)
#             print()
#             print('=='*80)
#             print(ref_y)
#             np.testing.assert_array_almost_equal(y, ref_y, 4)
# #            np.testing.assert_array_almost_equal(y, ref_y, 6)
# 
#         print("pass")
#         print('**'*80)
#     else:
#         print(data['seed'])
# 


with open(copy_path, 'wb') as outfile:
    pkl.dump(fixed, outfile)
