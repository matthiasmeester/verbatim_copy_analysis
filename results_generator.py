from math import sqrt

import pandas as pd

from dummy_index_map_creator import DummyIndexMapCreator
from heat_map_analysis import HeatMapAnalysis
from verbatim_heat_map_creator import VerbatimHeatMapCreator

index_map_creator = DummyIndexMapCreator((200, 200))
max_range = int(sqrt(200 ** 2 + 200 * 2))
result_df = pd.read_csv('output/results.csv')

for run in range(10):
    patches_05, copy_proportion_05 = index_map_creator.create_patch_map(num_patches=5)
    patches_10, copy_proportion_10 = index_map_creator.create_patch_map(num_patches=10)
    patches_20, copy_proportion_20 = index_map_creator.create_patch_map(num_patches=20)
    long_range_10, proportion_10 = index_map_creator.create_long_range_map(0.10)
    long_range_50, proportion_50 = index_map_creator.create_long_range_map(0.50)
    long_range_75, proportion_75 = index_map_creator.create_long_range_map(0.75)

    test_cases = [
        {
            'map': index_map_creator.create_full_random_map(),
            'real_verbatim': 0,
            'name': 'Full random',
            'category': 'full_random'
        },
        {
            'map': index_map_creator.create_full_verbatim_map(),
            'real_verbatim': 1,
            'name': 'Full verbatim',
            'category': 'full_verbatim'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=3),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 3x3',
            'category': 'checkerboard'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=1),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 1x1',
            'category': 'checkerboard'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=3),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 5x5',
            'category': 'checkerboard'
        },
        {
            'map': patches_05,
            'real_verbatim': copy_proportion_05,
            'name': f'Patches {copy_proportion_05}',
            'category': 'patches'
        },
        {
            'map': patches_10,
            'real_verbatim': copy_proportion_10,
            'name': f'Patches {copy_proportion_10}',
            'category': 'patches'
        },
        {
            'map': patches_20,
            'real_verbatim': copy_proportion_20,
            'name': f'Patches {copy_proportion_20}',
            'category': 'patches'
        },
        {
            'map': long_range_10,
            'real_verbatim': proportion_10,
            'name': f'Long range {proportion_10}',
            'category': 'long_range'
        },
        {
            'map': long_range_50,
            'real_verbatim': proportion_50,
            'name': f'Long range {proportion_50}',
            'category': 'long_range'
        },
        {
            'map': long_range_75,
            'real_verbatim': proportion_75,
            'name': f'Long range {proportion_75}',
            'category': 'long_range'
        }
    ]

    for test_case in test_cases:
        test_map = test_case['map']
        heat_map_creator = VerbatimHeatMapCreator(test_map)
        real_verbatim = test_case['real_verbatim']
        test_case_name = test_case['name']
        category = test_case['category']
        test_case_info = {
            'real_verbatim': real_verbatim,
            'test_case_name': test_case_name,
            'category': category,
            'run': run
        }

        # Mean heat
        mean_heat_d1_r1 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=1, inv_dist_weight_exp=1)).mean_heat_value()
        mean_heat_d2_r1 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=1, inv_dist_weight_exp=2)).mean_heat_value()
        mean_heat_d1_r2 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=2, inv_dist_weight_exp=1)).mean_heat_value()
        mean_heat_d2_r2 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=2, inv_dist_weight_exp=2)).mean_heat_value()
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "mean_heat_d=1|r=1",
            'predicted_verbatim': mean_heat_d1_r1,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "mean_heat_d=2|r=2",
            'predicted_verbatim': mean_heat_d2_r2,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "mean_heat_d=1|r=2",
            'predicted_verbatim': mean_heat_d1_r2,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "mean_heat_d=2|r=1",
            'predicted_verbatim': mean_heat_d2_r1,
        }), ignore_index=True
        )

        # Proportional neighbour analysis
        prop_r_1_p_0_05 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=1, inv_dist_weight_exp=0)).above_treshold_heat_index(0.05)
        prop_r_2_p_0_05 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=2, inv_dist_weight_exp=0)).above_treshold_heat_index(0.05)
        max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))
        prop_r_max_p_01 = max_range_heat_map.above_treshold_heat_index(0.1)
        prop_r_max_p_0_01 = max_range_heat_map.above_treshold_heat_index(0.01)
        prop_r_max_p_0_001 = max_range_heat_map.above_treshold_heat_index(0.001)

        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "prop_r=1|T=0.1",
            'predicted_verbatim': prop_r_1_p_0_05,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "prop_r=2|T=0.1",
            'predicted_verbatim': prop_r_2_p_0_05,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "prop_r=max|T=0.1",
            'predicted_verbatim': prop_r_max_p_01,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "prop_r=max|T=0.01",
            'predicted_verbatim': prop_r_max_p_0_01,
        }), ignore_index=True
        )
        result_df = result_df.append(dict(test_case_info, **{
            'algorithm_name': "prop_r=max|T=0.001",
            'predicted_verbatim': prop_r_max_p_0_001,
        }), ignore_index=True
        )

        result_df.to_csv('output/results.csv', index=False)
