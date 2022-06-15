from math import sqrt

import pandas as pd
from tqdm import tqdm

from src.dummy_index_map_creator import DummyIndexMapCreator
from src.heat_map_analysis import HeatMapAnalysis
from src.verbatim_heat_map_creator import VerbatimHeatMapCreator

index_map_creator = DummyIndexMapCreator((200, 200))
max_range = int(sqrt(200 ** 2 + 200 * 2))
_, max_noise_heat = VerbatimHeatMapCreator.noise_heat_statistics((200, 200), max_range, 0)
result_df = pd.DataFrame(columns=['run', 'real_verbatim', 'predicted_verbatim', 'test_case', 'category', 'algorithm_name'])

# Do 10 replications
for run in tqdm(range(10)):
    patches_05, copy_proportion_05 = index_map_creator.create_patch_map(num_patches=5)
    patches_10, copy_proportion_10 = index_map_creator.create_patch_map(num_patches=10)
    patches_20, copy_proportion_20 = index_map_creator.create_patch_map(num_patches=20)
    long_range_05, proportion_05 = index_map_creator.create_long_range_map(0.05)
    long_range_10, proportion_10 = index_map_creator.create_long_range_map(0.10)
    long_range_15, proportion_15 = index_map_creator.create_long_range_map(0.15)

    # Create different test cases
    test_cases = [
        {
            'map': index_map_creator.create_full_random_map(),
            'real_verbatim': 0,
            'name': 'Full random',
            'category': 'Full Random'
        },
        {
            'map': index_map_creator.create_full_verbatim_map(),
            'real_verbatim': 1,
            'name': 'Full verbatim',
            'category': 'Full Verbatim'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=3),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 3x3',
            'category': 'Checkerboard'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=1),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 1x1',
            'category': 'Checkerboard'
        },
        {
            'map': index_map_creator.create_checkerboard_map(square_size=3),
            'real_verbatim': 0.5,
            'name': 'Checkerboard 5x5',
            'category': 'Checkerboard'
        },
        {
            'map': patches_05,
            'real_verbatim': copy_proportion_05,
            'name': f'Patches 5',
            'category': 'Patches'
        },
        {
            'map': patches_10,
            'real_verbatim': copy_proportion_10,
            'name': f'Patches 10',
            'category': 'Patches'
        },
        {
            'map': patches_20,
            'real_verbatim': copy_proportion_20,
            'name': f'Patches 20',
            'category': 'Patches'
        },
        {
            'map': long_range_05,
            'real_verbatim': proportion_05,
            'name': f'Long range 0.05',
            'category': 'Long Range'
        },
        {
            'map': long_range_10,
            'real_verbatim': proportion_10,
            'name': f'Long range 0.10',
            'category': 'Long Range'
        },
        {
            'map': long_range_15,
            'real_verbatim': proportion_15,
            'name': f'Long range 0.15',
            'category': 'Long Range'
        }
    ]

    # Run different algorithms on the test cases
    for test_case in test_cases:
        test_map = test_case['map']
        heat_map_creator = VerbatimHeatMapCreator(test_map)
        real_verbatim = test_case['real_verbatim']
        test_case_name = test_case['name']
        category = test_case['category']
        test_case_info = {
            'real_verbatim': real_verbatim,
            'test_case': test_case_name,
            'category': category,
            'run': run
        }

        # Mean heat algorithms
        mean_heat_d1_r1 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(inv_dist_weight_exp=1, filter_radius=1)).mean_heat_value()
        mean_heat_d2_r1 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(inv_dist_weight_exp=2, filter_radius=1)).mean_heat_value()
        mean_heat_d1_r2 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(inv_dist_weight_exp=1, filter_radius=2)).mean_heat_value()
        mean_heat_d2_r2 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(inv_dist_weight_exp=2, filter_radius=2)).mean_heat_value()
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Mean Heat: d=1 | r=1",
                'predicted_verbatim': mean_heat_d1_r1,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Mean Heat: d=2 | r=2",
                'predicted_verbatim': mean_heat_d2_r2,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Mean Heat: d=1 | r=2",
                'predicted_verbatim': mean_heat_d1_r2,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Mean Heat: d=2 | r=1",
                'predicted_verbatim': mean_heat_d2_r1,
            }), ignore_index=True
        )

        # Proportional neighbour analysis algorithms
        prop_r_1_T_0_001 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=1, inv_dist_weight_exp=0)).above_treshold_heat_index(0.001)
        prop_r_2_T_0_001 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=2, inv_dist_weight_exp=0)).above_treshold_heat_index(0.001)
        prop_r_3_T_0_001 = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=3, inv_dist_weight_exp=0)).above_treshold_heat_index(0.001)
        max_range_heat_map = HeatMapAnalysis(heat_map_creator.get_verbatim_heat_map_filter_basis(filter_radius=max_range, inv_dist_weight_exp=0))
        prop_r_max_T_01 = max_range_heat_map.above_treshold_heat_index(0.1)
        prop_r_max_T_0_01 = max_range_heat_map.above_treshold_heat_index(0.01)
        prop_r_max_T_0_001 = max_range_heat_map.above_treshold_heat_index(0.001)
        prop_r_max_T_10mnh = max_range_heat_map.above_treshold_heat_index(10 * max_noise_heat)
        prop_r_max_T_100mnh = max_range_heat_map.above_treshold_heat_index(100 * max_noise_heat)

        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=1 | T=0.001",
                'predicted_verbatim': prop_r_1_T_0_001,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=2 | T=0.001",
                'predicted_verbatim': prop_r_2_T_0_001,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=3 | T=0.001",
                'predicted_verbatim': prop_r_3_T_0_001,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=0.1",
                'predicted_verbatim': prop_r_max_T_01,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=0.01",
                'predicted_verbatim': prop_r_max_T_0_01,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=0.001",
                'predicted_verbatim': prop_r_max_T_0_001,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=0.001",
                'predicted_verbatim': prop_r_max_T_0_001,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=10mnh",
                'predicted_verbatim': prop_r_max_T_10mnh,
            }), ignore_index=True
        )
        result_df = result_df.append(
            dict(test_case_info, **{
                'algorithm_name': "Prop Neigh r=max | T=100mnh",
                'predicted_verbatim': prop_r_max_T_100mnh,
            }), ignore_index=True
        )

        result_df.to_csv('output/results.csv', index=False)
