from typing import Dict
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self) -> None:
        board_matrix, alight_matrix, combined_matrix = self.get_demand_df_from_csv(
            1, 1, 1, True)
        self.dispatch_headway_mean: Dict[str, float] = {
            'B2': 200, 'B2A': 200, 'B3': 300, 'B5/B5K': 300, 'B16': 300, 'B20': 270, 'B19': 480.0, 'B21': 218.2}
        self.dispatch_headway_cv: Dict[str, float] = {
            'B2': 1.099, 'B2A': 0.879, 'B3': 0.984, 'B5/B5K': 0.254, 'B16': 0.639, 'B20': 0.936, 'B19': 1.0, 'B21': 1.08}

        self.dispatch_headway_mean = {
            route_id: H_mean for route_id, H_mean in self.dispatch_headway_mean.items()}

    def get_demand_df_from_csv(
        self, board_amp_factor, alight_amp_factor, bus_flow_amp_factor, is_separate
    ):
        file_str = "busoperation/setup/guangzhou_brt_data/demand.csv"
        df = pd.read_csv(file_str)

        # 1. fillin line-2 and line-2A in stop "sdjd"
        self.fill_a_with_b(0, 3, "sdjd", "hjxc", df)
        self.fill_a_with_b(1, 3, "sdjd", "hjxc", df)
        # 2. fillin line-16 in stop "hjxc"
        self.fill_a_with_b(4, 3, "hjxc", "sdjd", df)
        # 3. fill in line-3 in stop "ss", "hjxc", "sdjd", "gd"
        self.fill_a_with_b(2, 1, "ss", "xy", df)
        self.fill_a_with_b(2, 1, "hjxc", "xy", df)
        self.fill_a_with_b(2, 1, "sdjd", "xy", df)
        self.fill_a_with_b(2, 1, "gd", "xy", df)
        # 4. fill in line-16 and line-20 (although never reach) in stop "gd"
        # fill_a_with_b(4, 3, "gd", "sdjd", df)
        # fill_a_with_b(5, 3, "gd", "sdjd", df)

        board_df = df[
            (df["abs_line"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) & (
                df["type"] == "board")
        ]
        board_matrix = board_df.iloc[0:10, 3:13].to_numpy()
        board_matrix *= board_amp_factor

        alight_df = df[
            (df["abs_line"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) & (
                df["type"] == "alight")
        ]
        alight_matrix = alight_df.iloc[0:10, 3:13].to_numpy()
        alight_matrix *= alight_amp_factor
        alight_matrix /= bus_flow_amp_factor

        # print(board_matrix.sum())
        # print(alight_matrix)

        combined_matrix = board_matrix + alight_matrix
        # 2-d arrays, one is stop -> ln, the other is ln->stop
        # return np.transpose(board_df.to_numpy()), alight_df.to_numpy()

        return (
            np.transpose(board_matrix),
            np.transpose(alight_matrix),
            np.transpose(combined_matrix),
        )

    def fill_a_with_b(self, abs_line_no, ref_abs_line_no, fill_stop, ref_stop, df):
        b_idx, a_idx = abs_line_no * 2, abs_line_no * 2 + 1
        ref_b_idx, ref_a_idx = ref_abs_line_no * 2, ref_abs_line_no * 2 + 1
        b_ratio, a_ratio = (
            df.loc[ref_b_idx, fill_stop] / df.loc[ref_b_idx, ref_stop],
            df.loc[ref_a_idx, fill_stop] / df.loc[ref_a_idx, ref_stop],
        )
        df.loc[b_idx, fill_stop] = b_ratio * df.loc[b_idx, ref_stop]
        df.loc[a_idx, fill_stop] = a_ratio * df.loc[a_idx, ref_stop]

    def alight_fill_a_with_b(self, abs_line_no, ref_abs_line_no, fill_stop, ref_stop, df):
        ratio = df.loc[ref_abs_line_no, fill_stop] /\
            df.loc[ref_abs_line_no, ref_stop]
        df.loc[abs_line_no, fill_stop] = ratio * df.loc[abs_line_no, ref_stop]
