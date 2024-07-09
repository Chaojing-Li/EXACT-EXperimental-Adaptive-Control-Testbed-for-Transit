import pickle
import pandas as pd
from scipy.stats import norm
from copy import deepcopy
import os


class DataLoader:
    def __init__(self) -> None:
        self.data = pickle.load(
            open('setup/chengdu_route_3_data/data.pickle', 'rb'))
        self.tt_data = pickle.load(
            open('setup/chengdu_route_3_data/distribution.pickle', 'rb'))
        self.virtual_data = pickle.load(
            open('setup/chengdu_route_3_data/data_virtual.pickle', 'rb'))
        self.lambda_data = pickle.load(
            open('setup/chengdu_route_3_data/lamda_station.pickle', 'rb'))
        self.spacing_data = pickle.load(
            open('setup/chengdu_route_3_data/spacing.pickle', 'rb'))
        self.day_ids = [8, 9, 10]

    @property
    def trip_times(self):
        trip_times = []
        for day_id in self.day_ids:
            df = self.data['travel_time_{}'.format(day_id)]
            df['trip_time_seconds'] = df['trip_time'].dt.total_seconds()
            trip_time_seconds_list = df['trip_time_seconds'].tolist()
            trip_times.extend(trip_time_seconds_list)
        return trip_times

    @property
    def node_ids(self):
        node_ids = [str(x) for x in self.data['station_list']]
        return node_ids

    @property
    def virtual_bus_rtd_info(self):
        df = deepcopy(self.virtual_data)
        df['ACTDATETIME_8'] = pd.to_datetime(df['ACTDATETIME_8'])
        df['ACTDATETIME_9'] = pd.to_datetime(df['ACTDATETIME_9'])
        df['ACTDATETIME_10'] = pd.to_datetime(df['ACTDATETIME_10'])
        df['time_diff_ACTDATETIME_8'] = (
            df['ACTDATETIME_8'] - df['ACTDATETIME_8'].iloc[0]).dt.total_seconds()
        df['time_diff_ACTDATETIME_9'] = (
            df['ACTDATETIME_9'] - df['ACTDATETIME_9'].iloc[0]).dt.total_seconds()
        df['time_diff_ACTDATETIME_10'] = (
            df['ACTDATETIME_10'] - df['ACTDATETIME_10'].iloc[0]).dt.total_seconds()
        df['mean_time_diff'] = df[['time_diff_ACTDATETIME_8',
                                   'time_diff_ACTDATETIME_9', 'time_diff_ACTDATETIME_10']].mean(axis=1)
        df['std_time_diff'] = df[['time_diff_ACTDATETIME_8',
                                  'time_diff_ACTDATETIME_9', 'time_diff_ACTDATETIME_10']].std(axis=1)
        stop_rtd_time_info = dict(df.set_index('stationnum').apply(
            lambda row: (row['mean_time_diff'], row['std_time_diff']), axis=1))

        stop_rtd_time_info = {str(k): v for k, v in stop_rtd_time_info.items()}

        return stop_rtd_time_info

    @property
    def stop_pax_arrival_rate(self):
        df = deepcopy(self.lambda_data)
        stop_pax_arrival_rate = dict(zip(df['station_id'], df['lamda']))
        stop_pax_arrival_rate = {
            str(k): v/60 for k, v in stop_pax_arrival_rate.items()}
        return stop_pax_arrival_rate

    @property
    def link_time_info(self):
        link_time_info = {}
        for stop_id, params in zip(self.tt_data['station_num'], self.tt_data['params']):
            link_time_info[str(stop_id)] = params['norm']
        return link_time_info

    @property
    def spacing(self):
        link_spacing = {}
        for stop_id, spacing in zip(self.spacing_data['station_num'], self.spacing_data['spacing']):
            link_spacing[str(stop_id)] = spacing
        return link_spacing

    @property
    def dispatching_headway(self):
        Hs = []
        for day_id in self.day_ids:
            df = self.data['dep_fre_{}'.format(day_id)]
            # Convert the 'dep_fre' column to timedelta format
            df['dep_fre'] = pd.to_timedelta(df['dep_fre'])
            # Convert timedelta to seconds
            df['dep_fre_seconds'] = df['dep_fre'].dt.total_seconds()
            Hs.extend(df['dep_fre_seconds'].values.tolist())

        mu, std = norm.fit(Hs)
        return int(mu), std
