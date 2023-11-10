import numpy as np
import pickle
import math
import json


class DataMonitor(object):
    def __init__(self, env) -> None:
        self.junction_list = env.junction_list
        self.directions_order = env.directions_order
        self.clear_data()

    def clear_data(self):
        self.conduct_traj_recorder()
        self.conduct_data_recorder()

    def conduct_traj_recorder(self):
        self.traj_record = dict()
        for junc_id in self.junction_list:
            self.traj_record[junc_id] = dict()
            for direction in self.directions_order:
                self.traj_record[junc_id][direction] = dict()
        self.max_t = 0
        self.max_x = 0

    def conduct_data_recorder(self):
        self.data_record = dict()
        self.conflict_rate = []
        self.overall_fuel_record = [] 
        self.overall_co2_record = []
        self.overall_co_record = []
        self.overall_hc_record = []
        self.overall_nox_record = []

        for junc_id in self.junction_list:
            self.data_record[junc_id] = dict()
            for direction in self.directions_order :
                self.data_record[junc_id][direction] = dict()
                self.data_record[junc_id][direction]['t'] = [i for i in range(5000)]
                self.data_record[junc_id][direction]['queue_wait'] = np.zeros(5000)
                self.data_record[junc_id][direction]['queue_length'] = np.zeros(5000)
                self.data_record[junc_id][direction]['control_queue_wait'] = np.zeros(5000)
                self.data_record[junc_id][direction]['control_queue_length'] = np.zeros(5000)
                self.data_record[junc_id][direction]['throughput_av'] = np.zeros(5000)
                self.data_record[junc_id][direction]['throughput'] = np.zeros(5000)
                self.data_record[junc_id][direction]['throughput_hv'] = np.zeros(5000)
                self.data_record[junc_id][direction]['conflict'] = np.zeros(5000)
                self.data_record[junc_id][direction]['global_reward'] = np.zeros(5000)
                self.data_record[junc_id][direction]['fuel_consumption'] = np.zeros(5000)
                self.data_record[junc_id][direction]['co2_emissions'] = np.zeros(5000)
                self.data_record[junc_id][direction]['co_emissions'] = np.zeros(5000)
                self.data_record[junc_id][direction]['hc_emissions'] = np.zeros(5000)
                self.data_record[junc_id][direction]['nox_emissions'] = np.zeros(5000)

    def step(self, env):
        t = env.env_step
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                self.data_record[junc_id][direction]['queue_length'][t] = env.get_queue_len(junc_id, direction, 'all')
                self.data_record[junc_id][direction]['queue_wait'][t] = env.get_avg_wait_time(junc_id, direction, 'all')
                self.data_record[junc_id][direction]['control_queue_length'][t] = env.get_queue_len(junc_id, direction, 'rv')
                self.data_record[junc_id][direction]['control_queue_wait'][t] = env.get_avg_wait_time(junc_id, direction, 'rv')
                self.data_record[junc_id][direction]['throughput'][t] = len(env.inner_lane_newly_enter[junc_id][direction])
                self.data_record[junc_id][direction]['conflict'][t] = len(env.conflict_vehids)
                self.data_record[junc_id][direction]['global_reward'][t] = env.global_obs[junc_id]
                self.data_record[junc_id][direction]['fuel_consumption'][t] = env.get_avg_dir_fuel(junc_id, direction)
                self.data_record[junc_id][direction]['co2_emissions'][t] = env.get_avg_junc_co2(junc_id, direction)
                self.data_record[junc_id][direction]['co_emissions'][t] = env.get_avg_junc_co(junc_id, direction)
                self.data_record[junc_id][direction]['hc_emissions'][t] = env.get_avg_junc_hc(junc_id, direction)
                self.data_record[junc_id][direction]['nox_emissions'][t] = env.get_avg_junc_nox(junc_id, direction)
        
        self.conflict_rate.extend([len(env.conflict_vehids)/len(env.previous_action) if len(env.previous_action) else 0])
        self.overall_fuel_record.extend([env.get_avg_fuel_consumption()])
        self.overall_co2_record.extend([env.get_avg_co2_emissions()])
        self.overall_co_record.extend([env.get_avg_co_emissions()])
        self.overall_hc_record.extend([env.get_avg_hc_emissions()])
        self.overall_nox_record.extend([env.get_avg_nox_emissions()])

    def evaluate(self, env, save_traj = False, min_step = 500, max_step = 1000):
        overall_wait = []
        for junc_id in self.junction_list:
            total_wait = []
            for direction in self.directions_order:
                avg_wait = np.mean(self.data_record[junc_id][direction]['queue_wait'][min_step:max_step])
                total_wait.extend([avg_wait])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG WAITING TIME AT JUNCTION {junc_id} {direction}: {avg_wait}\n")
                print(f"AVG WAITING TIME AT JUNCTION {junc_id} {direction}: {avg_wait}\n")

            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG WAITING TIME AT JUNCTION {junc_id}: {np.mean(total_wait)}\n\n")
            print(f"OVERALL AVG WAITING TIME AT JUNCTION {junc_id}: {np.mean(total_wait)}\n\n")

            overall_wait.extend([np.mean(total_wait)])
        
        avg_overall_wait = np.mean(overall_wait)
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG WAIT TIME OF NETWORK: {avg_overall_wait}\n\n")
        print(f"OVERALL AVG WAIT TIME OF NETWORK: {avg_overall_wait}\n\n")
        
        total_fuel = []
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                avg_fuel_consumption = np.mean(self.data_record[junc_id][direction]['fuel_consumption'][min_step:max_step])
                total_fuel.extend([avg_fuel_consumption])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG FUEL CONSUMPTION AT JUNCTION {junc_id} {direction}: {avg_fuel_consumption}\n")
                print(f"AVG FUEL CONSUMPTION AT JUNCTION {junc_id} {direction}: {avg_fuel_consumption}\n")

            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG FUEL CONSUMPTION AT JUNCTION {junc_id}: {np.mean(total_fuel)}\n\n")    
            print(f"OVERALL AVG FUEL CONSUMPTION AT JUNCTION {junc_id}: {np.mean(total_fuel)}\n\n")

        avg_overall_fuel = np.mean(self.overall_fuel_record[min_step:max_step])
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG FUEL CONSUMPTION OF NETWORK: {avg_overall_fuel}\n\n")
        print(f"OVERALL AVG FUEL CONSUMPTION OF NETWORK: {avg_overall_fuel}")

        total_co2 = []
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                avg_co2_emissions = np.mean(self.data_record[junc_id][direction]['co2_emissions'][min_step:max_step])
                total_co2.extend([avg_co2_emissions])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG CO2 EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_co2_emissions}\n")
                print(f"AVG CO2 EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_co2_emissions}\n")
            
            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG CO2 EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_co2)}\n\n")
            print(f"OVERALL AVG CO2 EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_co2)}\n\n")

        avg_overall_co2 = np.mean(self.overall_co2_record[min_step:max_step])
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG CO2 EMISSIONS OF NETWORK: {avg_overall_co2}\n\n")
        print(f"OVERALL AVG CO2 EMISSIONS OF NETWORK: {avg_overall_co2}")

        total_co = []
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                avg_co_emissions = np.mean(self.data_record[junc_id][direction]['co_emissions'][min_step:max_step])
                total_co.extend([avg_co_emissions])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG CO EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_co_emissions}\n")
                print(f"AVG CO EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_co_emissions}\n")
            
            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG CO EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_co)}\n\n")
            print(f"OVERALL AVG CO EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_co)}\n\n")

        avg_overall_co = np.mean(self.overall_co_record[min_step:max_step])
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG CO EMISSIONS OF NETWORK: {avg_overall_co}\n\n")
        print(f"OVERALL AVG CO EMISSIONS OF NETWORK: {avg_overall_co}")

        total_hc = []
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                avg_hc_emissions = np.mean(self.data_record[junc_id][direction]['hc_emissions'][min_step:max_step])
                total_hc.extend([avg_hc_emissions])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG HC EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_hc_emissions}\n")
                print(f"AVG HC EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_hc_emissions}\n")
            
            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG HC EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_hc)}\n\n")
            print(f"OVERALL AVG HC EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_hc)}\n\n")

        avg_overall_hc = np.mean(self.overall_hc_record[min_step:max_step])
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG HC EMISSIONS OF NETWORK: {avg_overall_hc}\n\n")
        print(f"OVERALL AVG HC EMISSIONS OF NETWORK: {avg_overall_hc}")

        total_nox = []
        for junc_id in self.junction_list:
            for direction in self.directions_order:
                avg_nox_emissions = np.mean(self.data_record[junc_id][direction]['nox_emissions'][min_step:max_step])
                total_nox.extend([avg_nox_emissions])

                with open("eval_results/all_results.txt", "a") as file1:
                    file1.write(f"AVG NOX EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_nox_emissions}\n")
                print(f"AVG NOX EMISSIONS AT JUNCTION {junc_id} {direction}: {avg_nox_emissions}\n")
            
            with open("eval_results/all_results.txt", "a") as file1:
                file1.write(f"OVERALL AVG NOX EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_nox)}\n\n")
            print(f"OVERALL AVG NOX EMISSIONS AT JUNCTION {junc_id}: {np.mean(total_nox)}\n\n")

        avg_overall_nox = np.mean(self.overall_nox_record[min_step:max_step])
        with open("eval_results/all_results.txt", "a") as file1:
            file1.write(f"OVERALL AVG NOX EMISSIONS OF NETWORK: {avg_overall_nox}\n\n")
        print(f"OVERALL AVG NOX EMISSIONS OF NETWORK: {avg_overall_nox}")

        with open("eval_results/avg_results.csv", "a") as file2:
            file2.write(f"{avg_overall_wait},{avg_overall_fuel},{avg_overall_co2},{avg_overall_co},{avg_overall_hc},{avg_overall_nox}\n")

        if save_traj:
            with open("eval_results/eval_trajectory.json", "a") as file3:
                json.dump(env.trajectory, file3)

    def eval_traffic_flow(self, junc_id, time_range):
        inflow_intersection = []
        for t in range(time_range[0], time_range[1]):
            inflow_intersection.extend([0])
            for direction in self.directions_order:
                 inflow_intersection[-1] += self.data_record[junc_id][direction]['throughput'][t]
        return inflow_intersection, max(inflow_intersection), sum(inflow_intersection)/len(inflow_intersection)

    def save_to_pickle(self, file_name):
        saved_dict = {'data_record':self.data_record, 'junctions':self.junction_list, 'direction':self.directions_order}
        with open(file_name, "wb") as f:
            pickle.dump(saved_dict, f)
