
from typing import Set
from ray.rllib.utils.typing import AgentID
from core.sumo_interface import SUMO

from core.costomized_data_structures import Vehicle, Container
from core.net_map import NetMap
import numpy as np

import random, math
from gym.spaces.box import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from copy import deepcopy
from core.utils import start_edges, end_edges, dict_tolist, UNCONFLICT_SET
from gymnasium.spaces import Discrete
from core.monitor import DataMonitor

WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
EPSILON = 0.00001

class Env(MultiAgentEnv):
    def __init__(self, config) -> None:
        ## TODO: use config to pass parameters

        super().__init__()
        self.config = config
        self.print_debug = False
        self.cfg = config['cfg']
        self.map_xml = config['map_xml']
        self.directions_order = ['topstraight', 'topleft','rightstraight', 'rightleft','bottomstraight', 'bottomleft', 'leftstraight', 'leftleft']
        self._max_episode_steps = 10000000 ## unlimited simulation horizon   
        if 'max_episode_steps' in config.keys():
            self._max_episode_steps = self.config['max_episode_steps']
        self.traffic_light_program = self.config['traffic_light_program']

        self.junction_list = self.config['junction_list']
        self.sumo_interface = SUMO(self.cfg, render=self.config['render'])
        self.map = NetMap(self.map_xml, self.junction_list)

        self.spawn_rl_prob = config['spawn_rl_prob']
        self.default_rl_prob = config['probablity_RL']
        self.rl_prob_list = config['rl_prob_range'] if 'rl_prob_range' in config.keys() else None

        self.start_edges = start_edges
        self.end_edges = end_edges

        self.max_acc = 100
        self.min_acc = -100
        self.control_distance = 30
        self.control_zone_length = 100
        self.max_wait_time = 200
        self.vehicle_len = 5.0
        
        self.init_env()
        self.previous_global_waiting = dict()
        self.global_obs = dict()

        for junc_id in self.junction_list:
            self.previous_global_waiting[junc_id] = dict()
            self.global_obs[junc_id] = 0
            for direction in self.directions_order:
                self.previous_global_waiting[junc_id][direction] = 0
                self.previous_global_waiting[junc_id]['sum'] = 0
        
        ## off, standard, flexible
        if 'conflict_mechanism' in config:
            self.conflict_resolve_mechanism_type = config['conflict_mechanism']
        else:
            self.conflict_resolve_mechanism_type = 'off'

        ## ego_only, wait_only, PN_ego
        # self.reward_mode = 'ego_only'
        # self.reward_mode = 'PN_ego'
        # self.reward_mode = 'wait_only'

    @property
    def n_obs(self):
        ## TODO defination of obs
        return 16+80
    
    @property
    def action_space(self):
        ## Continuous acceleration values
        # return Box(
        #     low=-2,
        #     high=2,
        #     shape=(1, ),
        #     dtype=np.float32
        # )

        ## Discretized acceleration space TODO: Need more bins
        # return Discrete(9) # discrete acceleration values

        ## Original action space
        return Discrete(2)

    @property
    def observation_space(self):
        return Box(
            low=-1,
            high=1,
            shape=(self.n_obs, ),
            dtype=np.float32)
    
    @property
    def env_step(self):
        return self._step


    def _print_debug(self, fun_str):
        if self.print_debug:
            print('exec: '+fun_str+' at time step: '+str(self._step))

    def get_agent_ids(self) -> Set[AgentID]:
        rl_veh_ids = []
        for veh in self.rl_vehicles:
            rl_veh_ids.extend([veh.id])
        return set(rl_veh_ids)

    def init_env(self):
        
        ## vehicle level
        self.vehicles = Container()
        self.rl_vehicles = Container()
        self.reward_record = dict()            
        self.veh_waiting_clock = dict()
        self.veh_waiting_juncs = dict()
        self.veh_name_mapping_table = dict()
        self.conflict_vehids=[]
        self.veh_fuel_consumption = dict()
        self.veh_co2_emissions = dict()
        self.veh_co_emissions = dict()
        self.veh_hc_emissions = dict()
        self.veh_nox_emissions = dict()
        self.veh_pmx_emissions = dict()

        # env level
        self._step = 0
        self.previous_obs = {}
        self.previous_action = {}
        self.previous_reward = {}
        self.previous_dones = {}

        # occupancy map
        self.inner_lane_obs = dict()
        self.inner_lane_occmap = dict()
        self.inner_lane_newly_enter = dict()
        for junc_id in self.junction_list:
            self.inner_lane_obs[junc_id] = dict()
            self.inner_lane_newly_enter[junc_id] = dict()
            self.inner_lane_occmap[junc_id] = dict()
            for direction in self.directions_order:
                self.inner_lane_obs[junc_id][direction] = []
                self.inner_lane_newly_enter[junc_id][direction] = []
                self.inner_lane_occmap[junc_id][direction] = [0 for _ in range(10)]

        # vehicle queue and control queue
        self.control_queue = dict()
        self.control_queue_waiting_time = dict()   
        self.control_fuel = dict()   
        self.control_speed = dict()  
        self.control_accel = dict()
        self.queue = dict()
        self.queue_waiting_time = dict()
        self.head_of_control_queue = dict()
        self.inner_speed = dict()
        for junc_id in self.junction_list:
            self.control_queue[junc_id] = dict()
            self.control_queue_waiting_time[junc_id] = dict()
            self.control_fuel[junc_id] = dict()
            self.control_speed[junc_id] = dict()
            self.control_accel[junc_id] = dict()
            self.queue[junc_id] = dict()
            self.queue_waiting_time[junc_id] = dict()
            self.head_of_control_queue[junc_id] = dict()
            self.inner_speed[junc_id] = []
            for direction in self.directions_order:
                self.control_queue[junc_id][direction] = []
                self.control_queue_waiting_time[junc_id][direction] = []
                self.control_fuel[junc_id][direction] = []
                self.control_speed[junc_id][direction] = []
                self.control_accel[junc_id][direction] = []
                self.queue[junc_id][direction] = []
                self.queue_waiting_time[junc_id][direction] = []
                self.head_of_control_queue[junc_id][direction] = []

        # fuel consumption and emissions
        self.fuel_consumption = dict()
        self.co2_emissions = dict()
        self.co_emissions = dict()
        self.hc_emissions = dict()
        self.nox_emissions = dict()
        self.pmx_emissions = dict()
        for junc_id in self.junction_list:
            self.fuel_consumption[junc_id] = dict()
            self.co2_emissions[junc_id] = dict()
            self.co_emissions[junc_id] = dict()
            self.hc_emissions[junc_id] = dict()
            self.nox_emissions[junc_id] = dict()
            self.pmx_emissions[junc_id] = dict()
            for direction in self.directions_order:
                self.fuel_consumption[junc_id][direction] = []
                self.co2_emissions[junc_id][direction] = []
                self.co_emissions[junc_id][direction] = []
                self.hc_emissions[junc_id][direction] = []
                self.nox_emissions[junc_id][direction] = []
                self.pmx_emissions[junc_id][direction] = []

        # trajectory dictionary
        self.trajectory = dict()
        self.trajectory['229'] = dict()
        self.trajectory['499'] = dict()
        self.trajectory['332'] = dict()
        self.trajectory['334'] = dict()

        ## global reward related        
        self.previous_global_waiting = dict()
        for junc_id in self.junction_list:
            self.previous_global_waiting[junc_id] = dict()
            for direction in self.directions_order:
                self.previous_global_waiting[junc_id][direction] = 0
                self.previous_global_waiting[junc_id]['sum'] = 0

        ## data monitor
        self.monitor = DataMonitor(self)

        self._print_debug('init_env')


    def get_avg_wait_time(self, junc_id, direction, mode = 'all'):
        ## mode = all, rv
        if mode == 'all':
            return np.mean(np.array(self.queue_waiting_time[junc_id][direction])) if len(self.queue_waiting_time[junc_id][direction])>0 else 0
        elif mode == 'rv':
            return np.mean(np.array(self.control_queue_waiting_time[junc_id][direction])) if len(self.control_queue_waiting_time[junc_id][direction])>0 else 0
        else:
            print('Error Mode in Queue Waiting time Calculation')
            return 0

    def get_avg_dir_fuel(self, junc_id, direction):
        return np.mean(np.array(self.fuel_consumption[junc_id][direction])) if len(self.fuel_consumption[junc_id][direction]) > 0 else 0
    
    def get_avg_control_junc_fuel(self, junc_id):
        if len(self.vehicles) == 0:
            return 0
        
        junc_fuel = []
        for direction in self.directions_order:
            junc_fuel.extend(self.control_fuel[junc_id][direction])

        if len(junc_fuel) == 0:
            return 0

        return float(np.mean(junc_fuel))
        
    def get_avg_fuel_consumption(self):
        if len(self.vehicles) == 0:
            return 0
         
        fuel_consumption = 0
        for veh in self.vehicles:
            fuel_consumption += self.sumo_interface.get_veh_fuel_consumption(self.vehicles[veh.id])
        
        avg_fuel_consumption = (fuel_consumption / len(self.vehicles))
        avg_fuel_consumption = avg_fuel_consumption if avg_fuel_consumption >= 0.0 else 0.0

        return avg_fuel_consumption
    
    def get_avg_control_junc_speed(self, junc_id):
        if len(self.vehicles) == 0:
            return 0
        
        junc_speed = []
        for direction in self.directions_order:
            junc_speed.extend(self.control_speed[junc_id][direction])

        if len(junc_speed) == 0:
            return 0

        return float(np.mean(junc_speed))
    
    def get_avg_control_junc_accel(self, junc_id):
        if len(self.vehicles) == 0:
            return 0
        
        junc_accel = []
        for direction in self.directions_order:
            junc_accel.extend(self.control_accel[junc_id][direction])

        if len(junc_accel) == 0:
            return 0

        return float(np.mean(junc_accel))
    
    def get_avg_junc_co2(self, junc_id, direction):
        return np.mean(np.array(self.co2_emissions[junc_id][direction])) if len(self.co2_emissions[junc_id][direction]) > 0 else 0
    
    def get_avg_co2_emissions(self):
        if len(self.vehicles) == 0:
            return 0
        
        co2_emissions = 0
        for veh in self.vehicles:
            co2_emissions += self.sumo_interface.get_veh_co2_emission(self.vehicles[veh.id])

        avg_co2_emissions = co2_emissions / len(self.vehicles)   
        avg_co2_emissions = avg_co2_emissions if avg_co2_emissions >= 0.0 else 0.0     

        return avg_co2_emissions
    
    def get_avg_junc_co(self, junc_id, direction):
        return np.mean(np.array(self.co_emissions[junc_id][direction])) if len(self.co_emissions[junc_id][direction]) > 0 else 0
    
    def get_avg_co_emissions(self):
        if len(self.vehicles) == 0:
            return 0
        
        co_emissions = 0
        for veh in self.vehicles:
            co_emissions += self.sumo_interface.get_veh_co_emission(self.vehicles[veh.id])

        avg_co_emissions = co_emissions / len(self.vehicles)   
        avg_co_emissions = avg_co_emissions if avg_co_emissions >= 0.0 else 0.0     

        return avg_co_emissions
    
    def get_avg_junc_hc(self, junc_id, direction):
        return np.mean(np.array(self.hc_emissions[junc_id][direction])) if len(self.hc_emissions[junc_id][direction]) > 0 else 0
    
    def get_avg_hc_emissions(self):
        if len(self.vehicles) == 0:
            return 0
        
        hc_emissions = 0
        for veh in self.vehicles:
            hc_emissions += self.sumo_interface.get_veh_hc_emission(self.vehicles[veh.id])

        avg_hc_emissions = hc_emissions / len(self.vehicles)   
        avg_hc_emissions = avg_hc_emissions if avg_hc_emissions >= 0.0 else 0.0     

        return avg_hc_emissions
    
    def get_avg_junc_nox(self, junc_id, direction):
        return np.mean(np.array(self.nox_emissions[junc_id][direction])) if len(self.nox_emissions[junc_id][direction]) > 0 else 0
    
    def get_avg_nox_emissions(self):
        if len(self.vehicles) == 0:
            return 0
        
        nox_emissions = 0
        for veh in self.vehicles:
            nox_emissions += self.sumo_interface.get_veh_nox_emission(self.vehicles[veh.id])

        avg_nox_emissions = nox_emissions / len(self.vehicles)   
        avg_nox_emissions = avg_nox_emissions if avg_nox_emissions >= 0.0 else 0.0     

        return avg_nox_emissions
    
    def get_avg_junc_pmx(self, junc_id, direction):
        return np.mean(np.array(self.pmx_emissions[junc_id][direction])) if len(self.pmx_emissions[junc_id][direction]) > 0 else 0
    
    def get_avg_pmx_emissions(self):
        if len(self.vehicles) == 0:
            return 0
        
        pmx_emissions = 0
        for veh in self.vehicles:
            pmx_emissions += self.sumo_interface.get_veh_pmx_emission(self.vehicles[veh.id])

        avg_pmx_emissions = pmx_emissions / len(self.vehicles)   
        avg_pmx_emissions = avg_pmx_emissions if avg_pmx_emissions >= 0.0 else 0.0     

        return avg_pmx_emissions

    def get_queue_len(self, junc_id, direction, mode='all'):
        ## mode = all, rv
        if mode == 'all':
            return len(self.queue[junc_id][direction])
        elif mode == 'rv':
            return len(self.control_queue[junc_id][direction])
        else:
            print('Error Mode in Queue Length Calculation')
            return 0

    # soft control to the stop line
    def soft_deceleration(self, veh):
        front_distance = self.map.edge_length(veh.road_id) - veh.laneposition
        exhibition = True
        if exhibition:
            front_distance = front_distance-3
            if front_distance < 5:
                return self.min_acc
            else:
                return -((veh.speed**2)/(2*front_distance+EPSILON))
        else:
            if front_distance < 16:
                return self.min_acc
            else:
                return -((veh.speed**2)/(2*front_distance+EPSILON))*10

    def rotated_directions_order(self, veh):
        if veh.road_id[0] != ':':
            facing_junction_id = self.map.get_facing_intersection(veh.road_id)
            if len(facing_junction_id) == 0:
                print('error in rotating')
                return self.directions_order
            else:
                dir, label = self.map.query_edge_direction(veh.road_id, veh.lane_index)
                if not label:
                    print("error in query lane direction and edge label")
                    return self.directions_order
                else:
                    ego_direction = label+dir
                    index = self.directions_order.index(ego_direction)
                    rotated_direction = []
                    for i in range(len(self.directions_order)):
                        rotated_direction.extend([self.directions_order[(i+index)%(len(self.directions_order)-1)]])
                    return rotated_direction
        else:
            for ind in range(len(veh.road_id)):
                if veh.road_id[len(veh.road_id)-1-ind] == '_':
                    break
            last_dash_ind = len(veh.road_id)-1-ind
            facing_junction_id = veh.road_id[1:last_dash_ind]
            dir, label = self.map.query_inner_edge_direction(veh.road_id, veh.lane_index)
            if not label:
                print("error in query lane direction and edge lable")
                return self.directions_order
            else:
                ego_direction = label+dir
                index = self.directions_order.index(ego_direction)
                rotated_direction = []
                for i in range(len(self.directions_order)):
                    rotated_direction.extend([self.directions_order[(i+index)%(len(self.directions_order)-1)]])
                return rotated_direction
            
    def change_conflict_mechanism_type(self, new_type):
        if not new_type in ['off', 'flexible', 'standard']:
            return False
        else:
            self.conflict_resolve_mechanism_type = new_type
            return True            
        
    def change_veh_route(self, veh_id, route):
        ## route should be a list of edge id
        self.sumo_interface.set_veh_route(veh_id, route)

    def change_rl_prob(self, rl_prob):
        ## assign all existing RL vehicle first
        changed_list = []
        if rl_prob < self.default_rl_prob:
            for veh in self.rl_vehicles:
                # route = self.routes[tc.vehicle.getRouteID(veh_id)]
                if random.random()>(rl_prob/self.default_rl_prob):
                    changed_list.extend([deepcopy(veh)])
                    self.vehicles[veh.id].type = 'IDM'
                    self.sumo_interface.set_color(self.vehicles[veh.id], WHITE)
            for veh in changed_list:
                self.rl_vehicles.pop(veh.id)
            self.change_default_spawn_rl_prob(rl_prob)
        else:
            self.change_default_spawn_rl_prob(rl_prob)
        return changed_list

    def change_default_spawn_rl_prob(self, prob):
        self.default_rl_prob = prob

    def change_spawn_rl_prob(self, edge_id, prob):
        self.spawn_rl_prob[edge_id] = prob

    def conflict_predetection(self, junc, ori):
        # detect potential conflict, refer to conflict resolving mechanism
        # input: junc:junction id, ori: moving direction
        # output: True: conflict or potential conflict, False: no conflict detected
        allowing_ori=[ori]
        for pair_set in UNCONFLICT_SET:
            if ori in pair_set:
                for k in pair_set:
                    if k!= ori:
                        allowing_ori.extend([k])
        if self.conflict_resolve_mechanism_type=='flexible':
            if ori in self.previous_global_waiting[junc]['largest']:
                for key in self.inner_lane_occmap[junc].keys():
                    if max(self.inner_lane_occmap[junc][key][:3])>0 and key not in allowing_ori:
                        return True
            else:
                for key in self.inner_lane_occmap[junc].keys():
                    if max(self.inner_lane_occmap[junc][key][:8])>0 and key not in allowing_ori:
                        return True
        elif self.conflict_resolve_mechanism_type=='standard':
            for key in self.inner_lane_occmap[junc].keys():
                if max(self.inner_lane_occmap[junc][key][:8])>0 and key not in allowing_ori:
                    return True
        elif self.conflict_resolve_mechanism_type=='off':
            pass
        else:
            pass
        return False

    def virtual_id_assign(self, veh_id):
        if not veh_id in self.veh_name_mapping_table.keys():
            self.veh_name_mapping_table[veh_id] = (veh_id, False)
            return veh_id
        else:
            if self.veh_name_mapping_table[veh_id][1]:
                virtual_new_id = veh_id+'@'+str(10*random.random())
                self.veh_name_mapping_table[veh_id] = (virtual_new_id, False)
                return virtual_new_id
            else:
                return self.veh_name_mapping_table[veh_id][0]
    
    def convert_virtual_id_to_real_id(self, virtual_id):
        return virtual_id.split('@')[0]

    def terminate_veh(self, virtual_id):
        real_id = virtual_id.split('@')[0]
        self.veh_name_mapping_table[real_id] = (self.veh_name_mapping_table[real_id][0], True)

    def need_to_control(self, veh):
        # determine whether the vehicles is inside the control zone
        return True if self.map.check_veh_location_to_control(veh) and \
                (self.map.edge_length(veh.road_id) - veh.laneposition) < self.control_distance \
                    else False

    def norm_value(self, value_list, max, min):
        for idx in range(len(value_list)):
            value_list[idx] = value_list[idx] if value_list[idx]<max else max
            value_list[idx] = value_list[idx] if value_list[idx]>min else min
        return np.array(value_list)/max

    def compute_reward(self, rl_veh, waiting_lst, action, junc, ori):

        self.reward_record[rl_veh.id] = dict()
        total_veh_control_queue = self._compute_total_num_control_queue(self.control_queue[junc])

        if not total_veh_control_queue:
            ## avoid empty queue at the beginning
            total_veh_control_queue = 1

        if action == 1:
            egoreward = waiting_lst[0]
        else:
            egoreward = -waiting_lst[0]
    
        ## Punish high fuel consumption
        egoreward = egoreward - (0.1 * self.get_avg_control_junc_fuel(junc))

        ## Reward higher average control speed
        avg_junc_speed = self.get_avg_control_junc_speed(junc)
        egoreward = egoreward + (0.1 * avg_junc_speed)

        ## Punish high average control accel
        avg_junc_accel = self.get_avg_control_junc_accel(junc)
        if avg_junc_accel > 0:
            egoreward = egoreward - (1.0 * avg_junc_accel)

        if rl_veh.id in self.conflict_vehids:
            ## punishing conflicting action
            egoreward = egoreward - 1

        ## global reward negative is bad, positive is good
        globalreward = self.global_obs[junc]
        self.reward_record[rl_veh.id]['ego'] = egoreward
        self.reward_record[rl_veh.id]['global'] = globalreward
        self.reward_record[rl_veh.id]['sum'] = egoreward + globalreward
        
        return egoreward

    def _traffic_light_program_update(self):
        if self._step> self.traffic_light_program['disable_light_start']:
            self.sumo_interface.disable_all_trafficlight(self.traffic_light_program['disable_state'])
  
    def compute_max_len_of_control_queue(self, junc_id):
        control_queue_len = []
        junc_info = self.control_queue[junc_id]
        for direction in self.directions_order:
            control_queue_len.extend([len(junc_info[direction])])
        return np.array(control_queue_len).max()

    def _compute_total_num_control_queue(self, junc_info):
        control_queue_len = []
        for direction in self.directions_order:
            control_queue_len.extend([len(junc_info[direction])])
        return sum(control_queue_len)

    def _update_obs(self):
        # clear the queues
        for junc_id in self.junction_list:
            self.inner_speed[junc_id] = []
            for direction in self.directions_order:
                self.control_queue[junc_id][direction] = []
                self.queue[junc_id][direction] = []
                self.head_of_control_queue[junc_id][direction] = []
                self.control_queue_waiting_time[junc_id][direction] = []
                self.queue_waiting_time[junc_id][direction] = []
                self.control_fuel[junc_id][direction] = []
                self.control_speed[junc_id][direction] = []
                self.control_accel[junc_id][direction] = []
                self.fuel_consumption[junc_id][direction] = []
                self.co2_emissions[junc_id][direction] = []
                self.co_emissions[junc_id][direction] = []
                self.hc_emissions[junc_id][direction] = []
                self.nox_emissions[junc_id][direction] = []
                self.pmx_emissions[junc_id][direction] = []
                
        # occupancy map
        self.inner_lane_obs = dict()
        self.inner_lane_occmap = dict()
        self.inner_lane_newly_enter = dict()
        for junc_id in self.junction_list:
            self.inner_lane_obs[junc_id] = dict()
            self.inner_lane_newly_enter[junc_id] = dict()
            self.inner_lane_occmap[junc_id] = dict()
            for direction in self.directions_order:
                self.inner_lane_obs[junc_id][direction] = []
                self.inner_lane_newly_enter[junc_id][direction] = []
                self.inner_lane_occmap[junc_id][direction] = [0 for _ in range(10)]


        for veh in self.vehicles:
            if len(veh.road_id)==0:
                ## avoid invalid vehicle information
                continue
            if veh.road_id[0] == ':':
                ## inside intersection: update inner obs and occmap
                direction, edge_label = self.map.query_inner_edge_direction(veh.road_id, veh.lane_index)
                for ind in range(len(veh.road_id)):
                    if veh.road_id[len(veh.road_id)-1-ind] == '_':
                        break
                last_dash_ind = len(veh.road_id)-1-ind
                if edge_label and veh.road_id[1:last_dash_ind] in self.junction_list:
                    self.inner_lane_obs[veh.road_id[1:last_dash_ind]][edge_label+direction].extend([veh])
                    self.inner_lane_occmap[veh.road_id[1:last_dash_ind]][edge_label+direction][min(int(10*veh.laneposition/self.map.edge_length(veh.road_id)), 9)] = 1
                    if veh not in self.prev_inner[veh.road_id[1:last_dash_ind]][edge_label+direction]:
                        self.inner_lane_newly_enter[veh.road_id[1:last_dash_ind]][edge_label+direction].extend([veh])
                    self.inner_speed[veh.road_id[1:last_dash_ind]].extend([veh.speed])
            else:
                ## update waiting time
                junc_id, direction = self.map.get_veh_moving_direction(veh)
                accumulating_waiting = veh.wait_time
                if len(junc_id) > 0:
                    if veh.id not in self.veh_waiting_juncs.keys():
                        self.veh_waiting_juncs[veh.id] = dict()
                        self.veh_waiting_juncs[veh.id][junc_id] = accumulating_waiting
                    else:
                        prev_wtm = 0
                        for prev_junc_id in self.veh_waiting_juncs[veh.id].keys():
                            if prev_junc_id != junc_id:
                                prev_wtm += self.veh_waiting_juncs[veh.id][prev_junc_id]
                        if accumulating_waiting - prev_wtm >= 0:
                            self.veh_waiting_juncs[veh.id][junc_id] = accumulating_waiting - prev_wtm
                        else:
                            self.veh_waiting_juncs[veh.id][junc_id] = accumulating_waiting
            
                ## updating control queue and waiting time of queue
                if self.map.get_distance_to_intersection(veh)<=self.control_zone_length:
                    self.queue[junc_id][direction].extend([veh])
                    self.queue_waiting_time[junc_id][direction].extend([self.veh_waiting_juncs[veh.id][junc_id]])
                    if veh.type == 'RL':
                        self.control_queue[junc_id][direction].extend([veh])
                        self.control_queue_waiting_time[junc_id][direction].extend([self.veh_waiting_juncs[veh.id][junc_id]])
                        self.control_fuel[junc_id][direction].extend([self.sumo_interface.get_veh_fuel_consumption(self.rl_vehicles[veh.id])])
                        self.control_speed[junc_id][direction].extend([self.sumo_interface.get_veh_speed(self.rl_vehicles[veh.id])])
                        self.control_accel[junc_id][direction].extend([self.sumo_interface.get_veh_accel(self.rl_vehicles[veh.id])])
            
            # Update fuel consumption and emissions for each intersection and direction
            if veh.road_id[0] == ':': # inside intersection 
                direction, edge_label = self.map.query_inner_edge_direction(veh.road_id, veh.lane_index)
                for ind in range(len(veh.road_id)):
                    if veh.road_id[len(veh.road_id)-1-ind] == '_':
                        break
                last_dash_ind = len(veh.road_id)-1-ind

                if edge_label and veh.road_id[1:last_dash_ind] in self.junction_list:
                    direction = edge_label+direction
                    junc_id = veh.road_id[1:last_dash_ind]

                    if len(junc_id) > 0: # Avoiding an issue where the junc_id is none of these
                        if veh.id not in self.veh_fuel_consumption.keys():
                            self.veh_fuel_consumption[veh.id] = dict()
                            self.veh_co2_emissions[veh.id] = dict()
                            self.veh_co_emissions[veh.id] = dict()
                            self.veh_hc_emissions[veh.id] = dict()
                            self.veh_nox_emissions[veh.id] = dict()
                            self.veh_pmx_emissions[veh.id] = dict()
                        
                        self.veh_fuel_consumption[veh.id][junc_id] = self.sumo_interface.get_veh_fuel_consumption(self.vehicles[veh.id])
                        self.fuel_consumption[junc_id][direction].extend([self.veh_fuel_consumption[veh.id][junc_id]])

                        self.veh_co2_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_co2_emission(self.vehicles[veh.id])
                        self.co2_emissions[junc_id][direction].extend([self.veh_co2_emissions[veh.id][junc_id]])

                        self.veh_co_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_co_emission(self.vehicles[veh.id])
                        self.co_emissions[junc_id][direction].extend([self.veh_co_emissions[veh.id][junc_id]])

                        self.veh_hc_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_hc_emission(self.vehicles[veh.id])
                        self.hc_emissions[junc_id][direction].extend([self.veh_hc_emissions[veh.id][junc_id]])

                        self.veh_nox_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_nox_emission(self.vehicles[veh.id])
                        self.nox_emissions[junc_id][direction].extend([self.veh_nox_emissions[veh.id][junc_id]])

                        self.veh_pmx_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_pmx_emission(self.vehicles[veh.id])
                        self.pmx_emissions[junc_id][direction].extend([self.veh_pmx_emissions[veh.id][junc_id]])

            else: # at or going to junction
                junc_id, direction = self.map.get_veh_moving_direction(veh)

                if len(junc_id) > 0 and junc_id in ["334", "332", "229", "499"]: # Avoiding an issue where the junc_id is none of these
                    if veh.id not in self.veh_fuel_consumption.keys():
                        self.veh_fuel_consumption[veh.id] = dict()
                        self.veh_co2_emissions[veh.id] = dict()
                        self.veh_co_emissions[veh.id] = dict()
                        self.veh_hc_emissions[veh.id] = dict()
                        self.veh_nox_emissions[veh.id] = dict()
                        self.veh_pmx_emissions[veh.id] = dict()
                    
                    if self.map.get_distance_to_intersection(veh) <= self.control_zone_length:
                        self.veh_fuel_consumption[veh.id][junc_id] = self.sumo_interface.get_veh_fuel_consumption(self.vehicles[veh.id])
                        self.fuel_consumption[junc_id][direction].extend([self.veh_fuel_consumption[veh.id][junc_id]])

                        self.veh_co2_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_co2_emission(self.vehicles[veh.id])
                        self.co2_emissions[junc_id][direction].extend([self.veh_co2_emissions[veh.id][junc_id]])

                        self.veh_co_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_co_emission(self.vehicles[veh.id])
                        self.co_emissions[junc_id][direction].extend([self.veh_co_emissions[veh.id][junc_id]])

                        self.veh_hc_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_hc_emission(self.vehicles[veh.id])
                        self.hc_emissions[junc_id][direction].extend([self.veh_hc_emissions[veh.id][junc_id]])

                        self.veh_nox_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_nox_emission(self.vehicles[veh.id])
                        self.nox_emissions[junc_id][direction].extend([self.veh_nox_emissions[veh.id][junc_id]])

                        self.veh_pmx_emissions[veh.id][junc_id] = self.sumo_interface.get_veh_pmx_emission(self.vehicles[veh.id])
                        self.pmx_emissions[junc_id][direction].extend([self.veh_pmx_emissions[veh.id][junc_id]])
                                
        ## update previous global waiting for next step reward calculation
        for junc_id in self.junction_list:
            weighted_sum = 0
            largest = 0
            for direction in self.directions_order:
                control_queue_length = self.get_queue_len(junc_id, direction, 'rv')
                waiting_time = self.get_avg_wait_time(junc_id, direction, 'rv')
                self.previous_global_waiting[junc_id][direction] = waiting_time
                if waiting_time >= largest:
                    self.previous_global_waiting[junc_id]['largest'] = [direction]
                    largest = waiting_time
                weighted_sum += waiting_time
            self.global_obs[junc_id] = (self.previous_global_waiting[junc_id]['sum'] - weighted_sum)/(self.previous_global_waiting[junc_id]['sum']*10+EPSILON)
            if self.global_obs[junc_id] < -1:
                self.global_obs[junc_id] = -1
            if self.global_obs[junc_id] > 1:
                self.global_obs[junc_id] = 1
            self.previous_global_waiting[junc_id]['sum'] = weighted_sum

    def _update_trajectory(self):   
        # Only concerned about evaluation period; Could be changed     
        if self.env_step >= 500 and self.env_step < 1000: # only collect data in the same range as eval results
            for veh in self.vehicles:
                junc_id, direction = "none", "none"

                if len(veh.road_id) == 0:
                    continue # avoid invalid vehicle information

                if veh.road_id[0] == ':': # inside intersection 
                    direction, edge_label = self.map.query_inner_edge_direction(veh.road_id, veh.lane_index)
                    for ind in range(len(veh.road_id)):
                        if veh.road_id[len(veh.road_id)-1-ind] == '_':
                            break
                    last_dash_ind = len(veh.road_id)-1-ind

                    if edge_label and veh.road_id[1:last_dash_ind] in self.junction_list:
                        direction = edge_label+direction
                        junc_id = veh.road_id[1:last_dash_ind]

                else: # at or going to junction
                    junc_id, direction = self.map.get_veh_moving_direction(veh)

                if junc_id in ["229", "499", "332", "334"]:
                    if veh.id not in self.trajectory[junc_id].keys():
                        self.trajectory[junc_id][veh.id] = dict()
                        self.trajectory[junc_id][veh.id]['time'] = []
                        self.trajectory[junc_id][veh.id]['speed'] = []
                        self.trajectory[junc_id][veh.id]['accel'] = []
                        self.trajectory[junc_id][veh.id]['dist_to_junc'] = []
                        self.trajectory[junc_id][veh.id]['fuel'] = []
                
                    self.trajectory[junc_id][veh.id]['time'].extend([self.env_step])
                    self.trajectory[junc_id][veh.id]['speed'].extend([veh.speed])
                    self.trajectory[junc_id][veh.id]['accel'].extend([veh.acceleration])
                    self.trajectory[junc_id][veh.id]['dist_to_junc'].extend([self.map.get_distance_to_intersection(veh)])
                    self.trajectory[junc_id][veh.id]['fuel'].extend([self.sumo_interface.get_veh_fuel_consumption(self.vehicles[veh.id])])


    def step_once(self, action={}):
        self._print_debug('step')
        self.new_departed = set()
        self.sumo_interface.set_max_speed_all(10)
        self._traffic_light_program_update()
        # check if the action input is valid
        if not (isinstance(action, dict) and len(action) == len(self.previous_obs)- sum(dict_tolist(self.previous_dones))):
            print('error!! action dict is invalid')
            return dict()
                
        ## Original action execution
        for virtual_id in action.keys():
            veh_id = self.convert_virtual_id_to_real_id(virtual_id)
            if action[virtual_id] == 1:
                junc_id, ego_dir = self.map.get_veh_moving_direction(self.rl_vehicles[veh_id])
                if self.conflict_predetection(junc_id, ego_dir):
                    ## conflict
                    self.sumo_interface.accl_control(self.rl_vehicles[veh_id], self.soft_deceleration(self.rl_vehicles[veh_id]))
                    self.conflict_vehids.extend([veh_id])
                else:
                    self.sumo_interface.accl_control(self.rl_vehicles[veh_id], self.max_acc)
            elif action[virtual_id] == 0:
                self.sumo_interface.accl_control(self.rl_vehicles[veh_id], self.soft_deceleration(self.rl_vehicles[veh_id]))
        
        # print(f"ACTIONS: {action}")
        ## Apply acceleration action 
        # for virtual_id in action.keys():
        #     veh_id = self.convert_virtual_id_to_real_id(virtual_id)

        #     junc_id, ego_dir = self.map.get_veh_moving_direction(self.rl_vehicles[veh_id])
        #     if self.conflict_predetection(junc_id, ego_dir): # Every time agent takes an agent, check if causes conflict
        #         self.conflict_vehids.extend([veh_id])

        #     if action[virtual_id] == 0:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], -2.0)
        #     elif action[virtual_id] == 1:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], -1.5)
        #     elif action[virtual_id] == 2:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], -1.0)
        #     elif action[virtual_id] == 3:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], -0.5)
        #     elif action[virtual_id] == 4:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], 0)
        #     elif action[virtual_id] == 5:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], 0.5)
        #     elif action[virtual_id] == 6:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], 1.0)
        #     elif action[virtual_id] == 7:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], 1.5)
        #     elif action[virtual_id] == 8:
        #         self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], 2.0)

        ## Apply continuous acceleration action
        # for virtual_id in action.keys():
        #     veh_id = self.convert_virtual_id_to_real_id(virtual_id)

        #     junc_id, ego_dir = self.map.get_veh_moving_direction(self.rl_vehicles[veh_id])
        #     if self.conflict_predetection(junc_id, ego_dir):
        #         self.conflict_vehids.extend([veh_id])
            
        #     self.sumo_interface.apply_accel(self.rl_vehicles[veh_id], action[virtual_id])
        
        # print(f"CONFLICT VEH IDS: {self.conflict_vehids}")


        #sumo step
        self.sumo_interface.step()

        # gathering states from sumo 
        sim_res = self.sumo_interface.get_sim_info()
        # setup for new departed vehicles    
        for veh_id in sim_res.departed_vehicles_ids:
            self.sumo_interface.subscribes.veh.subscribe(veh_id)
            length = self.sumo_interface.tc.vehicle.getLength(veh_id)
            route = self.sumo_interface.tc.vehicle.getRoute(veh_id)
            road_id  = self.sumo_interface.get_vehicle_edge(veh_id)
            if (road_id in self.spawn_rl_prob.keys() and random.random()<self.spawn_rl_prob[road_id]) or \
                (random.random()<self.default_rl_prob):
                self.rl_vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length)
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="RL", route=route, length=length, wait_time=0)
            else:
                self.vehicles[veh_id] = veh = Vehicle(id=veh_id, type="IDM", route=route, length=length, wait_time=0)
                
            self.sumo_interface.set_color(veh, WHITE if veh.type=="IDM" else RED)

            self.new_departed.add(veh)

        self.new_arrived = {self.vehicles[veh_id] for veh_id in sim_res.arrived_vehicles_ids}
        self.new_collided = {self.vehicles[veh_id] for veh_id in sim_res.colliding_vehicles_ids}
        self.new_arrived -= self.new_collided # Don't count collided vehicles as "arrived"

        # remove arrived vehicles from Env
        for veh in self.new_arrived:
            if veh.type == 'RL':
                self.rl_vehicles.pop(veh.id)
            self.vehicles.pop(veh.id)

        self._print_debug('before updating vehicles')
        # update vehicles' info for Env
        for veh_id, veh in self.vehicles.items():
            veh.prev_speed = veh.get('speed', None)
            veh.update(self.sumo_interface.subscribes.veh.get(veh_id))
            if veh.type == 'RL':
                self.rl_vehicles[veh_id].update(self.sumo_interface.subscribes.veh.get(veh_id))
            wt, _ = self.sumo_interface.get_veh_waiting_time(veh)
            if wt > 0:
                self.vehicles[veh_id].wait_time +=1

        ## update obs 
        self._update_obs()

        self._update_trajectory()

        obs = {}
        rewards = {}
        dones = {}

        for rl_veh in self.rl_vehicles:
            virtual_id = self.virtual_id_assign(rl_veh.id)
            if len(rl_veh.road_id) == 0:
                if virtual_id in action.keys():
                    ## collision occured and teleporting, I believe it should be inside the intersection
                    obs[virtual_id] = self.check_obs_constraint(self.previous_obs[virtual_id])
                    dones[virtual_id] = True
                    rewards[virtual_id] =  0
                    self.terminate_veh(virtual_id)
                    continue
                else:
                    ## then do nothing
                    continue
            junc_id, ego_dir = self.map.get_veh_moving_direction(rl_veh)
            if len(junc_id) == 0 or junc_id not in self.junction_list:
                # skip the invalid junc_id 
                continue

            ## Curr junc obs
            # junc_obs = []
            # obs_control_queue_length = []
            # obs_waiting_lst = []
            # obs_inner_lst = []
            # control_queue_max_len = self.compute_max_len_of_control_queue(junc_id) + EPSILON
        
            # for direction in self.rotated_directions_order(rl_veh):
            #     obs_control_queue_length.extend([self.get_queue_len(junc_id, direction, 'rv')/control_queue_max_len])
            #     obs_waiting_lst.extend([self.get_avg_wait_time(junc_id, direction, 'rv')])
            #     obs_inner_lst.append(self.inner_lane_occmap[junc_id][direction])

            ## Collecting obs from other junctions/directions
            # other_juncs = self.junction_list.copy()
            # other_juncs.remove(junc_id)
            # random.shuffle(other_juncs) # Preventing model overfitting to certain junction order

            # other_obs = [] # Order will be junc_queue, junc_wait, junc_map and then moves to next junction
            # other_temp_control_queue_length = []
            # other_temp_waiting_lst = []
            # other_temp_inner_lst = []

            # for other_junc_id in other_juncs:
            #     for direction in self.directions_order:
            #         other_temp_control_queue_length.extend([self.get_queue_len(other_junc_id, direction, 'rv')/control_queue_max_len])
            #         other_temp_waiting_lst.extend([self.get_avg_wait_time(other_junc_id, direction, 'rv')])
            #         other_temp_inner_lst.append(self.inner_lane_occmap[other_junc_id][direction])
            #     other_obs.extend(other_temp_control_queue_length)
            #     other_obs.extend(other_temp_waiting_lst)
            #     # other_obs.extend(np.reshape(np.array(other_temp_inner_lst), (80,)))

            #     other_temp_control_queue_length = []
            #     other_temp_waiting_lst = []
            #     other_temp_inner_lst = []

            # junc_obs.extend(obs_control_queue_length)
            # junc_obs.extend(obs_waiting_lst)
            # junc_obs.extend(np.reshape(np.array(obs_inner_lst), (80,)))

            # Combined obs
            # curr_obs = self.check_obs_constraint(np.concatenate((junc_obs, other_obs)))
            # curr_obs = self.check_obs_constraint(junc_obs)

            obs_control_queue_length = []
            obs_waiting_lst = []
            obs_inner_lst = []
            control_queue_max_len = self.compute_max_len_of_control_queue(junc_id) + EPSILON
            if self.need_to_control(rl_veh):
                ## need to control 
                ## average waiting time 
                for keyword in self.rotated_directions_order(rl_veh):
                    obs_control_queue_length.extend([self.get_queue_len(junc_id, keyword, 'rv')/control_queue_max_len])
                    obs_waiting_lst.extend([self.get_avg_wait_time(junc_id, keyword, 'rv')])
                    obs_inner_lst.append(self.inner_lane_occmap[junc_id][keyword])
                
                obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                if virtual_id in action.keys():
                    ## reward
                    rewards[virtual_id] = self.compute_reward(rl_veh, obs_waiting_lst, action[virtual_id], junc_id, ego_dir)
                obs[virtual_id] = self.check_obs_constraint(np.concatenate((obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst), (80,)))))
                dones[virtual_id] = False
            elif virtual_id in action.keys():
                ## update reward for the vehicle already enter intersection
                if rl_veh.road_id[0]==':':
                    ## inside the intersection
                    for keyword in self.rotated_directions_order(rl_veh):
                        obs_control_queue_length.extend([self.get_queue_len(junc_id, keyword, 'rv')/control_queue_max_len])
                        obs_waiting_lst.extend([self.get_avg_wait_time(junc_id, keyword, 'rv')])
                        obs_inner_lst.append(self.inner_lane_occmap[junc_id][keyword])

                    obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                    rewards[virtual_id] = self.compute_reward(rl_veh, obs_waiting_lst, action[virtual_id], junc_id, ego_dir)
                    dones[virtual_id] = True
                    obs[virtual_id] = self.check_obs_constraint(np.concatenate((obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst), (80,)))))
                    self.terminate_veh(virtual_id)
                else:
                    ## change to right turn lane and no need to control
                    for keyword in self.rotated_directions_order(rl_veh):
                        obs_control_queue_length.extend([self.get_queue_len(junc_id, keyword, 'rv')/control_queue_max_len])
                        obs_waiting_lst.extend([self.get_avg_wait_time(junc_id, keyword, 'rv')])
                        obs_inner_lst.append(self.inner_lane_occmap[junc_id][keyword])

                    obs_waiting_lst = self.norm_value(obs_waiting_lst, self.max_wait_time, 0)
                    rewards[virtual_id] = 0
                    dones[virtual_id] = True
                    obs[virtual_id] = self.check_obs_constraint(np.concatenate((obs_control_queue_length, np.array(obs_waiting_lst), np.reshape(np.array(obs_inner_lst), (80,)))))
                    self.terminate_veh(virtual_id)    

        dones['__all__'] = False
        infos = {}
        truncated = {}
        truncated['__all__'] = False
        if self._step >= self._max_episode_steps:
            for key in dones.keys():
                truncated[key] = True
        self._step += 1
        self.previous_obs, self.previous_reward, self.previous_action, self.previous_dones, self.prev_inner\
              = deepcopy(obs), deepcopy(rewards), deepcopy(action), deepcopy(dones), deepcopy(self.inner_lane_obs)
        self.monitor.step(self)

        self.conflict_vehids=[]
        self._print_debug('finish process step')
        if len(dict_tolist(rewards))>0 and self.print_debug:
            print('avg reward: '+str(np.array(dict_tolist(rewards)).mean())+' max reward: '+str(np.array(dict_tolist(rewards)).max())+' min reward: '+str(np.array(dict_tolist(rewards)).min()))

        return obs, rewards, dones, truncated, infos

    def step(self, action={}):
        if len(action) == 0:
            print("empty action")
        
        obs, rewards, dones, truncated, infos = self.step_once(action)

        # COMMENT OUT THIS PORTION IF DOING BASELINE SCRIPTS
        ## avoid empty obs or all agents are done during simulation
        all_done = True
        for id in dones.keys():
            if not dones[id] and id!='__all__':
                all_done = False
        
        if all_done:
            new_obs = {}
            while len(new_obs)==0:
                new_obs, new_rewards, new_dones, new_truncated, new_infos = self.step_once()
            for id in new_obs.keys():
                obs[id] = new_obs[id]
                dones[id] = new_dones[id]
        return obs, rewards, dones, truncated, infos


    def reset(self, *, seed=None, options=None):
        self._print_debug('reset')
        # soft reset
        while not self.sumo_interface.reset_sumo():
            pass

        if self.rl_prob_list:
            self.default_rl_prob = random.choice(self.rl_prob_list)
            print("new RV percentage = "+str(self.default_rl_prob))

        self.init_env()
        obs = {}
        if options:
            if options['mode'] == 'HARD':
                obs, _, _, _, infos = self.step_once()
                return obs, infos

        while len(obs)==0:
            obs, _, _, _, infos = self.step_once()
        return obs, infos

    def close(self):
        ## close env
        self.sumo_interface.close()

    def action_space_sample(self, agent_ids: list = None):
        self._print_debug('action sample')
        """Returns a random action for each environment, and potentially each
            agent in that environment.

        Args:
            agent_ids: List of agent ids to sample actions for. If None or
                empty list, sample actions for all agents in the
                environment.

        Returns:
            A random action for each environment.
        """

        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        return {
            agent_id: self.action_space.sample()
            for agent_id in agent_ids
            if agent_id != "__all__"
        }


    def observation_space_sample(self, agent_ids: list = None):
        self._print_debug('obs sample')
        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        return {
            agent_id: self.observation_space.sample()
            for agent_id in agent_ids
            if agent_id != "__all__"
        }


    def check_obs_constraint(self, obs):
        if not self.observation_space.contains(obs):
            obs= np.asarray([x if x>= self.observation_space.low[0] else self.observation_space.low[0] for x in obs]\
                                    , dtype=self.observation_space.dtype)
            obs = np.asarray([x if x<= self.observation_space.high[0] else self.observation_space.high[0] for x in obs]\
                                    , dtype=self.observation_space.dtype)
            if not self.observation_space.contains(obs):
                print('dddd')
                raise ValueError(
                    "Observation is invalid, got {}".format(obs)
                )
        return obs