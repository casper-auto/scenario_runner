#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import random
from collections import deque
import math

from srunner.custom_agents.controller_custom import SteeringController
from srunner.custom_agents.frenet_utils import *

import numpy as np

def get_speed_ms(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2)

def get_accel_ms(vehicle):
    acc = vehicle.get_acceleration()
    return math.sqrt(acc.x ** 2 + acc.y ** 2)

def global_to_local(ref_orig, orientation, p):
    delta = np.array([p.x - ref_orig.x, p.y - ref_orig.y])

    s = math.sin(math.radians(-orientation))
    c = math.cos(math.radians(-orientation))

    out = np.array([delta[0] * c - delta[1] * s,
    delta[0] * s + delta[1] * c])

    return out

def clip_fnc(value, lower, upper):
    return lower if value < lower else upper if value > upper else value


def is_car_in_front_and_close(target_location, current_location, orientation, max_distance):

    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = math.sqrt(target_vector[0]*target_vector[0] + target_vector[1]*target_vector[1])       
    #print (norm_target)
    #print (target_location.x, target_location.y, current_location.x, current_location.y)

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = 0.0
    if (norm_target > 0.0):
        #print ('forward_vector: ', forward_vector)
        #print ('target_vector: ', target_vector)
        #print ('norm_target', norm_target)
        cos = np.dot(forward_vector, target_vector) / norm_target
        cos = clip_fnc(cos, -1, 1)   

        d_angle = math.degrees(math.acos(cos))

    car_in_front = d_angle < 90.0
    car_close = norm_target < max_distance
    car_in_front_close = car_in_front and car_close

    return car_in_front_close, norm_target

class FrontVehicle():
    def __init__(self):
        self._s = 1000
        self._v = 300.0

class IDMMergingAgent(Agent):
    """
    IDMMergingAgent implements a basic agent with idm behavior that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    It uses PID for lateral motion planning and IDM for longitudinal planning.
    - IDM parameters need to be initialized randomly at the begining in the acceptable range
    - FOV (longitudinal and lateral threshold) should be initialized randomely
    - Cooperativeness (change mind) parameter should be initialized randomely
    """

    def __init__(self, vehicle, start_wp, cooperative_driver = False, dense_traffic = False, speed_limit = 25):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(IDMMergingAgent, self).__init__(vehicle)

        self._dt = 0.05
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._k_t_look_ahead = 1.0

        #Front Car Perception
        self._front_car_range = 50.0  # meters

        #Merging location and map variables
        self._hop_resolution = 2.0
        self._target_wp = None

        self._loc_mrg = carla.Location(20.8,114.1,3)
        self._merging_s_ad = None
        self._grp_ad = None
        self._path_ad_list = None
        self._s_map_ad = None
        self._vehicle_AD = None

        self._path_agts_list = None
        self._s_map_ego = None
        self._grp = None
        self._merging_s_agts = None
        
        self._start_interaction_ad_s = -40

        #IDM parameters
        #self._idm_s_0 = random.uniform(0.5, 2.0)
        #self._idm_t_headway = random.uniform(0.5, 2.0)
        if (dense_traffic):
            self._idm_s_0 = random.uniform(0.5, 1.25)
            self._idm_t_headway = random.uniform(0.5, 1.25)
        else:
            self._idm_s_0 = random.uniform(1.0, 2.5)
            self._idm_t_headway = random.uniform(1.0, 2.5)

        # print("Building merging lane vehicle")

        # self._idm_v_ref = random.uniform(2.5, 3.0)
        self._idm_v_ref = random.uniform(0.8 * speed_limit, speed_limit)	
        self._idm_acceleration_max = random.uniform(2.0, 3.5)
        self._idm_deceleration = random.uniform(2.0, 4.0)
        self._idm_acceleration_exp = 4

        #cooperativeness
        #self._cooperativeness = random.uniform(0.0, 1.0)
        if (cooperative_driver):
            self._cooperativeness = random.uniform(0.25, 1.0)
        else:
            self._cooperativeness = random.uniform(0.0, 0.75)

        self._cooperating = self._cooperativeness > 0.5
        self._change_of_mind_thld = 0.9

        # Default PID
        self._K_P_def = 1
        self._K_D_def = 0.01
        self._K_I_def = 1.0
        self._e_buffer_def = deque(maxlen=20)
        self._prev_error = 0.0
        self._prev_throttle = 0.0

        #controller
        self._steering_control = SteeringController(
            self._vehicle, **args_lateral_dict)

        self._start_wp = start_wp
        self._current_s_pos = None


    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))
        
        route_trace = self._trace_route(self._start_wp, end_waypoint)
        assert route_trace

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        
        route.append([end_waypoint, None])
        self._target_wp = end_waypoint
        self._path_agts_list, self._s_map_agts = route_to_list(route)

        # int_val = 0
        # for wp, _ in route:
        #     print("i: %d Wp: [%f, %f]" % (int_val, wp.transform.location.x, wp.transform.location.y))
        #     int_val += 1        

        ind_merging = closest_point_ind(self._path_agts_list, self._loc_mrg.x, self._loc_mrg.y)
        self._merging_s_agts = self._s_map_agts[ind_merging]

        return route

    def _trace_route_ad(self, start_waypoint, end_waypoint, ad_vehicle):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp_ad is None:
            dao = GlobalRoutePlannerDAO(ad_vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp_ad = grp

        # Obtain route plan
        route = self._grp_ad.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        
        route.append([end_waypoint, None])
        self._path_ad_list, self._s_map_ad = route_to_list(route)

        # int_val = 0
        # for wp, _ in route:
        #     print("i: %d Wp: [%f, %f]" % (int_val, wp.transform.location.x, wp.transform.location.y))
        #     int_val += 1

        ind_merging = closest_point_ind(self._path_ad_list, self._loc_mrg.x, self._loc_mrg.y)
        self._merging_s_ad = self._s_map_ad[ind_merging]

    def distance_to_front_vehicle(self, target_location, current_location):
        """
        Return distance to front car.

        """
        dx = target_location.x - current_location.x
        dy = target_location.y - current_location.y
        return math.sqrt (dx * dx + dy * dy)

    def calculate_idm_acc(self, front_car):
        acc = 0
        ego_velocity = get_speed_ms(self._vehicle)
        delta_v = ego_velocity - front_car._v

        #Todo: to replace 5 with 0.5 * (ego_car_length + front_car_length)
        delta_s = front_car._s - 5
        if (delta_s < 0.5):
            delta_s = 0.5

        s_star = self._idm_s_0 + ego_velocity * self._idm_t_headway + ego_velocity * delta_v / (2 * math.sqrt (self._idm_acceleration_max * self._idm_deceleration))
        if (s_star < self._idm_s_0):
            s_star = self._idm_s_0

        v_term = math.pow((ego_velocity/self._idm_v_ref), self._idm_acceleration_exp)
        s_term = math.pow((s_star/delta_s), 2)

        acc = self._idm_acceleration_max * (1 - v_term - s_term )
        acc = clip_fnc(acc, -self._idm_deceleration, self._idm_acceleration_max)

        return acc

    def throttle_brake_pid_control(self, current_spd, acc_idm):
        """
        Estimate the throttle of the vehicle based on the PID equations
        """

        throttle = 0.0
        brake = 0.0       

        
        if acc_idm > 0:

            tgt_spd = current_spd + acc_idm * 2 * self._dt

            
            # Default controller

            # _e = (tgt_spd - current_spd)
            # self._e_buffer_def.append(_e)

            # if len(self._e_buffer_def) >= 2:
            #     _de = (self._e_buffer_def[-1] - self._e_buffer_def[-2]) / self._dt
            #     _ie = sum(self._e_buffer_def) * self._dt
            # else:
            #     _de = 0.0
            #     _ie = 0.0

            # throttle = clip_fnc(self._K_P_def * _e + (self._K_D_def * _de / self._dt) + (self._K_I_def * _ie * self._dt), 0.0, 1.0)

            
            # # Throttle incremental controller

            kp = 0.05 # CONSTANT TO BE MOVED TO CONSTRUCTOR
            ki = 0.05 # CONSTANT TO BE MOVED TO CONSTRUCTOR
            kd = 0.2 # CONSTANT TO BE MOVED TO CONSTRUCTOR

            # kp = 0.006; # CONSTANT TO BE MOVED TO CONSTRUCTOR
            # ki = 0.006; # CONSTANT TO BE MOVED TO CONSTRUCTOR
            # kd = 0.015; # CONSTANT TO BE MOVED TO CONSTRUCTOR

            v_error = tgt_spd - current_spd

            self._e_buffer_def.append(v_error)

            p_term = kp * v_error
            i_term = ki * sum(self._e_buffer_def) * self._dt
            d_term = kd * (v_error - self._prev_error) / self._dt

            throttle = clip_fnc(self._prev_throttle + p_term + i_term + d_term, 0.0, 1.0)
            
            self._prev_throttle = throttle
            self._prev_error = v_error

            # print("P: %f, I: %f, D: %f" % (p_term, i_term, d_term))
            
        else:

            brake = clip_fnc(round((-acc_idm / self._idm_deceleration) * 2) / 2, 0.0, 1.0)
            self._e_buffer_def.clear
            self._prev_throttle = 0.0
            self._prev_error = 0.0

        
        return throttle, brake


    def ad_vehicle_coop(self):

        ad_cooperating = False
        dist_ad = 1000

        ad_vehicle_transform = self._vehicle_AD.get_transform()
        ad_vehicle_location = ad_vehicle_transform.location
        ad_vehicle_rotation = ad_vehicle_transform.rotation

        if self._path_ad_list == None:
            ad_vehicle_waypoint = self._map.get_waypoint(ad_vehicle_location)
            self._trace_route_ad(ad_vehicle_waypoint, self._target_wp, self._vehicle_AD)            

        ad_frenet_s, ad_frenet_d, valid_conv_ad = get_frenet(ad_vehicle_location.x, ad_vehicle_location.y, self._path_ad_list, self._s_map_ad)

        if not(valid_conv_ad):
            return False, 0.0

        ego_vehicle_transform = self._vehicle.get_transform()
        ego_vehicle_location = ego_vehicle_transform.location
        ego_vehicle_rotation = ego_vehicle_transform.rotation     

        ego_frenet_s, ego_frenet_d, valid_conv_ego = get_frenet(ego_vehicle_location.x, ego_vehicle_location.y, self._path_agts_list, self._s_map_agts)

        if not(valid_conv_ego):
            return False, 0.0

        # We use 0.0 as the merging point, pre-merge(-), post--merge(+)
        ad_proj_val = ad_frenet_s - self._merging_s_ad
        ego_proj_val = ego_frenet_s - self._merging_s_agts

        # Only start interaction when car can see you
        if ad_proj_val < self._start_interaction_ad_s:
            return False, 0.0

        if ad_proj_val > ego_proj_val:# - 5.0:

            # If the car has cut far enough always cooperate
            if ad_proj_val > 0.0:
                ad_cooperating = True
                dist_ad = ad_proj_val - ego_proj_val
            else: 
                #Car is on the interactive area
                change_of_mind_val = random.uniform(0.0, 1.0)
                ad_cooperating = self._cooperating
                if change_of_mind_val > self._change_of_mind_thld:
                    ad_cooperating = not ad_cooperating                    

                dist_ad = ad_proj_val - ego_proj_val
                self._cooperating = ad_cooperating

            return ad_cooperating, dist_ad
        else:
            return False, 0.0

    def get_front_vehicle(self, vehicle_list, ego_s):

        min_distance = 1e10
        front_vehicle = None
        found_front_vehicle = False

        for target_vehicle in vehicle_list:

            loc = target_vehicle.get_location()

            # do not account for the ego vehicle or AD vehicle
            if target_vehicle.id == self._vehicle.id or target_vehicle.id == self._vehicle_AD.id:
                continue

            # Discard cars that are off in the z direction
            if loc.z < -10 or loc.z > 10:
                continue

            # if the object is not in our road it's not an obstacle
            target_frenet_s, target_frenet_d, valid_conv_ego = get_frenet(loc.x, loc.y, self._path_agts_list, self._s_map_agts)

            # Select the front car
            dist_car = target_frenet_s - ego_s 
            if dist_car > 0 and abs(target_frenet_d) < 3.5:
                if dist_car < min_distance:
                    min_distance = dist_car
                    front_vehicle = target_vehicle
                    found_front_vehicle = True

        return found_front_vehicle, front_vehicle, min_distance

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # print("Run step for veh: %d" % (self._vehicle.id))

        # Init control
        control_var = carla.VehicleControl()
        control_var.throttle = 0.0
        control_var.brake = 0.0
        control_var.steer = 0.0
        control_var.hand_brake = False
        control_var.manual_gear_shift = False

        if not self._path_agts_list or not self._s_map_agts:
            print("Frenet path has not been set yet. Return empty control")
            return control_var


        # Ego vehicle speed
        ego_spd = get_speed_ms(self._vehicle)

        # Don't do anything for the flying cars
        ego_vehicle_location = self._vehicle.get_location()
        if ego_vehicle_location.z < -10 or ego_vehicle_location.z > 10:
            return control_var
        else:
            # Look ahead distance
            look_ahead_distance = max(self._k_t_look_ahead * ego_spd, self._hop_resolution)

            # Get lookahead using frenet formulas
            ego_frenet_s, _, valid_conv_ego = get_frenet(ego_vehicle_location.x, ego_vehicle_location.y, self._path_agts_list, self._s_map_agts)
            self._current_s_pos = ego_frenet_s

            if not valid_conv_ego:
                print("Conversion to frenet issue")
                return control_var

            x_target, y_target, valid_conv_ego = get_xy(ego_frenet_s + look_ahead_distance, 0.0, self._path_agts_list, self._s_map_agts)

            if not valid_conv_ego:
                print("Conversion to frenet issue")
                return control_var

            target_loc = carla.Location(x_target, y_target, 0.0)

            control_var.steer = self._steering_control.run_step(target_loc)                

        #a virtual car in far distance
        idm_front_vehicle = FrontVehicle()
        idm_ad_vehicle = FrontVehicle()

        # aux_initialized_vars
        front_vehicle_detected = False
        ad_veh_coop = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")

        #Figure out AD vehicle ID
        if not self._vehicle_AD:
            vehicle_filter = "vehicle.lincoln2020.*"
            ad_vehicle_list = self._world.get_actors().filter(vehicle_filter)
            len_ad_list = len(ad_vehicle_list)
            if len_ad_list == 1:
                self._vehicle_AD = ad_vehicle_list[0]
            elif len_ad_list == 0:
                print("No ego vehicles")
            else:
                print("Too many cars in AD list")        

        # Check for cars in front of you and select closest
        front_vehicle_detected, vehicle_front, dist_front = self.get_front_vehicle(vehicle_list, ego_frenet_s)

        #Check if AD car is on adjacent lane and cooperating
        ad_veh_coop, dist_ad_veh = self.ad_vehicle_coop()

        if front_vehicle_detected:
            idm_front_vehicle._s = dist_front
            idm_front_vehicle._v = get_speed_ms(vehicle_front)

        idm_acc_front = self.calculate_idm_acc(idm_front_vehicle)
        # print("Front vehicle - s: %f, v: %f, idm_acc: %f" % (idm_front_vehicle._s, idm_front_vehicle._v, idm_acc_front))

        if ad_veh_coop:
            idm_ad_vehicle._s = dist_ad_veh
            idm_ad_vehicle._v = get_speed_ms(self._vehicle_AD)
            idm_acc_ad = self.calculate_idm_acc(idm_ad_vehicle)
            # print("AD vehicle - s: %f, v: %f, idm_acc: %f" % (idm_ad_vehicle._s, idm_ad_vehicle._v, idm_acc_ad))
        else:
            idm_acc_ad = 1000

        #if ad_veh_coop and idm_acc_ad < idm_acc_front:
        #    print("*************************************************************************************************** Vehicle %d is cooperating" % self._vehicle.id)


        idm_acc_final = min(idm_acc_front, idm_acc_ad)
        # print("Applied IDM acc: %f" % (idm_acc_final))


        # Control variables
        self._state = AgentState.NAVIGATING
        
        control_var.throttle, control_var.brake = self.throttle_brake_pid_control (ego_spd, idm_acc_final)

        # print("V: %f, Vmax: %f, Acc_idm: %f, Throttle: %f, Brake: %f" % (ego_spd, self._idm_v_ref, idm_acc_final, control_var.throttle, control_var.brake))

        return control_var
