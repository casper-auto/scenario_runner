#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import random
from collections import deque
import math

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import get_speed

from srunner.custom_agents.controller_custom import SteeringController
from srunner.custom_agents.frenet_utils import *

import numpy as np
import time

import matplotlib.pyplot as plt
from copy import deepcopy

def get_speed_ms(vehicle):
    return get_speed(vehicle) / 3.6

def get_accel_ms(vehicle):
    acc = vehicle.get_acceleration()
    return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

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

class FrontAgent():
    def __init__(self):
        self._s = 1000
        self._v = 300.0

class IDMDriveDistanceAgent(Agent):
    """
    IDMDriveDistanceAgent implements a basic agent with idm behavior that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    It uses PID for lateral motion planning and IDM for longitudinal planning.
    - IDM parameters need to be initialized randomly at the begining in the acceptable range
    - FOV (longitudinal and lateral threshold) should be initialized randomely
    - Cooperativeness (change mind) parameter should be initialized randomely
    """

    def __init__(self, vehicle, start_wp, reference_speed=None, cooperative_driver=False, dense_traffic=False, name="NoName", demo_id=-1):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(IDMDriveDistanceAgent, self).__init__(vehicle)

        # If no input for demo_id we assume it's a regular case
        self._demo_scenario = demo_id != -1

        #If we are in a demo scenario. Use the names to identify which vehicles have specific behaviors
        if self._demo_scenario:
            if name != "NoName":
                # Cars that are always not cooperating
                if name == "CloserVeh0" or name == "FurtherVeh3" or name == "FurtherVeh2":
                    self._narrow_flag = True
                    self._aggression_flag = True

                # Cars cooperating based on case
                # Case 0 - Third car cooperates, 4th does not?
                elif name == "FurtherVeh0" and demo_id == 0:
                    self._narrow_flag = False
                    self._aggression_flag = random.uniform(0.0, 1.0) > 0.75
                # Case 1 - Third car does not cooperates, 4th does
                elif name == "FurtherVeh1" and demo_id == 1:
                    self._narrow_flag = False
                    self._aggression_flag = random.uniform(0.0, 1.0) > 0.75
                # Case 2 - Third car does not cooperates, 4th does
                elif name == "FurtherVeh1" and demo_id == 2:
                    self._narrow_flag = False
                    self._aggression_flag = random.uniform(0.0, 1.0) > 0.75
                else:
                    # It is always friendly but it will become non friendly if                  
                    self._narrow_flag = False            
                    self._aggression_flag = False
                    
            else:
                print("For demo scenario all cars need a identifying name")
        else:
            #Random for all cars for scenarios that are not demo scenario
            # inattentive/attentive logic
            # narrow flag -> Inattentive
            self._narrow_flag = random.uniform(0.0, 1.0) > 0.67
            
            # aggression flag
            self._aggression_flag = random.uniform(0.0, 1.0) > 0.5

        # Adi's behavior variable initialization
        # narrow->Inattentive setting and wide->Attentive setting
        self._vehicle_width = 2.0#1.0
        self._vehicle_length = 2.5
        self._narrow_conditions = (self._vehicle_length * 4, self._vehicle_width)
        self._wide_conditions = (self._vehicle_length * 4, self._vehicle_width * 5.0)
        # patience time(4,7,10s)
        self._patience_list = [1.5, 3.0, 4.5]
        self._patience_t = self._patience_list[random.randint(0, 2)]

        
        #Variable initialization
        self._dt = 0.05
        self._state = AgentState.NAVIGATING
        self._reference_speed = reference_speed
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._k_t_look_ahead = 1.0

        self._grp = None
        self._vehicle_AD = None
        self._hop_resolution = 2.0
        self._target_wp = None

        # Front Car Perception
        self._front_car_range = 50.0  # meters

        # IDM parameters
        # self._idm_s_0 = random.uniform(0.5, 2.0)
        # self._idm_t_headway = random.uniform(0.5, 2.0)
        if dense_traffic:
            self._idm_s_0 = random.uniform(0.5, 1.25)
            self._idm_t_headway = random.uniform(0.5, 1.25)
        else:
            self._idm_s_0 = random.uniform(1.0, 2.5)
            self._idm_t_headway = random.uniform(1.0, 2.5)

        if self._reference_speed:
            self._idm_v_ref = self._reference_speed
        else:
            self._idm_v_ref = random.uniform(2.5, 3.0)

        print('building other lane vehicle')

        self._idm_acceleration_max = random.uniform(2.0, 3.5)
        self._idm_deceleration = random.uniform(1.5, 2.5)
        self._idm_acceleration_exp = random.uniform(3.5, 4.5)

        # cooperativeness
        # self._cooperativeness = random.uniform(0.0, 1.0)
        if cooperative_driver:
            self._cooperativeness = random.uniform(0.25, 1.0)
        else:
            self._cooperativeness = random.uniform(0.0, 0.75)

        self._long_thresh = 10.0
        self._thld_brake_ad = 1.5 # lat thresh
        self._perception_range = random.uniform(self._thld_brake_ad, 2.75)
        self._cooperating = self._cooperativeness > 0.5
        self._change_of_mind_thld = 0.9

        # Default PID
        self._e_buffer_def = deque(maxlen=20)
        self._prev_error = 0.0
        self._prev_throttle = 0.0

        #controller
        self._steering_control = SteeringController(
            self._vehicle, **args_lateral_dict)

        self._start_wp = start_wp

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

        return route

    def distance_to_front_vehicle(self, target_location, current_location):
        """
        Return distance to front car.

        """
        dx = target_location.x - current_location.x
        dy = target_location.y - current_location.y
        return math.sqrt(dx * dx + dy * dy)

    def calculate_idm_acc(self, front_car):
        acc = 0
        ego_velocity = clip_fnc(get_speed_ms(self._vehicle), 0.0, 10.0)
        delta_v = ego_velocity - front_car._v

        # Todo: to replace 5 with 0.5 * (ego_car_length + front_car_length)
        delta_s = front_car._s - 5
        if delta_s < 0.5:
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
            
            # # Throttle incremental controller

            kp = 0.05 # CONSTANT TO BE MOVED TO CONSTRUCTOR
            ki = 0.05 # CONSTANT TO BE MOVED TO CONSTRUCTOR
            kd = 0.2  # CONSTANT TO BE MOVED TO CONSTRUCTOR
            
            v_error = tgt_spd - current_spd

            self._e_buffer_def.append(v_error)

            p_term = kp * v_error
            i_term = ki * sum(self._e_buffer_def) * self._dt
            d_term = kd * (v_error - self._prev_error) / self._dt

            throttle = clip_fnc(self._prev_throttle + p_term + i_term + d_term, 0.0, 1.0)
            
            self._prev_throttle = throttle
            self._prev_error = v_error
           
        else:

            brake = clip_fnc(round((-acc_idm / self._idm_deceleration) * 2) / 2, 0.0, 1.0)
            self._e_buffer_def.clear
            self._prev_throttle = 0.0
            self._prev_error = 0.0

        
        return throttle, brake


    def ad_vehicle_coop(self, target_vehicle):

        ad_cooperating = False
        dist_ad = 1000

        loc = target_vehicle.get_location()
        target_vehicle_waypoint = self._map.get_waypoint(loc)
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)


        # print("ego_vehicle", target_vehicle.get_transform().rotation.yaw) # ego_vehicle_waypoint.lane_width, ego_vehicle_waypoint.transform.location.x, ego_vehicle_location.x)
        # print("road", ego_vehicle_waypoint.transform.rotation.yaw)
        # plt.plot()
        loc_wp_target = ego_vehicle_waypoint.transform.location
        yaw_ref = ego_vehicle_waypoint.transform.rotation.yaw
        ad_in_ego_ref = global_to_local(loc_wp_target, yaw_ref, loc)

        # AD POINTS
        ad_yaw = np.deg2rad(target_vehicle.get_transform().rotation.yaw)
        length = 2.5
        width = 1.0
        loc_front = carla.Location(x=loc.x + length*np.cos(ad_yaw), y=loc.y + length*np.sin(ad_yaw))
        loc_back = carla.Location(x=loc.x - length*np.cos(ad_yaw), y=loc.y - length*np.sin(ad_yaw))
        ad_front_in_ego_ref = global_to_local(loc_wp_target, yaw_ref, loc_front)
        ad_back_in_ego_ref = global_to_local(loc_wp_target, yaw_ref, loc_back)

        maxx = np.max([ad_in_ego_ref[0], ad_front_in_ego_ref[0], ad_back_in_ego_ref[0]])
        if maxx>0:
            minx = np.min([ad_in_ego_ref[0], ad_front_in_ego_ref[0], ad_back_in_ego_ref[0]])
        else: 
            minx = 1000
        miny = np.min([np.abs(ad_in_ego_ref[1]), np.abs(ad_front_in_ego_ref[1]), np.abs(ad_back_in_ego_ref[1]) ])
        
        #Actions by behavior models
        if self._narrow_flag: #Inattentive
            if minx < self._narrow_conditions[0] and miny < self._narrow_conditions[1]:
                ad_cooperating = True
                dist_ad = minx
                #print("narrow")
            else:
                ad_cooperating = False
                dist_ad = minx
        else:
            if minx < self._wide_conditions[0] and miny < self._wide_conditions[1]:
                # Agression LOGIC
                ad_cooperating = False
                dist_ad = minx
                if self._aggression_flag: #Aggressive
                    #self._idm_t_headway = self._idm_t_headway / 2.0
                    self._idm_v_ref+=10
                    self._narrow_flag = True
                    #print("wide and aggressive")
                else: #Friendly
                    ad_cooperating = True
                    dist_ad = minx
                    self._patience_t -= self._dt
                    if self._patience_t < 0:
                        #print("change aggeressive")
                        self._aggression_flag = True
                    if miny < self._narrow_conditions[1]:
                        #print("change narrow")
                        self._narrow_flag = True
            else:
                ad_cooperating = False
                dist_ad = minx

        '''
        print("minx:{}, miny:{}".format(minx, miny))
        # ALWAYS STOP IF IN FRONT 
        if minx < self._long_thresh and miny<self._thld_brake_ad:
            # print("braking!")
            ad_cooperating = True
            dist_ad = minx

        elif minx < self._long_thresh*1.75 and miny<self._perception_range:
            ad_cooperating = True
            dist_ad = minx

            # Car is on the interactive area
            change_of_mind_val = random.uniform(0.0, 1.0)
            ad_cooperating = self._cooperating
            if change_of_mind_val > self._change_of_mind_thld:
                ad_cooperating = not ad_cooperating
                #self._cooperating = not self._cooperating
                #print("ID: %d - Changed his mind" % (self._vehicle.id))
            
            # print("loc dist", minx, miny, self._perception_range)
        '''

        return ad_cooperating, dist_ad

    def get_front_agent(self, agent_list, ego_s, margin=0):

        min_distance = 1e10
        front_agent = None
        found_front_agent = False

        for target_agent in agent_list:

            loc = target_agent.get_location()

            # do not account for the ego agent or AD agent
            if target_agent.id == self._vehicle.id or target_agent.id == self._vehicle_AD.id:
                continue

            # Discard cars that are off in the z direction
            if loc.z < -10 or loc.z > 10:
                continue

            # if the object is not in our road it's not an obstacle
            target_frenet_s, target_frenet_d, valid_conv_ego = get_frenet(loc.x, loc.y, self._path_agts_list, self._s_map_agts)

            # Select the front car
            dist_car = target_frenet_s - ego_s
            half_car_width = target_agent.bounding_box.extent.y
            
            if dist_car > -self._vehicle.bounding_box.extent.x and abs(target_frenet_d) < (self._start_wp.lane_width / 2.0 + half_car_width + margin):
                if dist_car < min_distance:
                    min_distance = dist_car
                    front_agent = target_agent
                    found_front_agent = True

        return found_front_agent, front_agent, min_distance

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
        idm_front_vehicle = FrontAgent()
        idm_ad_vehicle = FrontAgent()
        idm_ped = FrontAgent()

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
        front_vehicle_detected, vehicle_front, dist_front = self.get_front_agent(vehicle_list, ego_frenet_s)

        #Check if AD car is on adjacent lane and cooperating
        ad_veh_coop, dist_ad_veh = self.ad_vehicle_coop(self._vehicle_AD)

        if front_vehicle_detected:
            idm_front_vehicle._s = dist_front
            idm_front_vehicle._v = get_speed_ms(vehicle_front)

        idm_acc_front = self.calculate_idm_acc(idm_front_vehicle)
        #print("Front vehicle - s: %f, v: %f, idm_acc: %f" % (idm_front_vehicle._s, idm_front_vehicle._v, idm_acc_front))

        if ad_veh_coop:
            idm_ad_vehicle._s = dist_ad_veh # 0.2
            idm_ad_vehicle._v = get_speed_ms(self._vehicle_AD)
            idm_acc_ad = self.calculate_idm_acc(idm_ad_vehicle)
            #print("AD vehicle - s: %f, v: %f, idm_acc: %f" % (idm_ad_vehicle._s, idm_ad_vehicle._v, idm_acc_ad))
        else:
            idm_acc_ad = 1000

        
        #if ad_veh_coop and idm_acc_ad < idm_acc_front:
        #    print("*************************************************************************************************** Vehicle %d is cooperating" % self._vehicle.id)

        # Check for pedestrians in your lane
        pedestrian_list = actor_list.filter("*walker*")
        # Check for pedestrians in front of you and select closest
        ped_front_detected, ped_front, dist_front_ped = self.get_front_agent(pedestrian_list, ego_frenet_s, margin = 1.2)#0.5) Modified to add pedestrians
        if ped_front_detected:
            if dist_front_ped<3:
                dist_front_ped=1000
            idm_ped._s = dist_front_ped
            idm_ped._v = 0.0
        if dist_front_ped>2:#1:
            idm_acc_ped = self.calculate_idm_acc(idm_ped)
       
        # Get the minimum of all the acelerations front vehicle, AD, front pedestrian
        idm_acc_final = min(min(idm_acc_front, idm_acc_ad), idm_acc_ped)
        #print("Applied IDM acc: %f" % (idm_acc_final))

        # print("{}, idm_ad:{}, idm_front:{},idm_final:{}".format(self._vehicle.id, idm_acc_ad,idm_acc_front,idm_acc_final ))
        # Control variables
        self._state = AgentState.NAVIGATING
        
        control_var.throttle, control_var.brake = self.throttle_brake_pid_control (ego_spd, idm_acc_final)

        # print("V: %f, Vmax: %f, Acc_idm: %f, Throttle: %f, Brake: %f" % (ego_spd, self._idm_v_ref, idm_acc_final, control_var.throttle, control_var.brake))

        return control_var
