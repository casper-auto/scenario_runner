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

from srunner.custom_agents.pid_class import *

import numpy as np
import time

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

class FrontVehicle():
    def __init__(self):
        self._s = 1000
        self._v = 300.0

class IDMLaneChangeAgent(Agent):
    """
    IDMLaneChangeAgent implements a basic agent with idm behavior that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    It uses PID for lateral motion planning and IDM for longitudinal planning.
    - IDM parameters need to be initialized randomly at the begining in the acceptable range
    - FOV (longitudinal and lateral threshold) should be initialized randomely
    - Cooperativeness (change mind) parameter should be initialized randomely
    """

    def __init__(self, vehicle, start_wp, reference_speed=None, cooperative_driver=False, dense_traffic=False):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(IDMLaneChangeAgent, self).__init__(vehicle)

        self._dt = 0.05
        self._state = AgentState.NAVIGATING
        self._reference_speed = reference_speed
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}

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

        self._thld_brake_ad = 1.0
        self._perception_range = random.uniform(self._thld_brake_ad, 1.75)
        self._cooperating = self._cooperativeness > 0.5
        self._change_of_mind_thld = 0.9

        # Throttle/Brake PID control
        # Todo: To be verified
        self._K_P = .4
        self._K_D = 0.4*self._dt
        self._K_I = 0.05/self._dt
        self.pid = PID(self._dt)
        self._e_buffer = deque(maxlen=10)

        # Default PID
        self._K_P_def = 1
        self._K_D_def = 0.0
        self._K_I_def = 0.0
        self._e_buffer_def = deque(maxlen=20)

        #controller
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed':self._idm_v_ref,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = self._idm_v_ref
        self._grp = None
        self.history = []

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

        self._local_planner.set_global_plan(route_trace)

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

    def throttle_brake_pid_control(self, current_spd, target_spd):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_acc:  target aceleration in m/s^2
        :param current_acc: current aceleration of the vehicle in m/s^2
        :return: throttle control in the range [-1.0, 1.0]
        """
        _e = target_spd - current_spd
        self._e_buffer_def.append(_e)

        if len(self._e_buffer_def) >= 2:
            _de = (self._e_buffer_def[-1] - self._e_buffer_def[-2]) / self._dt
            _ie = sum(self._e_buffer_def) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return clip_fnc((self._K_P_def * _e) + (self._K_D_def * _de / self._dt) + (self._K_I_def * _ie * self._dt), -1.0, 1.0)


    def velocity_pid_control(self, target_acc):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :param target_acc:  target aceleration in m/s^2
        :param current_acc: current aceleration of the vehicle in m/s^2
        :return: throttle control in the range [-1.0, 1.0]
        """
        speed = get_speed_ms(self._vehicle)
        target_speed = speed+target_acc*self._dt

        control = self.pid.control(speed, target_speed)

        return control

    def ad_vehicle_coop(self, target_vehicle):

        ad_cooperating = False
        dist_ad = 1000

        loc = target_vehicle.get_location()
        target_vehicle_waypoint = self._map.get_waypoint(loc)
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        # Only use ad car if it's on adjacent lane on the right (otherwise it is dealt as a front car)
        if target_vehicle_waypoint.road_id == ego_vehicle_waypoint.road_id and target_vehicle_waypoint.lane_id == ego_vehicle_waypoint.lane_id - 1:
            loc_wp_target = ego_vehicle_waypoint.transform.location
            yaw_ref = ego_vehicle_waypoint.transform.rotation.yaw
            ad_in_ego_ref = global_to_local(loc_wp_target, yaw_ref, loc)
            # print("AD pos in Ref: [%f, %f]" % (ad_in_ego_ref[0], ad_in_ego_ref[1]))

            # Position based cooperation
            # lateral_dist_min = ego_vehicle_waypoint.lane_width / 2 + target_vehicle.bounding_box.extent.y + self._perception_range
            lateral_dist_min = self._perception_range + target_vehicle.bounding_box.extent.y

            # longitudinal_dist_min = target_vehicle.bounding_box.extent.x / 2
            longitudinal_dist_min = target_vehicle.bounding_box.extent.x

            # lateral_hard_lim = ego_vehicle_waypoint.lane_width / 2 + target_vehicle.bounding_box.extent.y - self._thld_brake_ad
            lateral_hard_lim = self._thld_brake_ad + target_vehicle.bounding_box.extent.y

            # print("Cooperative distances - Long: %f, Lat inter: %f, Lat hard lim: %f" % (longitudinal_dist_min, lateral_dist_min, lateral_hard_lim))

            if ad_in_ego_ref[0] > longitudinal_dist_min:

                # If the car has cut far enough always cooperate
                if ad_in_ego_ref[1] < lateral_hard_lim:
                    ad_cooperating = True
                    dist_ad = ad_in_ego_ref[0]
                    ad_vehicle = target_vehicle

                elif ad_in_ego_ref[1] >= lateral_hard_lim and ad_in_ego_ref[1] < lateral_dist_min:
                    # Car is on the interactive area
                    change_of_mind_val = random.uniform(0.0, 1.0)
                    ad_cooperating = self._cooperating
                    if change_of_mind_val > self._change_of_mind_thld:
                        ad_cooperating = not ad_cooperating
                        #self._cooperating = not self._cooperating
                        #print("ID: %d - Changed his mind" % (self._vehicle.id))

                    dist_ad = ad_in_ego_ref[0]
                    ad_vehicle = target_vehicle

        return ad_cooperating, dist_ad

    def get_front_vehicle(self, vehicle_list):

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        min_distance = 1e10
        front_vehicle = None
        found_front_vehicle = False

        for target_vehicle in vehicle_list:

            loc = target_vehicle.get_location()

            # do not account for the ego vehicle
            if target_vehicle.id == self._vehicle.id:
                continue

            # Discard cars that are off in the z direction
            if loc.z < -10 or loc.z > 10:
                continue

            # if the object is not in our road it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(loc)
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            # Select the front car, either your lane or ad car if it's very close to the lane
            car_in_front = False
            if target_vehicle_waypoint.lane_id == ego_vehicle_waypoint.lane_id:
                car_in_front, dist_car = is_car_in_front_and_close(loc, ego_vehicle_location, self._vehicle.get_transform().rotation.yaw, self._front_car_range)
            else:
                continue

            if car_in_front: #or ad_vehicle_close
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

        # standard local planner behavior
        control = self._local_planner.run_step(debug=debug)

        # Don't do anything for the flying cars
        ego_vehicle_location = self._vehicle.get_location()
        if ego_vehicle_location.z < -10 or ego_vehicle_location.z > 10:
            control.brake = 0
            control.throtte = 0
            return control

        #print("Vehicle: %d" % self._vehicle.id)

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

        # Check for cars in front of you and select closest
        front_vehicle_detected, vehicle_front, dist_front = self.get_front_vehicle(vehicle_list)

        #Check if AD car is on adjacent lane and cooperating
        vehicle_filter = "vehicle.lincoln2020.*"
        ad_vehicle_list = self._world.get_actors().filter(vehicle_filter)
        len_ad_list = len(ad_vehicle_list)
        if len_ad_list == 1:
            ad_vehicle = ad_vehicle_list[0]
        elif len_ad_list == 0:
            print("No ego vehicles")
        else:
            print("Too many cars in AD list")
        ad_veh_coop, dist_ad_veh = self.ad_vehicle_coop(ad_vehicle)

        if front_vehicle_detected:
            idm_front_vehicle._s = dist_front
            idm_front_vehicle._v = get_speed_ms(vehicle_front)

        idm_acc_front = self.calculate_idm_acc(idm_front_vehicle)
        #print("Front vehicle - s: %f, v: %f, idm_acc: %f" % (idm_front_vehicle._s, idm_front_vehicle._v, idm_acc_front))

        if ad_veh_coop:
            idm_ad_vehicle._s = dist_ad_veh
            idm_ad_vehicle._v = get_speed_ms(ad_vehicle)
            idm_acc_ad = self.calculate_idm_acc(idm_ad_vehicle)
            #print("AD vehicle - s: %f, v: %f, idm_acc: %f" % (idm_ad_vehicle._s, idm_ad_vehicle._v, idm_acc_ad))
        else:
            idm_acc_ad = 1000

        #if ad_veh_coop and idm_acc_ad < idm_acc_front:
        #    print("*************************************************************************************************** Vehicle %d is cooperating" % self._vehicle.id)


        idm_acc_final = min(idm_acc_front, idm_acc_ad)
        #print("Applied IDM acc: %f" % (idm_acc_final))


        # Control variables
        self._state = AgentState.NAVIGATING
        ego_spd = get_speed_ms(self._vehicle)
        tgt_spd = ego_spd + idm_acc_final * 3 * self._dt

        throttle_brake = self.throttle_brake_pid_control (ego_spd, tgt_spd)
        # throttle_brake = self.velocity_pid_control(idm_acc_final)

        if (throttle_brake < 0):
            control.brake = -throttle_brake
            control.throttle = 0.0
        else:
            control.brake = 0.0
            control.throttle = throttle_brake

        #print("V: %f, Vmax: %f, Acc_idm: %f, Throttle: %f, Brake: %f" % (ego_spd, self._idm_v_ref, idm_acc_final, control.throttle, control.brake))

        return control
