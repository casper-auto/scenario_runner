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

import bisect

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import get_speed

# from srunner.custom_agents.controller_custom import SteeringController
# from srunner.custom_agents.frenet_utils import *

import numpy as np
import time

import matplotlib.pyplot as plt
from copy import deepcopy

from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

def get_speed_ms(vehicle):
    return get_speed(vehicle) / 3.6

def get_accel_ms(vehicle):
    acc = vehicle.get_acceleration()
    return math.sqrt(acc.x ** 2 + acc.y ** 2 + acc.z ** 2)

def clip_fnc(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

def points_distance(target_location, current_location):
    dx = target_location.x - current_location.x
    dy = target_location.y - current_location.y
    return math.sqrt(dx * dx + dy * dy)

def local_to_global(center, theta, p):
    s = math.sin(theta)
    c = math.cos(theta)
    out = Point(p.x * c - p.y * s + center.x, p.x * s + p.y * c + center.y)
    return out

def global_to_local(ref_orig, orientation, p):
    delta = Point(p.x - ref_orig.x, p.y - ref_orig.y)
    s = math.sin(-orientation)
    c = math.cos(-orientation)
    out = Point(delta.x * c - delta.y * s, delta.x * s + delta.y * c)
    return out

def closest_2points_index(path, x, y):
    vects = np.array([[p.x - x, p.y - y] for p in path])
    dists = np.linalg.norm(vects, axis=1)
    closest_index = np.argmin(dists[1:-1])
    if dists[closest_index+1]>dists[closest_index-1]:
        p1 = Point(path[closest_index-1].x, path[closest_index-1].y)
        p2 = Point(path[closest_index].x, path[closest_index].y)
        return p1, p2, closest_index-1
    else:
        p1 = Point(path[closest_index].x, path[closest_index].y)
        p2 = Point(path[closest_index+1].x, path[closest_index+1].y)
        return p1, p2, closest_index

# Transform from Cartesian x,y coordinates to Frenet s,d coordinates
def get_frenet(x, y, path, s_map):
    if path == None:
        print("Empty map. Cannot return Frenet coordinates")
        return 0.0, 0.0, False
    p1, p2, prev_idx = closest_2points_index(path, x, y)
    theta = math.atan2(p2.y - p1.y, p2.x - p1.x)
    # Get the point in the local coordinate with center p1
    local_p = global_to_local(p1, theta, Point(x,y))
    # Get the coordinates in the Frenet frame
    p_s = s_map[prev_idx] + local_p.x
    p_d = local_p.y
    return p_s, p_d, True

# Transform from Frenet s,d coordinates to Cartesian x,y
def get_xy(s, d, path, s_map):
    if path == None or s_map == None:
        print("Empty path. Cannot compute Cartesian coordinates")
        return 0.0, 0.0, False
    # If the value is out of the actual path send a warning
    if s < 0.0:
        prev_point = 0
    elif s > s_map[-1]:
        prev_point = len(s_map) -2
    else:
        # Find the previous point
        idx = bisect.bisect_left(s_map, s)
        prev_point = idx - 1    
    p1 = path[prev_point]
    p2 = path[prev_point + 1]
    # Transform from local to global
    theta = math.atan2(p2.y - p1.y, p2.x - p1.x)
    p_xy = local_to_global(p1, theta, Point(s - s_map[prev_point], d))
    return p_xy.x, p_xy.y, True

class SteeringController():
    """
    SteeringController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
                
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

        physics_control = self._vehicle.get_physics_control()
        self._max_steering = (physics_control.wheels[0].max_steer_angle + physics_control.wheels[1].max_steer_angle) / 2

    def run_step(self, target_loc):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        vehicle_tf = self._vehicle.get_transform()

        steer_val = self._pid_control(target_loc, vehicle_tf)
        steer_val_norm = steer_val / self._max_steering

        return steer_val_norm
        

    def _pid_control(self, target_loc, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([target_loc.x - v_begin.x, target_loc.y - v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)

class FrontAgent():
    def __init__(self):
        self._s = 1000
        self._v = 300.0

class build_drive_agent(Agent):
    def __init__(self, vehicle, start_wp, reference_speed=2.0, name="NoName", demo_id=-1, ego_agent=None):
        super(build_drive_agent, self).__init__(vehicle)
        self.start_wp = start_wp
        self.ego_agent = ego_agent

        self.wp_sampling_distance = 2.0
        self.control_look_ahead = 1.0

        self._dt = 0.05
        # Default PID
        self._e_buffer_def = deque(maxlen=20)
        self._prev_error = 0.0
        self._prev_throttle = 0.0

        steer_pid = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 0.05}
        self.steering_control = SteeringController(self._vehicle, **steer_pid)

        self._idm_s_0 = 1.0
        self._idm_t_headway = 1.0
        self._idm_acceleration_max = 3.5
        self._idm_deceleration = 2.5
        self._idm_acceleration_exp = 4.5
        self._idm_v_ref = reference_speed

        self.grp = None
        self.set_globalrouteplanner()
        print(self._vehicle)

    def empty_control(self):
        # Init control
        control_empty = carla.VehicleControl()
        control_empty.throttle = 0.0
        control_empty.brake = 0.0
        control_empty.steer = 0.0
        control_empty.hand_brake = False
        control_empty.manual_gear_shift = False
        return control_empty

    def set_globalrouteplanner(self):
        # Setting up global router
        if self.grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self.wp_sampling_distance)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self.grp = grp

    def route_to_list(self, nav_path):
        path_list, s_map = [], []
        distance_acum = 0.0
        dist_prev = 0.0
        for cur_waypoint, _ in nav_path:
            x = cur_waypoint.transform.location.x
            y = cur_waypoint.transform.location.y
            if len(path_list)!=0:
                dist_prev = distance(path_list[-1].x, path_list[-1].y, x, y)
                if dist_prev>0.1:
                    distance_acum += dist_prev
            if len(path_list)==0 or dist_prev>0.1:
                path_list.append(Point(x, y))
                s_map.append(distance_acum)

        # print(path_list)
        return path_list, s_map

    def trace_route(self, start_waypoint, end_waypoint):
        self.set_globalrouteplanner()
        self.target_wp = end_waypoint
        # Obtain route plan
        route = self.grp.trace_route(start_waypoint.transform.location,
                                     end_waypoint.transform.location)
        route.append([end_waypoint, None])
        self.path_list_agts, self.s_map_agts = self.route_to_list(route)
        return route

    def set_destination(self, location):
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))
        route_trace = self.trace_route(self.start_wp, end_waypoint)
        assert route_trace

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
            throttle = np.clip(self._prev_throttle + p_term + i_term + d_term, 0.0, 1.0)
            self._prev_throttle = throttle
            self._prev_error = v_error
        else:
            brake = np.clip(round((-acc_idm / self._idm_deceleration) * 2) / 2, 0.0, 1.0)
            self._e_buffer_def.clear
            self._prev_throttle = 0.0
            self._prev_error = 0.0      
        return throttle, brake

    def get_front_agent(self, agent_list, ego_s, margin=0):
        min_distance = 1e10
        front_agent = None
        found_front_agent = False
        for target_agent in agent_list:
            # do not account for the ego agent
            if target_agent.id == self._vehicle.id:
                continue
            # do not account for the AD agent
            # if target_agent.id == self.ego_agent.id:
            #     continue
            # Discard cars that are off in the z direction
            loc = target_agent.get_location()
            if loc.z < -10 or loc.z > 10:
                continue
            # if the object is not in our road it's not an obstacle
            target_frenet_s, target_frenet_d, valid_conv_ego = get_frenet(loc.x, loc.y, self.path_list_agts, self.s_map_agts)
            # Select the front car
            dist_car = target_frenet_s - ego_s
            half_car_width = target_agent.bounding_box.extent.y
            if dist_car > -self._vehicle.bounding_box.extent.x and abs(target_frenet_d) < (self.start_wp.lane_width / 2.0 + half_car_width + margin):
                if dist_car < min_distance:
                    min_distance = dist_car
                    front_agent = target_agent
                    found_front_agent = True
        return found_front_agent, front_agent, min_distance

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

    def run_step(self, debug=False):
        control_var = self.empty_control()

        if not self.path_list_agts or not self.s_map_agts:
            print("Frenet path has not been set yet. Return empty control")
            return control_var
        # Don't do anything for the cars not on road
        vehicle_location = self._vehicle.get_location()
        if vehicle_location.z < -10 or vehicle_location.z > 10:
            return control_var

        vehicle_speed = get_speed_ms(self._vehicle)
        look_ahead_distance = max(self.control_look_ahead * vehicle_speed, self.wp_sampling_distance)

        ego_frenet_s, _, valid_conv_ego = get_frenet(vehicle_location.x, vehicle_location.y, self.path_list_agts, self.s_map_agts)
        self.current_s_pos = ego_frenet_s
        if not valid_conv_ego:
            print("Conversion to frenet issue")
            return control_var

        x_target, y_target, valid_conv_ego = get_xy(ego_frenet_s + look_ahead_distance, 0.0, self.path_list_agts, self.s_map_agts)
        if not valid_conv_ego:
            print("Conversion to frenet issue")
            return control_var
        target_loc = carla.Location(x_target, y_target, 0.0)

        control_var.steer = self.steering_control.run_step(target_loc)  

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        front_vehicle_detected, vehicle_front, dist_front = self.get_front_agent(vehicle_list, ego_frenet_s)

        #a virtual car in far distance
        idm_front_vehicle = FrontAgent()

        if front_vehicle_detected:
            idm_front_vehicle._s = dist_front
            idm_front_vehicle._v = get_speed_ms(vehicle_front)

        idm_acc_front = self.calculate_idm_acc(idm_front_vehicle)
        #print("Front vehicle - s: %f, v: %f, idm_acc: %f" % (idm_front_vehicle._s, idm_front_vehicle._v, idm_acc_front))

        idm_acc_final = idm_acc_front

        self._state = AgentState.NAVIGATING
        
        control_var.throttle, control_var.brake = self.throttle_brake_pid_control(vehicle_speed, idm_acc_final)

        # if self.ego_agent.get_location().x<90:
        #     control_var.throttle = 1.0
        return control_var