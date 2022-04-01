#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic scenario behaviors required to realize
complex, realistic scenarios such as "follow a leading vehicle", "lane change",
etc.

The atomic behaviors are implemented with py_trees.
"""

from __future__ import print_function

import copy
import math
import operator
import os
import random
import time
import subprocess

import numpy as np
import py_trees
from py_trees.blackboard import Blackboard
import networkx

import carla
from agents.navigation.basic_agent import BasicAgent, LocalPlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.actorcontrols.actor_control import ActorControl
from srunner.scenariomanager.timer import GameTime
from srunner.tools.scenario_helper import detect_lane_obstacle
from srunner.tools.scenario_helper import generate_target_waypoint_list_multilane

from srunner.custom_agents.ss_drive_agent import build_drive_agent

import srunner.tools

def calculate_distance(location, other_location, global_planner=None):
    if global_planner:
        distance = 0
        # Get the route
        route = global_planner.trace_route(location, other_location)
        # Get the distance of the route
        for i in range(1, len(route)):
            curr_loc = route[i][0].transform.location
            prev_loc = route[i - 1][0].transform.location
            distance += curr_loc.distance(prev_loc)
        return distance
    return location.distance(other_location)

def get_actor_control(actor):
    control = actor.get_control()
    actor_type = actor.type_id.split('.')[0]
    if not isinstance(actor, carla.Walker):
        control.steering = 0
    else:
        control.speed = 0
    return control, actor_type


class AtomicBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name, actor=None):
        super(AtomicBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.name = name
        self._actor = actor

    def setup(self, unused_timeout=15):
        self.logger.debug("%s.setup()" % (self.__class__.__name__))
        return True

    def initialise(self):
        if self._actor is not None:
            try:
                check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
                terminate_wf = copy.copy(check_attr(py_trees.blackboard.Blackboard()))
                py_trees.blackboard.Blackboard().set(
                    "terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            except AttributeError:
                # It is ok to continue, if the Blackboard variable does not exist
                pass
        self.logger.debug("%s.initialise()" % (self.__class__.__name__))

    def terminate(self, new_status):
        self.logger.debug("%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))


class ss_AgentBehavior(AtomicBehavior):
    def __init__(self, actor, target_location=None, start_waypoint=None, reference_speed=None,
                 name="ss_AgentBehavior", demo_id=-1, ego_agent=None):
        super(ss_AgentBehavior, self).__init__(name)
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
        self.agent = build_drive_agent(actor,
                                       start_waypoint,
                                       reference_speed=reference_speed,
                                       name=name,
                                       demo_id=demo_id,
                                       ego_agent=ego_agent)
        self.control = carla.VehicleControl()
        if target_location:
            self.agent.set_destination((target_location.x, target_location.y, target_location.z))
        self.distance = 0
        self.location = None
        self.actor = actor
        self.ego_agent = ego_agent
        self.name = name
        self.demo_id = demo_id
        self.start_move = 135
        self.key_time = None

    def initialise(self):
        self.control = carla.VehicleControl()
        self.control.brake = 1.0
        self.actor.apply_control(self.control)
        self.distance = 0
        self.location = CarlaDataProvider.get_location(self.actor)
        super(ss_AgentBehavior, self).initialise()

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        self.control = self.agent.run_step()

        new_location = CarlaDataProvider.get_location(self.actor)
        self.distance += calculate_distance(self.location, new_location)
        self.location = new_location

        if self.ego_agent.get_location().x>self.start_move:
            self.control = carla.VehicleControl()
            self.control.brake = 1.0
        else:
            if self.demo_id=="f":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>115:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

            elif self.demo_id=="m":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if (not self.ego_agent.get_location().x>100) and self.key_time==None:
                        self.key_time = GameTime.get_time()
                        print(self.key_time)
                    if self.key_time==None:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif GameTime.get_time()-self.key_time<1 and self.ego_agent.get_location().x>99:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<130:
                        self.control = carla.VehicleControl()
                        self.control.throttle = 1.0
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>115:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

            elif self.demo_id=="p3":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>135:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh1":
                    if self.ego_agent.get_location().x>135:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

            elif self.demo_id=="p4":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>135:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh1":
                    if self.ego_agent.get_location().x>135:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

            elif self.demo_id=="p5":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>115:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<160 and self.actor.get_location().y>150:
                        self.control.throttle = 0.3*self.control.throttle*(160-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-150)
                    elif self.actor.get_location().y<150 and self.actor.get_location().y>130:
                        self.control.throttle = 0.3*self.control.throttle
                    elif self.actor.get_location().y<130 and self.actor.get_location().y>120:
                        self.control.throttle = self.control.throttle*(130-self.actor.get_location().y) + 0.3*self.control.throttle*(self.actor.get_location().y-120)
        
            elif self.demo_id=="p6":
                self.control = carla.VehicleControl()
                self.control.brake = 1.0

            elif self.demo_id=="p7":
                self.control = carla.VehicleControl()
                self.control.brake = 1.0

            elif self.demo_id=="p8":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if (not self.ego_agent.get_location().x>100) and self.key_time==None:
                        self.key_time = GameTime.get_time()
                        print(self.key_time)
                    if self.key_time==None:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif GameTime.get_time()-self.key_time<1 and self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<126.5:
                        self.control = carla.VehicleControl()
                        self.control.throttle = 1.0
                    elif self.actor.get_location().x<96:
                        self.control = carla.VehicleControl()
                        self.control.throttle = 0.8
                        self.control.steer = -0.4
                elif self.name=="Lane0Veh2":
                    if self.ego_agent.get_location().x>90:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>115:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

            elif self.demo_id=="p9":
                if self.name=="Lane0Veh0":
                    if self.ego_agent.get_location().x>100:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<148 and self.actor.get_location().y>128:
                        self.control.throttle = 0.6*(148-self.actor.get_location().y) + self.control.throttle*(self.actor.get_location().y-128)
                elif self.name=="Lane0Veh1":
                    if (not self.ego_agent.get_location().x>100) and self.key_time==None:
                        self.key_time = GameTime.get_time()
                        print(self.key_time)
                    if self.key_time==None:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif GameTime.get_time()-self.key_time<1 and self.ego_agent.get_location().x>99:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<130:
                        self.control = carla.VehicleControl()
                        self.control.throttle = 1.0
                elif self.name=="Lane0Veh2":
                    if (not self.ego_agent.get_location().x>100) and self.key_time==None:
                        self.key_time = GameTime.get_time()
                        print(self.key_time)
                    if self.key_time==None:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif GameTime.get_time()-self.key_time<1 and self.ego_agent.get_location().x>99:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0
                    elif self.actor.get_location().y<124:
                        self.control = carla.VehicleControl()
                        self.control.throttle = 1.0
                elif self.name=="Lane1Veh0":
                    if self.ego_agent.get_location().x>115:
                        self.control = carla.VehicleControl()
                        self.control.brake = 1.0

        self.actor.apply_control(self.control)
        return new_status


class ss_BasicPedestrianBehavior(AtomicBehavior):
    def __init__(self, actor, target_speed=None, plan=None, blackboard_queue_name=None,
                 avoid_collision=False, demo_id=0, name="ss_BasicPedestrianBehavior", ego_agent=None):
        """
        Set up actor and local planner
        """
        super(ss_BasicPedestrianBehavior, self).__init__(name, actor)
        self._actor_dict = {}
        self._actor_dict[actor] = None
        self._target_speed = target_speed# + random.choice([0, 0.5, 1, 2])
        self._local_planner_list = []
        self._plan = plan
        self._blackboard_queue_name = blackboard_queue_name
        if blackboard_queue_name is not None:
            self._queue = Blackboard().get(blackboard_queue_name)
        self._args_lateral_dict = {'K_P': 1.0, 'K_D': 0.01, 'K_I': 0.0, 'dt': 0.05}
        self._avoid_collision = avoid_collision
        self._unique_id = 0
        # Pause at each turning
        self._turing_flag = False
        self._prev_direction = None
        self._pause_time = random.choice([0,1,2])
        self._base_time_flag = True
        self._base_time = 0

        self.ego_agent = ego_agent
        self.start_move = 115

    def initialise(self):
        super(ss_BasicPedestrianBehavior, self).initialise()
        self._unique_id = int(round(time.time() * 1e9))

        try:
            # check whether WF for this actor is already running and add new WF to running_WF list
            check_attr = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
            running = check_attr(py_trees.blackboard.Blackboard())
            active_wf = copy.copy(running)
            active_wf.append(self._unique_id)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
        except AttributeError:
            # no WF is active for this actor
            py_trees.blackboard.Blackboard().set("terminate_WF_actor_{}".format(self._actor.id), [], overwrite=True)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), [self._unique_id], overwrite=True)

        for actor in self._actor_dict:
            self._apply_local_planner(actor)
        return True

    def _apply_local_planner(self, actor):
        if self._target_speed is None:
            self._target_speed = CarlaDataProvider.get_velocity(actor)
        else:
            self._target_speed = self._target_speed

        if isinstance(actor, carla.Walker):
            self._local_planner_list.append("Walker")
            if self._plan is not None:
                if isinstance(self._plan[0], carla.Location):
                    self._actor_dict[actor] = self._plan
                else:
                    self._actor_dict[actor] = [element[0].transform.location for element in self._plan]
                actor_location = CarlaDataProvider.get_location(actor)
                self._prev_direction = self._actor_dict[actor][0] - actor_location

    def update(self):
        """
        Compute next control step for the given waypoint plan, obtain and apply control to actor
        """
        new_status = py_trees.common.Status.RUNNING

        check_term = operator.attrgetter("terminate_WF_actor_{}".format(self._actor.id))
        terminate_wf = check_term(py_trees.blackboard.Blackboard())

        check_run = operator.attrgetter("running_WF_actor_{}".format(self._actor.id))
        active_wf = check_run(py_trees.blackboard.Blackboard())

        # Termination of WF if the WFs unique_id is listed in terminate_wf
        # only one WF should be active, therefore all previous WF have to be terminated
        if self._unique_id in terminate_wf:
            terminate_wf.remove(self._unique_id)
            if self._unique_id in active_wf:
                active_wf.remove(self._unique_id)

            py_trees.blackboard.Blackboard().set(
                "terminate_WF_actor_{}".format(self._actor.id), terminate_wf, overwrite=True)
            py_trees.blackboard.Blackboard().set(
                "running_WF_actor_{}".format(self._actor.id), active_wf, overwrite=True)
            # new_status = py_trees.common.Status.SUCCESS
            return new_status

        if self._blackboard_queue_name is not None:
            while not self._queue.empty():
                actor = self._queue.get()
                if actor is not None and actor not in self._actor_dict:
                    self._apply_local_planner(actor)

        success = True
        for actor, local_planner in zip(self._actor_dict, self._local_planner_list):
            if actor is not None and actor.is_alive and local_planner is not None:
                # If the actor is a pedestrian, we have to use the WalkerAIController
                # The walker is sent to the next waypoint in its plan
                if isinstance(actor, carla.Walker):
                    actor_location = CarlaDataProvider.get_location(actor)
                    if self._actor_dict[actor]:
                        success = False
                        location = self._actor_dict[actor][0]
                        direction = location - actor_location
                        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
                        if direction_norm > 1.0:
                            control = actor.get_control()

                            if detect_lane_obstacle(actor, margin=1.0):
                                control.speed = 0
                            else:
                                control.speed = self._target_speed

                            # pause at each turning point (if turing angle is big)
                            if self._turing_flag and GameTime.get_time() - self._base_time < self._pause_time:
                                control.speed = 0
                            else:
                                self._turing_flag = False

                            control.direction = direction / direction_norm
                            self._prev_direction = control.direction

                            if self.ego_agent.get_location().x>self.start_move:
                                control.speed = 0

                            actor.apply_control(control)
                        else:
                            self._actor_dict[actor] = self._actor_dict[actor][1:]
                            # turing flag enabled
                            if len(self._actor_dict[actor]) > 0:
                                actor_location = CarlaDataProvider.get_location(actor)
                                direction = self._actor_dict[actor][0] - actor_location
                                # compare direction with prev_direction
                                inner_dot = self._prev_direction.x*direction.x + self._prev_direction.y*direction.y
                                prev_direction_norm = math.sqrt(self._prev_direction.x**2 + self._prev_direction.y**2)
                                direction_norm = math.sqrt(direction.x**2 + direction.y**2)
                                cos_angle = inner_dot / (prev_direction_norm * direction_norm)
                                if cos_angle < 0.5: # < 0.5 means bigger than 60 degrees
                                    self._turing_flag = True
                                    self._base_time = GameTime.get_time()

        if success:
            control = actor.get_control()
            control.speed = 0
            actor.apply_control(control)

        return new_status