#!/usr/bin/env python

# Customized scenarios: Intention aware lane change
# Author: Peng Xu
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Intention aware lane change scenario:

The scenario realizes a relative lane change behavior, under circumstance
that vehicles in the target lane don't have enough gap to merge in initially.
The ego vehicle is expected to move closer to remind the neighboring vehicle
to open a bigger gap until it is safe to complete the lane change.

The scenario ends either via a timeout, or if the ego vehicle stopped close
enough to the leading vehicle
"""

import random
import math

try:
    import queue
except ImportError:
    import Queue as queue

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorTransformSetterWithVelocity,
                                                                      ActorDestroy,
                                                                      AlwaysSuccessTrigger,
                                                                      BasicPedestrianBehavior,
                                                                      KeepVelocity,
                                                                      HandBrakeVehicle,
                                                                      IDMInteractiveLaneChangeAgentBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToLocationAlongRoute,
                                                                               InTriggerDistanceToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut, GameTime
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           generate_target_waypoint_in_route,
                                           get_waypoint_in_distance,
                                           get_waypoint_in_distance_backwards)


class CasperInteractiveLaneChange(BasicScenario):

    """
    This is a single ego vehicle scenario
    """
    category = "CasperInteractiveLaneChange"

    timeout = 20            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=True, debug_mode=False,
                 terminate_on_failure=True, criteria_enable=True, timeout=80,
                 dense_traffic=True, cooperative_drivers=False, report_enable=False, num_of_players=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._dense_traffic = dense_traffic
        self._cooperative_drivers = cooperative_drivers

        self._emergence_vehicle_location = 50
        self._illegal_bicycle_location = self._emergence_vehicle_location + 10

        self._other_vehicle_i_location = 0

        print("dense_traffic: ", dense_traffic)
        print("cooperative_traffic: ", cooperative_drivers)

        if (self._dense_traffic):
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(5.5, 10.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(5.5, 10.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(5.5, 10.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(5.5, 10.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(5.5, 10.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(5.5, 10.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(5.5, 10.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(5.5, 10.0)
        else:
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(8.0, 12.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(8.0, 12.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(8.0, 12.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(8.0, 12.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(8.0, 12.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(8.0, 12.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(8.0, 12.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(8.0, 12.0)


        self._emergence_vehicle_speed = 0
        self._illegal_bicycle_speed = 0
        self._other_vehicle_speed = 5

        ego_start_location = carla.Location(config.trigger_points[0].location.x,
                                            config.trigger_points[0].location.y,
                                            config.trigger_points[0].location.z)
        self._ego_start_waypoint = self._map.get_waypoint(ego_start_location)

        waypoint_left_lane = self._ego_start_waypoint.get_left_lane()
        self._reference_waypoint, _ = get_waypoint_in_distance_backwards(waypoint_left_lane, 30)

        self._other_vehicle_max_brake = 1.0

        self._emergence_vehicle_transform = None
        self._illegal_bicycle_transform = None

        self._other_vehicle_a_transform = None
        self._other_vehicle_b_transform = None
        self._other_vehicle_c_transform = None
        self._other_vehicle_d_transform = None
        self._other_vehicle_e_transform = None
        self._other_vehicle_f_transform = None
        self._other_vehicle_g_transform = None
        self._other_vehicle_h_transform = None
        self._other_vehicle_i_transform = None

        self._other_vehicle_a_wp = None
        self._other_vehicle_b_wp = None
        self._other_vehicle_c_wp = None
        self._other_vehicle_d_wp = None
        self._other_vehicle_e_wp = None
        self._other_vehicle_f_wp = None
        self._other_vehicle_g_wp = None
        self._other_vehicle_h_wp = None
        self._other_vehicle_i_wp = None

        self._on_tick_ref = 0

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(CasperInteractiveLaneChange, self).__init__("CasperInteractiveLaneChange",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          terminate_on_failure=terminate_on_failure,
                                                          criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            distance = random.randint(20, 80)
            new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            waypoint.transform.location.z += 39
            self.other_actors[0].set_transform(waypoint.transform)

        self._report_enable = report_enable
        #Evaluation report
        if self._report_enable:
            self._start_time = 0.0
            self._elapsed_time = 0.0
            self._on_tick_ref = self._world.on_tick(self._on_tick_callback)
            self._it_counter = 0
            self._throttle_counter = 0
            self._brake_counter = 0
            self._throttle_jerk_counter = 0
            self._brake_jerk_counter = 0
            self._acc_min = 2.0
            self._acc_max = -2.0
            #self._acc_ave = 0.0
            self._throttle_ave = 0.0
            self._brake_ave = 0.0
            self._jerk_min = 2.0
            self._jerk_max = -2.0
            #self._jerk_ave = 0.0
            self._throttle_jerk_ave = 0.0
            self._brake_jerk_ave = 0.0
            #self._angular_acc_min = 0.0
            self._angular_acc_max = 0.0
            self._angular_acc_ave = 0.0
            #self._angular_jerk_min = 0.0
            self._angular_jerk_max = 0.0
            self._angular_jerk_ave = 0.0
            self._time_previous = None
            self._velocity_previous = 0.0
            self._acc_previous = 0.0
            self._velocity_filter_size = 20
            self._acc_filter_size = 20
            self._jerk_filter_size = 20
            self._angular_velocity_previous = 0.0
            self._angular_acc_previous = 0.0
            self._angular_velocity_filter_size = 20
            self._angular_acc_filter_size = 20
            self._angular_jerk_filter_size = 20

            self._velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._angular_velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._angular_acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._angular_jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._velocity_sum = 0.0
            self._acc_sum = 0.0
            self._jerk_sum = 0.0
            self._angular_velocity_sum = 0.0
            self._angular_acc_sum = 0.0
            self._angular_jerk_sum = 0.0

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        emergence_vehicle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._emergence_vehicle_location)
        illegal_bicycle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._illegal_bicycle_location)

        other_vehicle_a_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_a_location)
        other_vehicle_b_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_b_location)
        other_vehicle_c_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_c_location)
        other_vehicle_d_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_d_location)
        other_vehicle_e_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_e_location)
        other_vehicle_f_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_f_location)
        other_vehicle_g_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_g_location)
        other_vehicle_h_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_h_location)
        other_vehicle_i_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_i_location)

        self._other_vehicle_a_wp = other_vehicle_a_waypoint
        self._other_vehicle_b_wp = other_vehicle_b_waypoint
        self._other_vehicle_c_wp = other_vehicle_c_waypoint
        self._other_vehicle_d_wp = other_vehicle_d_waypoint
        self._other_vehicle_e_wp = other_vehicle_e_waypoint
        self._other_vehicle_f_wp = other_vehicle_f_waypoint
        self._other_vehicle_g_wp = other_vehicle_g_waypoint
        self._other_vehicle_h_wp = other_vehicle_h_waypoint
        self._other_vehicle_i_wp = other_vehicle_i_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._ego_start_waypoint.transform.location.x,
                           self._ego_start_waypoint.transform.location.y,
                           self._ego_start_waypoint.transform.location.z + 1),
            self._ego_start_waypoint.transform.rotation)

        # other vehicles in the source lane
        emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z - 500),
            emergence_vehicle_waypoint.transform.rotation)
        self._emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z + 1),
            emergence_vehicle_waypoint.transform.rotation)

        yaw_1 = illegal_bicycle_waypoint.transform.rotation.yaw + 90
        illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z - 500),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))
        self._illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z + 1),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))

        # other vehicles in the target lane
        other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z - 500),
            other_vehicle_a_waypoint.transform.rotation)
        self._other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z + 1),
            other_vehicle_a_waypoint.transform.rotation)

        other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z - 500),
            other_vehicle_b_waypoint.transform.rotation)
        self._other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z + 1),
            other_vehicle_b_waypoint.transform.rotation)

        other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z - 500),
            other_vehicle_c_waypoint.transform.rotation)
        self._other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z + 1),
            other_vehicle_c_waypoint.transform.rotation)

        other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z - 500),
            other_vehicle_d_waypoint.transform.rotation)
        self._other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z + 1),
            other_vehicle_d_waypoint.transform.rotation)

        other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z - 500),
            other_vehicle_e_waypoint.transform.rotation)
        self._other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z + 1),
            other_vehicle_e_waypoint.transform.rotation)

        other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z - 500),
            other_vehicle_f_waypoint.transform.rotation)
        self._other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z + 1),
            other_vehicle_f_waypoint.transform.rotation)

        other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z - 500),
            other_vehicle_g_waypoint.transform.rotation)
        self._other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z + 1),
            other_vehicle_g_waypoint.transform.rotation)

        other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z - 500),
            other_vehicle_h_waypoint.transform.rotation)
        self._other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z + 1),
            other_vehicle_h_waypoint.transform.rotation)

        other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z - 500),
            other_vehicle_i_waypoint.transform.rotation)
        self._other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z + 1),
            other_vehicle_i_waypoint.transform.rotation)

        # keep frozon before the behavior tree starts
        self.ego_vehicles[0].set_simulate_physics(False)

        emergence_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', emergence_vehicle_transform)
        illegal_bicycle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', illegal_bicycle_transform)

        other_vehicle_a = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_a_transform, color='255,0,0')
        other_vehicle_b = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_b_transform, color='255,0,0')
        other_vehicle_c = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_c_transform, color='255,0,0')
        other_vehicle_d = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_d_transform, color='255,0,0')
        other_vehicle_e = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_e_transform, color='255,0,0')
        other_vehicle_f = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_f_transform, color='255,0,0')
        other_vehicle_g = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_g_transform, color='255,0,0')
        other_vehicle_h = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_h_transform, color='255,0,0')
        other_vehicle_i = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_i_transform, color='255,0,0')

        self.other_actors.append(emergence_vehicle)
        self.other_actors.append(illegal_bicycle)

        self.other_actors.append(other_vehicle_a)
        self.other_actors.append(other_vehicle_b)
        self.other_actors.append(other_vehicle_c)
        self.other_actors.append(other_vehicle_d)
        self.other_actors.append(other_vehicle_e)
        self.other_actors.append(other_vehicle_f)
        self.other_actors.append(other_vehicle_g)
        self.other_actors.append(other_vehicle_h)
        self.other_actors.append(other_vehicle_i)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        start_location = None
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location     # start location of the scenario

        ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()

        if start_location:
            if ego_vehicle_route:
                return InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                             ego_vehicle_route,
                                                             start_location,
                                                             5)

            return InTriggerDistanceToLocation(self.ego_vehicles[0],
                                               start_location,
                                               100.0)

        return None

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Target location for other car based on the current map
        target_location_others = carla.Location(-78.7, 152.2, 1)

        target_lane_vehicle_a = py_trees.composites.Parallel(
                    "Target Lane Vehicle A",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_a.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[2], target_location_others, self._other_vehicle_a_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_b = py_trees.composites.Parallel(
                    "Target Lane Vehicle B",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_b.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[3], target_location_others, self._other_vehicle_b_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_c = py_trees.composites.Parallel(
                    "Target Lane Vehicle C",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_c.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[4], target_location_others, self._other_vehicle_c_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_d = py_trees.composites.Parallel(
                    "Target Lane Vehicle D",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_d.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[5], target_location_others, self._other_vehicle_d_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_e = py_trees.composites.Parallel(
                    "Target Lane Vehicle E",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_e.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[6], target_location_others, self._other_vehicle_e_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_f = py_trees.composites.Parallel(
                    "Target Lane Vehicle F",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_f.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[7], target_location_others, self._other_vehicle_f_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_g = py_trees.composites.Parallel(
                    "Target Lane Vehicle G",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_g.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[8], target_location_others, self._other_vehicle_g_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_h = py_trees.composites.Parallel(
                    "Target Lane Vehicle H",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_h.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[9], target_location_others, self._other_vehicle_h_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_i = py_trees.composites.Parallel(
                    "Target Lane Vehicle I",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_i.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[10], target_location_others, self._other_vehicle_i_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))


        driving_in_target_lane = py_trees.composites.Parallel(
            "Driving in Target Lane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        driving_in_target_lane.add_child(target_lane_vehicle_a)
        driving_in_target_lane.add_child(target_lane_vehicle_b)
        driving_in_target_lane.add_child(target_lane_vehicle_c)
        driving_in_target_lane.add_child(target_lane_vehicle_d)
        driving_in_target_lane.add_child(target_lane_vehicle_e)
        driving_in_target_lane.add_child(target_lane_vehicle_f)
        driving_in_target_lane.add_child(target_lane_vehicle_g)
        driving_in_target_lane.add_child(target_lane_vehicle_h)
        driving_in_target_lane.add_child(target_lane_vehicle_i)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 51.0, "DriveDistance")
        endcondition.add_child(endcondition_merged)


        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._emergence_vehicle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._illegal_bicycle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_vehicle_a_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[3], self._other_vehicle_b_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[4], self._other_vehicle_c_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[5], self._other_vehicle_d_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[6], self._other_vehicle_e_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[7], self._other_vehicle_f_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[8], self._other_vehicle_g_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[9], self._other_vehicle_h_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[10], self._other_vehicle_i_transform))
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0)) # -random.random()*3

        drive_all = py_trees.composites.Parallel( "All cars driving", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        drive_all.add_child(endcondition_merged)
        drive_all.add_child(driving_in_target_lane)

        sequence.add_child(drive_all)

        return sequence

    def _on_tick_callback(self, toto):
        #print('***************************************************************************************')
        #print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        #print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        #print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        #print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        #print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        #print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        #if self._debug_mode:
        #print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous!=None:
            delta_time =  curr_time - self._time_previous
            #print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        #velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if (self._velocity_queue.qsize() < self._velocity_filter_size):
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        #acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if (self._acc_queue.qsize() < self._acc_filter_size):
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if (acc_current > self._acc_max):
                    self._acc_max = acc_current
                if (acc_current < self._acc_min):
                    self._acc_min = acc_current
                #self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if (acc_current >= 0.0):
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter+=1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter+=1


        #jerk
        if self._it_counter > self._velocity_filter_size + 2:
            if delta_time > 0.0:
                jerk_current = (acc_current - self._acc_previous)/delta_time
                self._jerk_queue.put(jerk_current)
                if self._jerk_queue.qsize() < self._jerk_filter_size:
                    self._jerk_sum += jerk_current
                else:
                    self._jerk_sum = self._jerk_sum + jerk_current - self._jerk_queue.get()
                jerk_current = self._jerk_sum / self._jerk_queue.qsize()

                if jerk_current > self._jerk_max:
                    self._jerk_max = jerk_current
                if jerk_current < self._jerk_min:
                    self._jerk_min = jerk_current
                #self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter +  jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter+=1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter +  jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter+=1


        #angular velocity
        angular_acc_current = 0.0
        angular_jerk_current = 0.0
        if self.ego_vehicles[0]:
            angular_velocity_current = self.ego_vehicles[0].get_angular_velocity().z
        else:
            angular_velocity_current = 0

        self._angular_velocity_queue.put(angular_velocity_current)
        if self._angular_velocity_queue.qsize() < self._angular_velocity_filter_size:
            self._angular_velocity_sum += angular_velocity_current
        else:
            self._angular_velocity_sum = self._angular_velocity_sum + angular_velocity_current - self._angular_velocity_queue.get()

        angular_velocity_current = self._angular_velocity_sum / self._angular_velocity_queue.qsize()

        if self._it_counter > self._angular_velocity_filter_size:
            #angular acc
            if delta_time > 0.0:
                angular_acc_current = (angular_velocity_current - self._angular_velocity_previous)/delta_time
                self._angular_acc_queue.put(angular_acc_current)
                if (self._angular_acc_queue.qsize() < self._angular_acc_filter_size):
                    self._angular_acc_sum += angular_acc_current
                else:
                    self._angular_acc_sum = self._angular_acc_sum + angular_acc_current - self._angular_acc_queue.get()
                angular_acc_current = self._angular_acc_sum / self._angular_acc_queue.qsize()

                if abs(angular_acc_current) > self._angular_acc_max:
                    self._angular_acc_max = abs(angular_acc_current)

                self._angular_acc_ave = (self._angular_acc_ave * self._it_counter + abs(angular_acc_current)) / (self._it_counter + 1)

        if self._it_counter > self._angular_velocity_filter_size + 2:
            #angular jerk
            if (delta_time > 0.0):
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if (self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size):
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if (abs(angular_jerk_current) > self._angular_jerk_max):
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        #store previous numbers
        self._time_previous = curr_time
        self._velocity_previous = velocity_current
        self._acc_previous = acc_current
        self._angular_velocity_previous = angular_velocity_current
        self._angular_acc_previous = angular_acc_current
        self._it_counter = self._it_counter + 1

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=self.terminate_on_failure)

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        # self.remove_all_actors()
        # if self._on_tick_ref != None:
        #     self._world.remove_on_tick(self._on_tick_ref)
        print('========================== ')

    def remove_on_tick(self):
        """
        Remove on_tick
        """
        if self._on_tick_ref != None:
            self._world.remove_on_tick(self._on_tick_ref)


class CasperInteractiveLaneChangeHumanDriver(BasicScenario):

    """
    This is a single ego vehicle scenario
    """
    category = "CasperInteractiveLaneChange"

    timeout = 20            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=True, debug_mode=False, terminate_on_failure=True, criteria_enable=True, timeout=80, dense_traffic=True, cooperative_drivers=False, num_of_players=0):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._dense_traffic = dense_traffic
        self._cooperative_drivers = cooperative_drivers

        self._emergence_vehicle_location = 50
        self._illegal_bicycle_location = self._emergence_vehicle_location + 10

        self._other_vehicle_i_location = 0

        print ("dense_traffic: ", dense_traffic)
        print ("cooperative_traffic: ", cooperative_drivers)

        if (self._dense_traffic):
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(5.5, 10.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(5.5, 10.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(5.5, 10.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(5.5, 10.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(5.5, 10.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(5.5, 10.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(5.5, 10.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(5.5, 10.0)
        else:
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(8.0, 12.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(8.0, 12.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(8.0, 12.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(8.0, 12.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(8.0, 12.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(8.0, 12.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(8.0, 12.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(8.0, 12.0)

        self._emergence_vehicle_speed = 0
        self._illegal_bicycle_speed = 0
        self._other_vehicle_speed = 5

        ego_start_location = carla.Location(config.trigger_points[0].location.x,
                                            config.trigger_points[0].location.y,
                                            config.trigger_points[0].location.z)
        self._ego_start_waypoint = self._map.get_waypoint(ego_start_location)

        waypoint_left_lane = self._ego_start_waypoint.get_left_lane()
        self._reference_waypoint, _ = get_waypoint_in_distance_backwards(waypoint_left_lane, 30)

        self._other_vehicle_max_brake = 1.0

        self._emergence_vehicle_transform = None
        self._illegal_bicycle_transform = None

        self._other_vehicle_a_transform = None
        self._other_vehicle_b_transform = None
        self._other_vehicle_c_transform = None
        self._other_vehicle_d_transform = None
        self._other_vehicle_e_transform = None
        self._other_vehicle_f_transform = None
        self._other_vehicle_g_transform = None
        self._other_vehicle_h_transform = None
        self._other_vehicle_i_transform = None

        self._other_vehicle_a_wp = None
        self._other_vehicle_b_wp = None
        self._other_vehicle_c_wp = None
        self._other_vehicle_d_wp = None
        self._other_vehicle_e_wp = None
        self._other_vehicle_f_wp = None
        self._other_vehicle_g_wp = None
        self._other_vehicle_h_wp = None
        self._other_vehicle_i_wp = None

        self._on_tick_ref = 0

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(CasperInteractiveLaneChangeHumanDriver, self).__init__("CasperInteractiveLaneChangeHumanDriver",
                                                                     ego_vehicles,
                                                                     config,
                                                                     world,
                                                                     debug_mode,
                                                                     terminate_on_failure=terminate_on_failure,
                                                                     criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            distance = random.randint(20, 80)
            new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            waypoint.transform.location.z += 39
            self.other_actors[0].set_transform(waypoint.transform)

        self._report_enable = report_enable
        #Evaluation report
        if self._report_enable:
            self._start_time = 0.0
            self._elapsed_time = 0.0
            self._on_tick_ref = self._world.on_tick(self._on_tick_callback)
            self._it_counter = 0
            self._throttle_counter = 0
            self._brake_counter = 0
            self._throttle_jerk_counter = 0
            self._brake_jerk_counter = 0
            self._acc_min = 2.0
            self._acc_max = -2.0
            #self._acc_ave = 0.0
            self._throttle_ave = 0.0
            self._brake_ave = 0.0
            self._jerk_min = 2.0
            self._jerk_max = -2.0
            #self._jerk_ave = 0.0
            self._throttle_jerk_ave = 0.0
            self._brake_jerk_ave = 0.0
            #self._angular_acc_min = 0.0
            self._angular_acc_max = 0.0
            self._angular_acc_ave = 0.0
            #self._angular_jerk_min = 0.0
            self._angular_jerk_max = 0.0
            self._angular_jerk_ave = 0.0
            self._time_previous = None
            self._velocity_previous = 0.0
            self._acc_previous = 0.0
            self._velocity_filter_size = 20
            self._acc_filter_size = 20
            self._jerk_filter_size = 20
            self._angular_velocity_previous = 0.0
            self._angular_acc_previous = 0.0
            self._angular_velocity_filter_size = 20
            self._angular_acc_filter_size = 20
            self._angular_jerk_filter_size = 20

            self._velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._angular_velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._angular_acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._angular_jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._velocity_sum = 0.0
            self._acc_sum = 0.0
            self._jerk_sum = 0.0
            self._angular_velocity_sum = 0.0
            self._angular_acc_sum = 0.0
            self._angular_jerk_sum = 0.0

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        emergence_vehicle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._emergence_vehicle_location)
        illegal_bicycle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._illegal_bicycle_location)

        other_vehicle_a_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_a_location)
        other_vehicle_b_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_b_location)
        other_vehicle_c_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_c_location)
        other_vehicle_d_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_d_location)
        other_vehicle_e_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_e_location)
        other_vehicle_f_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_f_location)
        other_vehicle_g_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_g_location)
        other_vehicle_h_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_h_location)
        other_vehicle_i_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_i_location)

        self._other_vehicle_a_wp = other_vehicle_a_waypoint
        self._other_vehicle_b_wp = other_vehicle_b_waypoint
        self._other_vehicle_c_wp = other_vehicle_c_waypoint
        self._other_vehicle_d_wp = other_vehicle_d_waypoint
        self._other_vehicle_e_wp = other_vehicle_e_waypoint
        self._other_vehicle_f_wp = other_vehicle_f_waypoint
        self._other_vehicle_g_wp = other_vehicle_g_waypoint
        self._other_vehicle_h_wp = other_vehicle_h_waypoint
        self._other_vehicle_i_wp = other_vehicle_i_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._ego_start_waypoint.transform.location.x,
                           self._ego_start_waypoint.transform.location.y,
                           self._ego_start_waypoint.transform.location.z + 1),
            self._ego_start_waypoint.transform.rotation)

        # other vehicles in the source lane
        emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z - 500),
            emergence_vehicle_waypoint.transform.rotation)
        self._emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z + 1),
            emergence_vehicle_waypoint.transform.rotation)

        yaw_1 = illegal_bicycle_waypoint.transform.rotation.yaw + 90
        illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z - 500),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))
        self._illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z + 1),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))

        # other vehicles in the target lane
        other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z - 500),
            other_vehicle_a_waypoint.transform.rotation)
        self._other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z + 1),
            other_vehicle_a_waypoint.transform.rotation)

        other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z - 500),
            other_vehicle_b_waypoint.transform.rotation)
        self._other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z + 1),
            other_vehicle_b_waypoint.transform.rotation)

        other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z - 500),
            other_vehicle_c_waypoint.transform.rotation)
        self._other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z + 1),
            other_vehicle_c_waypoint.transform.rotation)

        other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z - 500),
            other_vehicle_d_waypoint.transform.rotation)
        self._other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z + 1),
            other_vehicle_d_waypoint.transform.rotation)

        other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z - 500),
            other_vehicle_e_waypoint.transform.rotation)
        self._other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z + 1),
            other_vehicle_e_waypoint.transform.rotation)

        other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z - 500),
            other_vehicle_f_waypoint.transform.rotation)
        self._other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z + 1),
            other_vehicle_f_waypoint.transform.rotation)

        other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z - 500),
            other_vehicle_g_waypoint.transform.rotation)
        self._other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z + 1),
            other_vehicle_g_waypoint.transform.rotation)

        other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z - 500),
            other_vehicle_h_waypoint.transform.rotation)
        self._other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z + 1),
            other_vehicle_h_waypoint.transform.rotation)

        other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z - 500),
            other_vehicle_i_waypoint.transform.rotation)
        self._other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z + 1),
            other_vehicle_i_waypoint.transform.rotation)

        emergence_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', emergence_vehicle_transform)
        illegal_bicycle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', illegal_bicycle_transform)

        other_vehicle_a = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_a_transform, color='255,0,0')
        other_vehicle_b = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_b_transform, color='255,0,0')
        other_vehicle_c = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_c_transform, color='255,0,0')
        other_vehicle_d = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_d_transform, color='255,0,0')
        other_vehicle_e = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_e_transform, rolename='keyboard', color='255,255,0')
        other_vehicle_f = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_f_transform, color='255,0,0')
        other_vehicle_g = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_g_transform, color='255,0,0')
        other_vehicle_h = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_h_transform, color='255,0,0')
        other_vehicle_i = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_i_transform, color='255,0,0')

        self.other_actors.append(emergence_vehicle)
        self.other_actors.append(illegal_bicycle)

        self.other_actors.append(other_vehicle_a)
        self.other_actors.append(other_vehicle_b)
        self.other_actors.append(other_vehicle_c)
        self.other_actors.append(other_vehicle_d)
        self.other_actors.append(other_vehicle_e)
        self.other_actors.append(other_vehicle_f)
        self.other_actors.append(other_vehicle_g)
        self.other_actors.append(other_vehicle_h)
        self.other_actors.append(other_vehicle_i)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        start_location = None
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location     # start location of the scenario

        ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()

        if start_location:
            if ego_vehicle_route:
                return InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                             ego_vehicle_route,
                                                             start_location,
                                                             5)

            return InTriggerDistanceToLocation(self.ego_vehicles[0],
                                               start_location,
                                               100.0)

        return None

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Target location for other car based on the current map
        target_location_others = carla.Location(-78.7, 152.2, 1)

        target_lane_vehicle_a = py_trees.composites.Parallel(
                    "Target Lane Vehicle A",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_a.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[2], target_location_others, self._other_vehicle_a_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_b = py_trees.composites.Parallel(
                    "Target Lane Vehicle B",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_b.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[3], target_location_others, self._other_vehicle_b_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_c = py_trees.composites.Parallel(
                    "Target Lane Vehicle C",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_c.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[4], target_location_others, self._other_vehicle_c_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_d = py_trees.composites.Parallel(
                    "Target Lane Vehicle D",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_d.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[5], target_location_others, self._other_vehicle_d_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_e = py_trees.composites.Parallel(
                    "Target Lane Vehicle E",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_e.add_child(HumanDriverBehavior(self.other_actors[6], target_location_others))

        target_lane_vehicle_f = py_trees.composites.Parallel(
                    "Target Lane Vehicle F",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_f.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[7], target_location_others, self._other_vehicle_f_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_g = py_trees.composites.Parallel(
                    "Target Lane Vehicle G",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_g.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[8], target_location_others, self._other_vehicle_g_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_h = py_trees.composites.Parallel(
                    "Target Lane Vehicle H",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_h.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[9], target_location_others, self._other_vehicle_h_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_i = py_trees.composites.Parallel(
                    "Target Lane Vehicle I",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_i.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[10], target_location_others, self._other_vehicle_i_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))


        driving_in_target_lane = py_trees.composites.Parallel(
            "Driving in Target Lane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        driving_in_target_lane.add_child(target_lane_vehicle_a)
        driving_in_target_lane.add_child(target_lane_vehicle_b)
        driving_in_target_lane.add_child(target_lane_vehicle_c)
        driving_in_target_lane.add_child(target_lane_vehicle_d)
        driving_in_target_lane.add_child(target_lane_vehicle_e)
        driving_in_target_lane.add_child(target_lane_vehicle_f)
        driving_in_target_lane.add_child(target_lane_vehicle_g)
        driving_in_target_lane.add_child(target_lane_vehicle_h)
        driving_in_target_lane.add_child(target_lane_vehicle_i)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 51.0, "DriveDistance" )
        endcondition.add_child(endcondition_merged)


        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._emergence_vehicle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._illegal_bicycle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_vehicle_a_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[3], self._other_vehicle_b_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[4], self._other_vehicle_c_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[5], self._other_vehicle_d_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[6], self._other_vehicle_e_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[7], self._other_vehicle_f_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[8], self._other_vehicle_g_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[9], self._other_vehicle_h_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[10], self._other_vehicle_i_transform))
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, -random.random()*3))

        #sequence.add_child(driving_in_target_lane)
        #sequence.add_child(StopVehicle(self.other_actors[2], self._other_vehicle_max_brake))

        #drive_all = py_trees.composites.Parallel( "All cars driving", policy=py_trees.common.ParallelPolicy.SuccessOnSelected(children=[]))
        drive_all = py_trees.composites.Parallel( "All cars driving", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        drive_all.add_child(endcondition_merged)
        drive_all.add_child(driving_in_target_lane)

        sequence.add_child(drive_all)

        #sequence.add_child(driving_in_target_lane)
        #sequence.add_child(endcondition)

        return sequence

    def _on_tick_callback(self, toto):
        #print('***************************************************************************************')
        #print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        #print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        #print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        #print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        #print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        #print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        #if self._debug_mode:
        #print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous!=None:
            delta_time =  curr_time - self._time_previous
            #print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        #velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if (self._velocity_queue.qsize() < self._velocity_filter_size):
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        #acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if (self._acc_queue.qsize() < self._acc_filter_size):
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if (acc_current > self._acc_max):
                    self._acc_max = acc_current
                if (acc_current < self._acc_min):
                    self._acc_min = acc_current
                #self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if (acc_current >= 0.0):
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter+=1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter+=1


        #jerk
        if self._it_counter > self._velocity_filter_size + 2:
            if delta_time > 0.0:
                jerk_current = (acc_current - self._acc_previous)/delta_time
                self._jerk_queue.put(jerk_current)
                if self._jerk_queue.qsize() < self._jerk_filter_size:
                    self._jerk_sum += jerk_current
                else:
                    self._jerk_sum = self._jerk_sum + jerk_current - self._jerk_queue.get()
                jerk_current = self._jerk_sum / self._jerk_queue.qsize()

                if jerk_current > self._jerk_max:
                    self._jerk_max = jerk_current
                if jerk_current < self._jerk_min:
                    self._jerk_min = jerk_current
                #self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter +  jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter+=1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter +  jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter+=1


        #angular velocity
        angular_acc_current = 0.0
        angular_jerk_current = 0.0
        if self.ego_vehicles[0]:
            angular_velocity_current = self.ego_vehicles[0].get_angular_velocity().z
        else:
            angular_velocity_current = 0

        self._angular_velocity_queue.put(angular_velocity_current)
        if self._angular_velocity_queue.qsize() < self._angular_velocity_filter_size:
            self._angular_velocity_sum += angular_velocity_current
        else:
            self._angular_velocity_sum = self._angular_velocity_sum + angular_velocity_current - self._angular_velocity_queue.get()

        angular_velocity_current = self._angular_velocity_sum / self._angular_velocity_queue.qsize()

        if self._it_counter > self._angular_velocity_filter_size:
            #angular acc
            if delta_time > 0.0:
                angular_acc_current = (angular_velocity_current - self._angular_velocity_previous)/delta_time
                self._angular_acc_queue.put(angular_acc_current)
                if (self._angular_acc_queue.qsize() < self._angular_acc_filter_size):
                    self._angular_acc_sum += angular_acc_current
                else:
                    self._angular_acc_sum = self._angular_acc_sum + angular_acc_current - self._angular_acc_queue.get()
                angular_acc_current = self._angular_acc_sum / self._angular_acc_queue.qsize()

                if abs(angular_acc_current) > self._angular_acc_max:
                    self._angular_acc_max = abs(angular_acc_current)

                self._angular_acc_ave = (self._angular_acc_ave * self._it_counter + abs(angular_acc_current)) / (self._it_counter + 1)

        if self._it_counter > self._angular_velocity_filter_size + 2:
            #angular jerk
            if (delta_time > 0.0):
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if (self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size):
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if (abs(angular_jerk_current) > self._angular_jerk_max):
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        #store previous numbers
        self._time_previous = curr_time
        self._velocity_previous = velocity_current
        self._acc_previous = acc_current
        self._angular_velocity_previous = angular_velocity_current
        self._angular_acc_previous = angular_acc_current
        self._it_counter = self._it_counter + 1

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=self.terminate_on_failure)

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        # self.remove_all_actors()
        # if self._on_tick_ref != None:
        #     self._world.remove_on_tick(self._on_tick_ref)
        print('========================== ')

    def remove_on_tick(self):
        """
        Remove on_tick
        """
        if self._on_tick_ref != None:
            self._world.remove_on_tick(self._on_tick_ref)


class CasperInteractiveLaneChangeMultiplePlayers(BasicScenario):

    """
    This is a single ego vehicle scenario
    """
    category = "CasperInteractiveLaneChange"

    timeout = 20            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=True, debug_mode=False, terminate_on_failure=True, criteria_enable=True, timeout=80, dense_traffic=True, cooperative_drivers=False, num_of_players=0, player='steeringwheel'):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._dense_traffic = dense_traffic
        self._cooperative_drivers = cooperative_drivers
        self.num_of_players = num_of_players
        self.player = player

        self._emergence_vehicle_location = 50
        self._illegal_bicycle_location = self._emergence_vehicle_location + 10

        self._other_vehicle_i_location = 0

        print ("dense_traffic: ", dense_traffic)
        print ("cooperative_traffic: ", cooperative_drivers)

        if (self._dense_traffic):
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(5.5, 10.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(5.5, 10.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(5.5, 10.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(5.5, 10.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(5.5, 10.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(5.5, 10.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(5.5, 10.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(5.5, 10.0)
        else:
            self._other_vehicle_h_location = self._other_vehicle_i_location + random.uniform(8.0, 12.0)
            self._other_vehicle_g_location = self._other_vehicle_h_location + random.uniform(8.0, 12.0)
            self._other_vehicle_f_location = self._other_vehicle_g_location + random.uniform(8.0, 12.0)
            self._other_vehicle_e_location = self._other_vehicle_f_location + random.uniform(8.0, 12.0)
            self._other_vehicle_d_location = self._other_vehicle_e_location + random.uniform(8.0, 12.0)
            self._other_vehicle_c_location = self._other_vehicle_d_location + random.uniform(8.0, 12.0)
            self._other_vehicle_b_location = self._other_vehicle_c_location + random.uniform(8.0, 12.0)
            self._other_vehicle_a_location = self._other_vehicle_b_location + random.uniform(8.0, 12.0)


        self._emergence_vehicle_speed = 0
        self._illegal_bicycle_speed = 0
        self._other_vehicle_speed = 5

        ego_start_location = carla.Location(config.trigger_points[0].location.x,
                                            config.trigger_points[0].location.y,
                                            config.trigger_points[0].location.z)
        self._ego_start_waypoint = self._map.get_waypoint(ego_start_location)

        waypoint_left_lane = self._ego_start_waypoint.get_left_lane()
        self._reference_waypoint, _ = get_waypoint_in_distance_backwards(waypoint_left_lane, 30)

        self._other_vehicle_max_brake = 1.0

        self._emergence_vehicle_transform = None
        self._illegal_bicycle_transform = None

        self._other_vehicle_a_transform = None
        self._other_vehicle_b_transform = None
        self._other_vehicle_c_transform = None
        self._other_vehicle_d_transform = None
        self._other_vehicle_e_transform = None
        self._other_vehicle_f_transform = None
        self._other_vehicle_g_transform = None
        self._other_vehicle_h_transform = None
        self._other_vehicle_i_transform = None

        self._other_vehicle_a_wp = None
        self._other_vehicle_b_wp = None
        self._other_vehicle_c_wp = None
        self._other_vehicle_d_wp = None
        self._other_vehicle_e_wp = None
        self._other_vehicle_f_wp = None
        self._other_vehicle_g_wp = None
        self._other_vehicle_h_wp = None
        self._other_vehicle_i_wp = None

        self._on_tick_ref = 0

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(CasperInteractiveLaneChangeMultiplePlayers, self).__init__("CasperInteractiveLaneChangeMultiplePlayers",
                                                                         ego_vehicles,
                                                                         config,
                                                                         world,
                                                                         debug_mode,
                                                                         terminate_on_failure=terminate_on_failure,
                                                                         criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            distance = random.randint(20, 80)
            new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            waypoint.transform.location.z += 39
            self.other_actors[0].set_transform(waypoint.transform)

        self._report_enable = report_enable
        #Evaluation report
        if self._report_enable:
            self._start_time = 0.0
            self._elapsed_time = 0.0
            self._on_tick_ref = self._world.on_tick(self._on_tick_callback)
            self._it_counter = 0
            self._throttle_counter = 0
            self._brake_counter = 0
            self._throttle_jerk_counter = 0
            self._brake_jerk_counter = 0
            self._acc_min = 2.0
            self._acc_max = -2.0
            #self._acc_ave = 0.0
            self._throttle_ave = 0.0
            self._brake_ave = 0.0
            self._jerk_min = 2.0
            self._jerk_max = -2.0
            #self._jerk_ave = 0.0
            self._throttle_jerk_ave = 0.0
            self._brake_jerk_ave = 0.0
            #self._angular_acc_min = 0.0
            self._angular_acc_max = 0.0
            self._angular_acc_ave = 0.0
            #self._angular_jerk_min = 0.0
            self._angular_jerk_max = 0.0
            self._angular_jerk_ave = 0.0
            self._time_previous = None
            self._velocity_previous = 0.0
            self._acc_previous = 0.0
            self._velocity_filter_size = 20
            self._acc_filter_size = 20
            self._jerk_filter_size = 20
            self._angular_velocity_previous = 0.0
            self._angular_acc_previous = 0.0
            self._angular_velocity_filter_size = 20
            self._angular_acc_filter_size = 20
            self._angular_jerk_filter_size = 20

            self._velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._angular_velocity_queue = queue.Queue(maxsize=self._velocity_filter_size)
            self._angular_acc_queue = queue.Queue(maxsize=self._acc_filter_size)
            self._angular_jerk_queue = queue.Queue(maxsize=self._jerk_filter_size)
            self._velocity_sum = 0.0
            self._acc_sum = 0.0
            self._jerk_sum = 0.0
            self._angular_velocity_sum = 0.0
            self._angular_acc_sum = 0.0
            self._angular_jerk_sum = 0.0

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        emergence_vehicle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._emergence_vehicle_location)
        illegal_bicycle_waypoint, _ = get_waypoint_in_distance(self._ego_start_waypoint, self._illegal_bicycle_location)

        other_vehicle_a_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_a_location)
        other_vehicle_b_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_b_location)
        other_vehicle_c_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_c_location)
        other_vehicle_d_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_d_location)
        other_vehicle_e_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_e_location)
        other_vehicle_f_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_f_location)
        other_vehicle_g_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_g_location)
        other_vehicle_h_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_h_location)
        other_vehicle_i_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_vehicle_i_location)

        self._other_vehicle_a_wp = other_vehicle_a_waypoint
        self._other_vehicle_b_wp = other_vehicle_b_waypoint
        self._other_vehicle_c_wp = other_vehicle_c_waypoint
        self._other_vehicle_d_wp = other_vehicle_d_waypoint
        self._other_vehicle_e_wp = other_vehicle_e_waypoint
        self._other_vehicle_f_wp = other_vehicle_f_waypoint
        self._other_vehicle_g_wp = other_vehicle_g_waypoint
        self._other_vehicle_h_wp = other_vehicle_h_waypoint
        self._other_vehicle_i_wp = other_vehicle_i_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._ego_start_waypoint.transform.location.x,
                           self._ego_start_waypoint.transform.location.y,
                           self._ego_start_waypoint.transform.location.z + 1),
            self._ego_start_waypoint.transform.rotation)

        # other vehicles in the source lane
        emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z - 500),
            emergence_vehicle_waypoint.transform.rotation)
        self._emergence_vehicle_transform = carla.Transform(
            carla.Location(emergence_vehicle_waypoint.transform.location.x,
                           emergence_vehicle_waypoint.transform.location.y,
                           emergence_vehicle_waypoint.transform.location.z + 1),
            emergence_vehicle_waypoint.transform.rotation)

        yaw_1 = illegal_bicycle_waypoint.transform.rotation.yaw + 90
        illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z - 500),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))
        self._illegal_bicycle_transform = carla.Transform(
            carla.Location(illegal_bicycle_waypoint.transform.location.x,
                           illegal_bicycle_waypoint.transform.location.y,
                           illegal_bicycle_waypoint.transform.location.z + 1),
            carla.Rotation(illegal_bicycle_waypoint.transform.rotation.pitch, yaw_1,
                           illegal_bicycle_waypoint.transform.rotation.roll))

        # other vehicles in the target lane
        other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z - 500),
            other_vehicle_a_waypoint.transform.rotation)
        self._other_vehicle_a_transform = carla.Transform(
            carla.Location(other_vehicle_a_waypoint.transform.location.x,
                           other_vehicle_a_waypoint.transform.location.y,
                           other_vehicle_a_waypoint.transform.location.z + 1),
            other_vehicle_a_waypoint.transform.rotation)

        other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z - 500),
            other_vehicle_b_waypoint.transform.rotation)
        self._other_vehicle_b_transform = carla.Transform(
            carla.Location(other_vehicle_b_waypoint.transform.location.x,
                           other_vehicle_b_waypoint.transform.location.y,
                           other_vehicle_b_waypoint.transform.location.z + 1),
            other_vehicle_b_waypoint.transform.rotation)

        other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z - 500),
            other_vehicle_c_waypoint.transform.rotation)
        self._other_vehicle_c_transform = carla.Transform(
            carla.Location(other_vehicle_c_waypoint.transform.location.x,
                           other_vehicle_c_waypoint.transform.location.y,
                           other_vehicle_c_waypoint.transform.location.z + 1),
            other_vehicle_c_waypoint.transform.rotation)

        other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z - 500),
            other_vehicle_d_waypoint.transform.rotation)
        self._other_vehicle_d_transform = carla.Transform(
            carla.Location(other_vehicle_d_waypoint.transform.location.x,
                           other_vehicle_d_waypoint.transform.location.y,
                           other_vehicle_d_waypoint.transform.location.z + 1),
            other_vehicle_d_waypoint.transform.rotation)

        other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z - 500),
            other_vehicle_e_waypoint.transform.rotation)
        self._other_vehicle_e_transform = carla.Transform(
            carla.Location(other_vehicle_e_waypoint.transform.location.x,
                           other_vehicle_e_waypoint.transform.location.y,
                           other_vehicle_e_waypoint.transform.location.z + 1),
            other_vehicle_e_waypoint.transform.rotation)

        other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z - 500),
            other_vehicle_f_waypoint.transform.rotation)
        self._other_vehicle_f_transform = carla.Transform(
            carla.Location(other_vehicle_f_waypoint.transform.location.x,
                           other_vehicle_f_waypoint.transform.location.y,
                           other_vehicle_f_waypoint.transform.location.z + 1),
            other_vehicle_f_waypoint.transform.rotation)

        other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z - 500),
            other_vehicle_g_waypoint.transform.rotation)
        self._other_vehicle_g_transform = carla.Transform(
            carla.Location(other_vehicle_g_waypoint.transform.location.x,
                           other_vehicle_g_waypoint.transform.location.y,
                           other_vehicle_g_waypoint.transform.location.z + 1),
            other_vehicle_g_waypoint.transform.rotation)

        other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z - 500),
            other_vehicle_h_waypoint.transform.rotation)
        self._other_vehicle_h_transform = carla.Transform(
            carla.Location(other_vehicle_h_waypoint.transform.location.x,
                           other_vehicle_h_waypoint.transform.location.y,
                           other_vehicle_h_waypoint.transform.location.z + 1),
            other_vehicle_h_waypoint.transform.rotation)

        other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z - 500),
            other_vehicle_i_waypoint.transform.rotation)
        self._other_vehicle_i_transform = carla.Transform(
            carla.Location(other_vehicle_i_waypoint.transform.location.x,
                           other_vehicle_i_waypoint.transform.location.y,
                           other_vehicle_i_waypoint.transform.location.z + 1),
            other_vehicle_i_waypoint.transform.rotation)

        emergence_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', emergence_vehicle_transform)
        illegal_bicycle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', illegal_bicycle_transform)

        other_vehicle_a = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_a_transform, rolename=self.player+'1', color='255,0,0')
        other_vehicle_b = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_b_transform, rolename=self.player+'2', color='255,0,0')
        other_vehicle_c = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_c_transform, rolename=self.player+'3', color='255,0,0')
        other_vehicle_d = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_d_transform, rolename=self.player+'4', color='255,0,0')
        other_vehicle_e = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_e_transform, rolename=self.player+'5', color='255,0,0')
        other_vehicle_f = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_f_transform, rolename=self.player+'6', color='255,0,0')
        other_vehicle_g = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_g_transform, rolename=self.player+'7', color='255,0,0')
        other_vehicle_h = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_h_transform, rolename=self.player+'8', color='255,0,0')
        other_vehicle_i = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_i_transform, rolename=self.player+'9', color='255,0,0')

        self.other_actors.append(emergence_vehicle)
        self.other_actors.append(illegal_bicycle)

        self.other_actors.append(other_vehicle_a)
        self.other_actors.append(other_vehicle_b)
        self.other_actors.append(other_vehicle_c)
        self.other_actors.append(other_vehicle_d)
        self.other_actors.append(other_vehicle_e)
        self.other_actors.append(other_vehicle_f)
        self.other_actors.append(other_vehicle_g)
        self.other_actors.append(other_vehicle_h)
        self.other_actors.append(other_vehicle_i)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Target location for other car based on the current map
        target_location_others = carla.Location(-78.7, 152.2, 1)

        target_lane_vehicle_a = py_trees.composites.Parallel(
                    "Target Lane Vehicle A",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_a.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[2], target_location_others, self._other_vehicle_a_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_b = py_trees.composites.Parallel(
                    "Target Lane Vehicle B",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        target_lane_vehicle_b.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[3], target_location_others, self._other_vehicle_b_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_c = py_trees.composites.Parallel(
                    "Target Lane Vehicle C",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 1:
            target_lane_vehicle_c.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[4], target_location_others, self._other_vehicle_c_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_d = py_trees.composites.Parallel(
                    "Target Lane Vehicle D",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 2:
            target_lane_vehicle_d.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[5], target_location_others, self._other_vehicle_d_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_e = py_trees.composites.Parallel(
                    "Target Lane Vehicle E",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 3:
            target_lane_vehicle_e.add_child(HumanDriverBehavior(self.other_actors[6], target_location_others))

        target_lane_vehicle_f = py_trees.composites.Parallel(
                    "Target Lane Vehicle F",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 4:
            target_lane_vehicle_f.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[7], target_location_others, self._other_vehicle_f_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_g = py_trees.composites.Parallel(
                    "Target Lane Vehicle G",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 5:
            target_lane_vehicle_g.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[8], target_location_others, self._other_vehicle_g_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_h = py_trees.composites.Parallel(
                    "Target Lane Vehicle H",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 6:
            target_lane_vehicle_h.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[9], target_location_others, self._other_vehicle_h_wp,  cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))

        target_lane_vehicle_i = py_trees.composites.Parallel(
                    "Target Lane Vehicle I",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if self.num_of_players < 7:
            target_lane_vehicle_i.add_child(IDMInteractiveLaneChangeAgentBehavior(self.other_actors[10], target_location_others, self._other_vehicle_i_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic))


        driving_in_target_lane = py_trees.composites.Parallel(
            "Driving in Target Lane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        driving_in_target_lane.add_child(target_lane_vehicle_a)
        driving_in_target_lane.add_child(target_lane_vehicle_b)
        driving_in_target_lane.add_child(target_lane_vehicle_c)
        driving_in_target_lane.add_child(target_lane_vehicle_d)
        driving_in_target_lane.add_child(target_lane_vehicle_e)
        driving_in_target_lane.add_child(target_lane_vehicle_f)
        driving_in_target_lane.add_child(target_lane_vehicle_g)
        driving_in_target_lane.add_child(target_lane_vehicle_h)
        driving_in_target_lane.add_child(target_lane_vehicle_i)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 51.0, "DriveDistance" )
        endcondition.add_child(endcondition_merged)


        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._emergence_vehicle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._illegal_bicycle_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_vehicle_a_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[3], self._other_vehicle_b_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[4], self._other_vehicle_c_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[5], self._other_vehicle_d_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[6], self._other_vehicle_e_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[7], self._other_vehicle_f_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[8], self._other_vehicle_g_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[9], self._other_vehicle_h_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[10], self._other_vehicle_i_transform))
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0))

        #sequence.add_child(driving_in_target_lane)
        #sequence.add_child(StopVehicle(self.other_actors[2], self._other_vehicle_max_brake))

        #drive_all = py_trees.composites.Parallel( "All cars driving", policy=py_trees.common.ParallelPolicy.SuccessOnSelected(children=[]))
        drive_all = py_trees.composites.Parallel( "All cars driving", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        drive_all.add_child(endcondition_merged)
        drive_all.add_child(driving_in_target_lane)

        sequence.add_child(drive_all)

        #sequence.add_child(driving_in_target_lane)
        #sequence.add_child(endcondition)

        return sequence

    def _on_tick_callback(self, toto):
        #print('***************************************************************************************')
        #print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        #print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        #print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        #print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        #print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        #print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        #if self._debug_mode:
        #print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous!=None:
            delta_time =  curr_time - self._time_previous
            #print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        #velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if (self._velocity_queue.qsize() < self._velocity_filter_size):
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        #acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if (self._acc_queue.qsize() < self._acc_filter_size):
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if (acc_current > self._acc_max):
                    self._acc_max = acc_current
                if (acc_current < self._acc_min):
                    self._acc_min = acc_current
                #self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if (acc_current >= 0.0):
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter+=1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter+=1


        #jerk
        if self._it_counter > self._velocity_filter_size + 2:
            if delta_time > 0.0:
                jerk_current = (acc_current - self._acc_previous)/delta_time
                self._jerk_queue.put(jerk_current)
                if self._jerk_queue.qsize() < self._jerk_filter_size:
                    self._jerk_sum += jerk_current
                else:
                    self._jerk_sum = self._jerk_sum + jerk_current - self._jerk_queue.get()
                jerk_current = self._jerk_sum / self._jerk_queue.qsize()

                if jerk_current > self._jerk_max:
                    self._jerk_max = jerk_current
                if jerk_current < self._jerk_min:
                    self._jerk_min = jerk_current
                #self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter +  jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter+=1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter +  jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter+=1


        #angular velocity
        angular_acc_current = 0.0
        angular_jerk_current = 0.0
        if self.ego_vehicles[0]:
            angular_velocity_current = self.ego_vehicles[0].get_angular_velocity().z
        else:
            angular_velocity_current = 0

        self._angular_velocity_queue.put(angular_velocity_current)
        if self._angular_velocity_queue.qsize() < self._angular_velocity_filter_size:
            self._angular_velocity_sum += angular_velocity_current
        else:
            self._angular_velocity_sum = self._angular_velocity_sum + angular_velocity_current - self._angular_velocity_queue.get()

        angular_velocity_current = self._angular_velocity_sum / self._angular_velocity_queue.qsize()

        if self._it_counter > self._angular_velocity_filter_size:
            #angular acc
            if delta_time > 0.0:
                angular_acc_current = (angular_velocity_current - self._angular_velocity_previous)/delta_time
                self._angular_acc_queue.put(angular_acc_current)
                if (self._angular_acc_queue.qsize() < self._angular_acc_filter_size):
                    self._angular_acc_sum += angular_acc_current
                else:
                    self._angular_acc_sum = self._angular_acc_sum + angular_acc_current - self._angular_acc_queue.get()
                angular_acc_current = self._angular_acc_sum / self._angular_acc_queue.qsize()

                if abs(angular_acc_current) > self._angular_acc_max:
                    self._angular_acc_max = abs(angular_acc_current)

                self._angular_acc_ave = (self._angular_acc_ave * self._it_counter + abs(angular_acc_current)) / (self._it_counter + 1)

        if self._it_counter > self._angular_velocity_filter_size + 2:
            #angular jerk
            if (delta_time > 0.0):
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if (self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size):
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if (abs(angular_jerk_current) > self._angular_jerk_max):
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        #store previous numbers
        self._time_previous = curr_time
        self._velocity_previous = velocity_current
        self._acc_previous = acc_current
        self._angular_velocity_previous = angular_velocity_current
        self._angular_acc_previous = angular_acc_current
        self._it_counter = self._it_counter + 1

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=self.terminate_on_failure)

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        # self.remove_all_actors()
        # if self._on_tick_ref != None:
        #     self._world.remove_on_tick(self._on_tick_ref)
        print('========================== ')

    def remove_on_tick(self):
        """
        Remove on_tick
        """
        if self._on_tick_ref != None:
            self._world.remove_on_tick(self._on_tick_ref)
