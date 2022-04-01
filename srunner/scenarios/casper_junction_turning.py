#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

from __future__ import print_function

import math
import py_trees
import random

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
                                                                      IDMDriveDistanceBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTriggerDistanceToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           generate_target_waypoint_in_route,
                                           get_waypoint_in_distance)


def get_opponent_transform(added_dist, waypoint, trigger_location):
    """
    Calculate the transform of the adversary
    """
    lane_width = waypoint.lane_width

    offset = {"orientation": 270, "position": 90, "k": 1.0}
    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    location = _wp.transform.location
    orientation_yaw = _wp.transform.rotation.yaw + offset["orientation"]
    position_yaw = _wp.transform.rotation.yaw + offset["position"]

    offset_location = carla.Location(
        offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
        offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
    location += offset_location
    location.z = trigger_location.z
    transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))

    return transform


def get_right_driving_lane(waypoint):
    """
    Gets the driving / parking lane that is most to the right of the waypoint
    as well as the number of lane changes done
    """
    lane_changes = 0

    while True:
        wp_next = waypoint.get_right_lane()
        lane_changes += 1

        if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
            break
        elif wp_next.lane_type == carla.LaneType.Shoulder:
            # Filter Parkings considered as Shoulders
            if is_lane_a_parking(wp_next):
                lane_changes += 1
                waypoint = wp_next
            break
        else:
            waypoint = wp_next

    return waypoint, lane_changes


def is_lane_a_parking(waypoint):
    """
    This function filters false negative Shoulder which are in reality Parking lanes.
    These are differentiated from the others because, similar to the driving lanes,
    they have, on the right, a small Shoulder followed by a Sidewalk.
    """

    # Parking are wide lanes
    if waypoint.lane_width > 2:
        wp_next = waypoint.get_right_lane()

        # That are next to a mini-Shoulder
        if wp_next is not None and wp_next.lane_type == carla.LaneType.Shoulder:
            wp_next_next = wp_next.get_right_lane()

            # Followed by a Sidewalk
            if wp_next_next is not None and wp_next_next.lane_type == carla.LaneType.Sidewalk:
                return True

    return False


class CasperJunctionTurning(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_target_velocity = 10
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._other_actor_transform = None
        self._num_lane_changes = 0
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning, self).__init__("CasperJunctionTurning",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Get the waypoint right after the junction
        waypoint = generate_target_waypoint(self._reference_waypoint, 1)

        # Move a certain distance to the front
        start_distance = 8
        waypoint = waypoint.next(start_distance)[0]

        # Get the last driving lane to the right
        waypoint, self._num_lane_changes = get_right_driving_lane(waypoint)
        # And for synchrony purposes, move to the front a bit
        added_dist = self._num_lane_changes

        while True:

            # Try to spawn the actor
            try:
                self._other_actor_transform = get_opponent_transform(added_dist, waypoint, self._trigger_location)
                first_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.diamondback.century', self._other_actor_transform)
                first_vehicle.set_simulate_physics(enabled=False)
                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print(" Base transform is blocking objects ", self._other_actor_transform)
                added_dist += 0.5
                self._spawn_attempted += 1
                if self._spawn_attempted >= self._number_of_attempts:
                    raise r

        # Set the transform to -500 z after we are able to spawn it
        actor_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle.set_transform(actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionRightTurn")

        lane_width = self._reference_waypoint.lane_width
        dist_to_travel = lane_width + (1.10 * lane_width * self._num_lane_changes)

        bycicle_start_dist = 13 + dist_to_travel

        if self._ego_route is not None:
            trigger_distance = InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                     self._ego_route,
                                                                     self._other_actor_transform.location,
                                                                     bycicle_start_dist)
        else:
            trigger_distance = InTriggerDistanceToVehicle(self.other_actors[0],
                                                          self.ego_vehicles[0],
                                                          bycicle_start_dist)

        actor_velocity = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        actor_traverse = DriveDistance(self.other_actors[0], 0.30 * dist_to_travel)
        post_timer_velocity_actor = KeepVelocity(self.other_actors[0], self._other_actor_target_velocity)
        post_timer_traverse_actor = DriveDistance(self.other_actors[0], 0.70 * dist_to_travel)
        end_condition = TimeOut(5)

        # non leaf nodes
        scenario_sequence = py_trees.composites.Sequence()

        actor_ego_sync = py_trees.composites.Parallel(
            "Synchronization of actor and ego vehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        after_timer_actor = py_trees.composites.Parallel(
            "After timeout actor will cross the remaining lane_width",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # building the tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform,
                                                         name='TransformSetterTS4'))
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], True))
        scenario_sequence.add_child(trigger_distance)
        scenario_sequence.add_child(HandBrakeVehicle(self.other_actors[0], False))
        scenario_sequence.add_child(actor_ego_sync)
        scenario_sequence.add_child(after_timer_actor)
        scenario_sequence.add_child(end_condition)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        actor_ego_sync.add_child(actor_velocity)
        actor_ego_sync.add_child(actor_traverse)

        after_timer_actor.add_child(post_timer_velocity_actor)
        after_timer_actor.add_child(post_timer_traverse_actor)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class CasperJunctionTurning2(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a right turn. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 terminate_on_failure=True, timeout=60, report_enable=False):
        """
        Setup all relevant parameters and create scenario
        """

        self._world = world
        self._wmap = CarlaDataProvider.get_map()

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0
        # Need a report?
        # self._report_enable = report_enable

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial setting of traffic participants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_vehicle_transform = None

        self._cyclist_target_velocity = 10
        self._cyclist_transform = None
        self._other_actor_transforms = []

        self._cooperative_drivers = True
        self._dense_traffic = True

        self.closer_lane_start_waypoint = None
        self.further_lane_start_waypoint = None

        self._closer_lane_vehicle_1_location = 10
        self._closer_lane_vehicle_2_location = -5 + random.random() * 10 + 30
        self._closer_lane_vehicle_3_location = -5 + random.random() * 10 + 50
        self._closer_lane_vehicle_4_location = -5 + random.random() * 10 + 70
        self._closer_lane_vehicle_5_location = -5 + random.random() * 10 + 90
        self._closer_lane_vehicle_6_location = -5 + random.random() * 10 + 115

        self._closer_lane_vehicle_1_transform = None
        self._closer_lane_vehicle_2_transform = None
        self._closer_lane_vehicle_3_transform = None
        self._closer_lane_vehicle_4_transform = None
        self._closer_lane_vehicle_5_transform = None
        self._closer_lane_vehicle_6_transform = None

        self.closer_lane_vehicle_1_waypoint = None
        self.closer_lane_vehicle_2_waypoint = None
        self.closer_lane_vehicle_3_waypoint = None
        self.closer_lane_vehicle_4_waypoint = None
        self.closer_lane_vehicle_5_waypoint = None

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning2, self).__init__("CasperJunctionTurning2",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=terminate_on_failure,
                                                 criteria_enable=criteria_enable)

        # Need a report?
        self._report_enable = report_enable

        if self._report_enable:

            #Evaluation report
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

        # add actors from xml file
        for idx, actor in enumerate(config.other_actors):
            if 'vehicle' in actor.model:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, color='255,0,0')
            else:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self._other_actor_transforms.append(actor.transform)
            self.other_actors.append(vehicle)

        closer_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[0].get_location())

        self.closer_lane_start_waypoint = closer_lane_start_waypoint

        closer_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_1_location, junction=True)
        closer_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_2_location, junction=True)
        closer_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_3_location, junction=True)
        closer_lane_vehicle_4_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_4_location, junction=True)
        closer_lane_vehicle_5_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_5_location, junction=True)
        closer_lane_vehicle_6_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_6_location, junction=True)

        self.closer_lane_vehicle_1_waypoint = closer_lane_vehicle_1_waypoint
        self.closer_lane_vehicle_2_waypoint = closer_lane_vehicle_2_waypoint
        self.closer_lane_vehicle_3_waypoint = closer_lane_vehicle_3_waypoint
        self.closer_lane_vehicle_4_waypoint = closer_lane_vehicle_4_waypoint
        self.closer_lane_vehicle_5_waypoint = closer_lane_vehicle_5_waypoint
        self.closer_lane_vehicle_6_waypoint = closer_lane_vehicle_5_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._reference_waypoint.transform.location.x,
                           self._reference_waypoint.transform.location.y,
                           self._reference_waypoint.transform.location.z + 1),
            self._reference_waypoint.transform.rotation)

        closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)
        self._closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)

        closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)
        self._closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)

        closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)
        self._closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)

        closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)
        self._closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)

        closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)
        self._closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)

        closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)
        self._closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)

        closer_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_1_transform, color='255,0,0')
        closer_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_2_transform, color='255,0,0')
        closer_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_3_transform, color='255,0,0')
        closer_lane_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_4_transform, color='255,0,0')
        closer_lane_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_5_transform, color='255,0,0')
        closer_lane_vehicle_6 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_6_transform, color='255,0,0')

        self._other_actor_transforms.append(self._closer_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_3_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_4_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_5_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_6_transform)

        self.other_actors.append(closer_lane_vehicle_1)
        self.other_actors.append(closer_lane_vehicle_2)
        self.other_actors.append(closer_lane_vehicle_3)
        self.other_actors.append(closer_lane_vehicle_4)
        self.other_actors.append(closer_lane_vehicle_5)
        self.other_actors.append(closer_lane_vehicle_6)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a right turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """

        lane_width = self._reference_waypoint.lane_width

        target_distance = 1000
        closer_lane_target_location = carla.Location(92.4, 20, 0)

        spawn_all = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SpawnAll")

        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[0], 
                                                             self._other_actor_transforms[0],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[2],
                                                             self._other_actor_transforms[2],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[3],
                                                             self._other_actor_transforms[3],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[4],
                                                             self._other_actor_transforms[4],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh3'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[5],
                                                             self._other_actor_transforms[5],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh4'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[6],
                                                             self._other_actor_transforms[6],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh5'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[7],
                                                             self._other_actor_transforms[7],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh6'))

        # closer_lane_veh0_sequence
        closer_lane_veh0_sequence = py_trees.composites.Sequence()
        closer_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[0],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_start_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh0"))

        # side_walk_ped0_sequence
        ped0_location0 = self.other_actors[1].get_location()
        ped0_location1 = carla.Location(ped0_location0.x, ped0_location0.y - 16)
        ped0_location2 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 16)
        ped0_location3 = carla.Location(ped0_location0.x, ped0_location0.y - 2)
        ped0_location4 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 2)
        ped0_plan = None
        demo_id = random.choice([0, 1, 2])
        if demo_id == 0:
            ped0_plan = [ped0_location1]
        elif demo_id == 1:
            ped0_plan = [ped0_location1, ped0_location2]
        elif demo_id == 2:
            ped0_plan = [ped0_location3, ped0_location4]
        side_walk_ped0_sequence = py_trees.composites.Sequence()
        side_walk_ped0_sequence.add_child(BasicPedestrianBehavior(self.other_actors[1],
                                                                  target_speed=1,
                                                                  plan=ped0_plan,
                                                                  blackboard_queue_name=None,
                                                                  avoid_collision=False,
                                                                  name="BasicPedestrianBehavior"))
        # Send the pedestrian to far away and keep walking
        alien_transform = carla.Transform(carla.Location(ped0_location0.x,
                                                         -ped0_location0.y,
                                                         ped0_location0.z+1.0))
        side_walk_ped0_sequence.add_child(ActorTransformSetter(self.other_actors[1],
                                                               alien_transform,
                                                               False,
                                                               name="ActorTransformerPed0"))
        side_walk_ped0_sequence.add_child(KeepVelocity(self.other_actors[1],
                                                       target_velocity=1.0,
                                                       distance=1000,
                                                       name='KeepVelocityPed0'))

        # closer_lane_veh1_sequence
        closer_lane_veh1_sequence = py_trees.composites.Sequence()
        closer_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[2],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_1_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh1"))

        # closer_lane_veh2_sequence
        closer_lane_veh2_sequence = py_trees.composites.Sequence()
        closer_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[3],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_2_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh2"))

        # closer_lane_veh3_sequence
        closer_lane_veh3_sequence = py_trees.composites.Sequence()
        closer_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[4],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_3_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh3"))

        # closer_lane_veh4_sequence
        closer_lane_veh4_sequence = py_trees.composites.Sequence()
        closer_lane_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[5],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_4_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh4"))

        # closer_lane_veh5_sequence
        closer_lane_veh5_sequence = py_trees.composites.Sequence()
        closer_lane_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[6],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_5_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh5"))

        # closer_lane_veh6_sequence
        closer_lane_veh6_sequence = py_trees.composites.Sequence()
        closer_lane_veh6_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[7],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_6_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh6"))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 40.0, "DriveDistance")
        endcondition.add_child(endcondition_merged)

        # building the tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(spawn_all)
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0)) # -random.random()*3

        closer_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="CloserLane")
        closer_lane_parallel.add_child(closer_lane_veh0_sequence)
        closer_lane_parallel.add_child(closer_lane_veh1_sequence)
        closer_lane_parallel.add_child(closer_lane_veh2_sequence)
        closer_lane_parallel.add_child(closer_lane_veh3_sequence)
        closer_lane_parallel.add_child(closer_lane_veh4_sequence)
        closer_lane_parallel.add_child(closer_lane_veh5_sequence)
        closer_lane_parallel.add_child(closer_lane_veh6_sequence)

        side_walk_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SideWalk")
        side_walk_parallel.add_child(side_walk_ped0_sequence)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")
        root.add_child(closer_lane_parallel)
        root.add_child(side_walk_parallel)
        root.add_child(endcondition_merged)

        sequence.add_child(root)

        return sequence

    def _on_tick_callback(self, toto):
        # print('***************************************************************************************')
        # print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        # print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        # print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        # print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        # print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        # print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        # if self._debug_mode:
        #     print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous:
            delta_time = curr_time - self._time_previous
            # print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        # velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2)
                + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if self._velocity_queue.qsize() < self._velocity_filter_size:
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        # acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if self._acc_queue.qsize() < self._acc_filter_size:
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if acc_current > self._acc_max:
                    self._acc_max = acc_current
                if acc_current < self._acc_min:
                    self._acc_min = acc_current
                # self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if acc_current >= 0.0:
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter += 1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter += 1


        # jerk
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
                # self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter + jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter += 1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter + jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter += 1

        # angular velocity
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
            # angular acc
            if delta_time > 0.0:
                angular_acc_current = (angular_velocity_current - self._angular_velocity_previous)/delta_time
                self._angular_acc_queue.put(angular_acc_current)
                if self._angular_acc_queue.qsize() < self._angular_acc_filter_size:
                    self._angular_acc_sum += angular_acc_current
                else:
                    self._angular_acc_sum = self._angular_acc_sum + angular_acc_current - self._angular_acc_queue.get()
                angular_acc_current = self._angular_acc_sum / self._angular_acc_queue.qsize()

                if abs(angular_acc_current) > self._angular_acc_max:
                    self._angular_acc_max = abs(angular_acc_current)

                self._angular_acc_ave = (self._angular_acc_ave * self._it_counter + abs(angular_acc_current)) / (self._it_counter + 1)

        if self._it_counter > self._angular_velocity_filter_size + 2:
            # angular jerk
            if delta_time > 0.0:
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size:
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if abs(angular_jerk_current) > self._angular_jerk_max:
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        # store previous numbers
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
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria
    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        return AlwaysSuccessTrigger()



    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class CasperJunctionTurning3(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. Scenario 4
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 terminate_on_failure=True, timeout=120, report_enable=False):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._wmap = CarlaDataProvider.get_map()

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0
        # Need a report?
        # self._report_enable = report_enable

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial setting of traffic participants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_vehicle_transform = None

        self._cooperative_drivers = True
        self._dense_traffic = True

        self._cyclist_target_velocity = 10
        self._cyclist_transform = None
        self._other_actor_transforms = []

        self._other_vehicle_4_location = 10
        self._other_vehicle_5_location = 20
        self._other_vehicle_6_location = 30
        self._other_vehicle_7_location = 40

        self._other_vehicle_4_transform = None
        self._other_vehicle_5_transform = None
        self._other_vehicle_6_transform = None
        self._other_vehicle_7_transform = None

        self.closer_lane_start_waypoint = None
        self.further_lane_start_waypoint = None
        self.other_vehicle_4_waypoint = None
        self.other_vehicle_5_waypoint = None
        self.other_vehicle_6_waypoint = None
        self.other_vehicle_7_waypoint = None

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning3, self).__init__("CasperJunctionTurning3",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=terminate_on_failure,
                                                 criteria_enable=criteria_enable)

        # Need a report?
        self._report_enable = report_enable

        if self._report_enable:

            #Evaluation report
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

        # add actors from xml file
        for idx, actor in enumerate(config.other_actors):
            if 'vehicle' in actor.model:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, color='255,0,0')
            else:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self._other_actor_transforms.append(actor.transform)
            self.other_actors.append(vehicle)

        closer_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[0].get_location())
        further_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[1].get_location())

        other_vehicle_4_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._other_vehicle_4_location)
        other_vehicle_5_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._other_vehicle_5_location)
        other_vehicle_6_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._other_vehicle_6_location)
        other_vehicle_7_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._other_vehicle_7_location)

        self.closer_lane_start_waypoint = closer_lane_start_waypoint
        self.further_lane_start_waypoint = further_lane_start_waypoint

        self.other_vehicle_4_waypoint = other_vehicle_4_waypoint
        self.other_vehicle_5_waypoint = other_vehicle_5_waypoint
        self.other_vehicle_6_waypoint = other_vehicle_6_waypoint
        self.other_vehicle_7_waypoint = other_vehicle_7_waypoint

        other_vehicle_4_transform = carla.Transform(
            carla.Location(other_vehicle_4_waypoint.transform.location.x,
                           other_vehicle_4_waypoint.transform.location.y,
                           other_vehicle_4_waypoint.transform.location.z),
            other_vehicle_4_waypoint.transform.rotation)
        self._other_vehicle_4_transform = carla.Transform(
            carla.Location(other_vehicle_4_waypoint.transform.location.x,
                           other_vehicle_4_waypoint.transform.location.y,
                           other_vehicle_4_waypoint.transform.location.z),
            other_vehicle_4_waypoint.transform.rotation)

        other_vehicle_5_transform = carla.Transform(
            carla.Location(other_vehicle_5_waypoint.transform.location.x,
                           other_vehicle_5_waypoint.transform.location.y,
                           other_vehicle_5_waypoint.transform.location.z),
            other_vehicle_5_waypoint.transform.rotation)
        self._other_vehicle_5_transform = carla.Transform(
            carla.Location(other_vehicle_5_waypoint.transform.location.x,
                           other_vehicle_5_waypoint.transform.location.y,
                           other_vehicle_5_waypoint.transform.location.z),
            other_vehicle_5_waypoint.transform.rotation)

        other_vehicle_6_transform = carla.Transform(
            carla.Location(other_vehicle_6_waypoint.transform.location.x,
                           other_vehicle_6_waypoint.transform.location.y,
                           other_vehicle_6_waypoint.transform.location.z),
            other_vehicle_6_waypoint.transform.rotation)
        self._other_vehicle_6_transform = carla.Transform(
            carla.Location(other_vehicle_6_waypoint.transform.location.x,
                           other_vehicle_6_waypoint.transform.location.y,
                           other_vehicle_6_waypoint.transform.location.z),
            other_vehicle_6_waypoint.transform.rotation)

        other_vehicle_7_transform = carla.Transform(
            carla.Location(other_vehicle_7_waypoint.transform.location.x,
                           other_vehicle_7_waypoint.transform.location.y,
                           other_vehicle_7_waypoint.transform.location.z),
            other_vehicle_7_waypoint.transform.rotation)
        self._other_vehicle_7_transform = carla.Transform(
            carla.Location(other_vehicle_7_waypoint.transform.location.x,
                           other_vehicle_7_waypoint.transform.location.y,
                           other_vehicle_7_waypoint.transform.location.z),
            other_vehicle_7_waypoint.transform.rotation)

        other_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_4_transform, color='255,0,0')
        other_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_5_transform, color='255,0,0')
        other_vehicle_6 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_6_transform, color='255,0,0')
        other_vehicle_7 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', other_vehicle_7_transform, color='255,0,0')

        self._other_actor_transforms.append(self._other_vehicle_4_transform)
        self._other_actor_transforms.append(self._other_vehicle_5_transform)
        self._other_actor_transforms.append(self._other_vehicle_6_transform)
        self._other_actor_transforms.append(self._other_vehicle_7_transform)

        self.other_actors.append(other_vehicle_4)
        self.other_actors.append(other_vehicle_5)
        self.other_actors.append(other_vehicle_6)
        self.other_actors.append(other_vehicle_7)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="IntersectionLeftTurn")

        lane_width = self._reference_waypoint.lane_width

        target_distance = 120
        closer_lane_target_location = carla.Location(92.5, 11.7, 0)
        further_lane_target_location = carla.Location(88.2, 180, 0)

        # other_veh1_sequence
        other_veh1_sequence = py_trees.composites.Sequence()
        other_veh1_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[0], self._other_actor_transforms[0],
                                                           velocity=8,
                                                           name='TransformSetterVeh1'))
        other_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[0],
                                                               target_distance,
                                                               target_location=closer_lane_target_location,
                                                               start_waypoint=self.closer_lane_start_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh1"))

        # other_veh2_sequence
        other_veh2_sequence = py_trees.composites.Sequence()
        other_veh2_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[1], self._other_actor_transforms[1],
                                                           velocity=8,
                                                           name='TransformSetterVeh2'))
        other_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[1],
                                                               target_distance,
                                                               target_location=further_lane_target_location,
                                                               start_waypoint=self.further_lane_start_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh2"))

        # other_ped0_sequence
        other_ped0_sequence = py_trees.composites.Sequence()
        # other_ped0_sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_actor_transforms[2],
        #                                                    name='TransformSetterPed0'))
        other_ped0_sequence.add_child(KeepVelocity(self.other_actors[2],
                                                  target_velocity=1,
                                                  name='DrivePed0'))

        # other_veh4_sequence
        other_veh4_sequence = py_trees.composites.Sequence()
        other_veh4_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[3], self._other_actor_transforms[3],
                                                           velocity=8,
                                                           name='TransformSetterVeh4'))
        other_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[3],
                                                               target_distance,
                                                               target_location=further_lane_target_location,
                                                               start_waypoint=self.other_vehicle_4_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh4"))

        # other_veh5_sequence
        other_veh5_sequence = py_trees.composites.Sequence()
        other_veh5_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[4], self._other_actor_transforms[4],
                                                           velocity=8,
                                                           name='TransformSetterVeh5'))
        other_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[4],
                                                               target_distance,
                                                               target_location=further_lane_target_location,
                                                               start_waypoint=self.other_vehicle_5_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh5"))

        # other_veh6_sequence
        other_veh6_sequence = py_trees.composites.Sequence()
        other_veh6_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[5], self._other_actor_transforms[5],
                                                           velocity=8,
                                                           name='TransformSetterVeh6'))
        other_veh6_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[5],
                                                               target_distance,
                                                               target_location=further_lane_target_location,
                                                               start_waypoint=self.other_vehicle_6_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh6"))

        # other_veh7_sequence
        other_veh7_sequence = py_trees.composites.Sequence()
        other_veh7_sequence.add_child(ActorTransformSetterWithVelocity(self.other_actors[6], self._other_actor_transforms[6],
                                                           velocity=8,
                                                           name='TransformSetterVeh7'))
        other_veh7_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[6],
                                                               target_distance,
                                                               target_location=further_lane_target_location,
                                                               start_waypoint=self.other_vehicle_7_waypoint,
                                                               reference_speed=8,
                                                               cooperative_driver=self._cooperative_drivers,
                                                               dense_traffic=self._dense_traffic,
                                                               name="DriveVeh7"))

        # building the tree
        root.add_child(other_veh1_sequence)
        root.add_child(other_veh2_sequence)
        root.add_child(other_ped0_sequence)

        root.add_child(other_veh4_sequence)
        root.add_child(other_veh5_sequence)
        root.add_child(other_veh6_sequence)
        root.add_child(other_veh7_sequence)

        return root

    def _on_tick_callback(self, toto):
        # print('***************************************************************************************')
        # print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        # print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        # print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        # print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        # print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        # print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        # if self._debug_mode:
        #     print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous:
            delta_time = curr_time - self._time_previous
            # print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        # velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if self._velocity_queue.qsize() < self._velocity_filter_size:
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        # acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if self._acc_queue.qsize() < self._acc_filter_size:
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if acc_current > self._acc_max:
                    self._acc_max = acc_current
                if acc_current < self._acc_min:
                    self._acc_min = acc_current
                # self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if acc_current >= 0.0:
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter += 1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter += 1


        # jerk
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
                # self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter + jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter += 1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter + jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter += 1

        # angular velocity
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
            # angular acc
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
            # angular jerk
            if delta_time > 0.0:
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size:
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if abs(angular_jerk_current) > self._angular_jerk_max:
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        # store previous numbers
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
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

class CasperJunctionTurning6(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. Scenario 4
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 terminate_on_failure=True, timeout=50, report_enable=False):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._wmap = CarlaDataProvider.get_map()

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0
        # Need a report?
        # self._report_enable = report_enable

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial setting of traffic participants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_vehicle_transform = None

        self._cooperative_drivers = True
        self._dense_traffic = True

        self._cyclist_target_velocity = 10
        self._cyclist_transform = None
        self._other_actor_transforms = []

        self.closer_lane_start_waypoint = None
        self.further_lane_start_waypoint = None

        self._closer_lane_vehicle_1_location = 10
        self._closer_lane_vehicle_2_location = -5 + random.random() * 10 + 30
        self._closer_lane_vehicle_3_location = -5 + random.random() * 10 + 50
        self._closer_lane_vehicle_4_location = -5 + random.random() * 10 + 70
        self._closer_lane_vehicle_5_location = -5 + random.random() * 10 + 90
        self._closer_lane_vehicle_6_location = -5 + random.random() * 10 + 115

        self._closer_lane_vehicle_1_transform = None
        self._closer_lane_vehicle_2_transform = None
        self._closer_lane_vehicle_3_transform = None
        self._closer_lane_vehicle_4_transform = None
        self._closer_lane_vehicle_5_transform = None
        self._closer_lane_vehicle_6_transform = None

        self.closer_lane_vehicle_1_waypoint = None
        self.closer_lane_vehicle_2_waypoint = None
        self.closer_lane_vehicle_3_waypoint = None
        self.closer_lane_vehicle_4_waypoint = None
        self.closer_lane_vehicle_5_waypoint = None

        self._further_lane_vehicle_1_location = -5 + random.random() * 10 + 15
        self._further_lane_vehicle_2_location = -5 + random.random() * 10 + 35
        self._further_lane_vehicle_3_location = -5 + random.random() * 10 + 55
        self._further_lane_vehicle_4_location = -5 + random.random() * 10 + 75
        self._further_lane_vehicle_5_location = -5 + random.random() * 10 + 100

        self._further_lane_vehicle_1_transform = None
        self._further_lane_vehicle_2_transform = None
        self._further_lane_vehicle_3_transform = None
        self._further_lane_vehicle_4_transform = None
        self._further_lane_vehicle_5_transform = None

        self.further_lane_vehicle_1_waypoint = None
        self.further_lane_vehicle_2_waypoint = None
        self.further_lane_vehicle_3_waypoint = None
        self.further_lane_vehicle_4_waypoint = None
        self.further_lane_vehicle_5_waypoint = None

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning6, self).__init__("CasperJunctionTurning6",
                                                ego_vehicles,
                                                config,
                                                world,
                                                debug_mode,
                                                terminate_on_failure=terminate_on_failure,
                                                criteria_enable=criteria_enable)

        # Need a report?
        self._report_enable = report_enable

        if self._report_enable:

            #Evaluation report
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

        # add actors from xml file
        for idx, actor in enumerate(config.other_actors):
            if 'vehicle' in actor.model:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, color='255,0,0')
            else:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self._other_actor_transforms.append(actor.transform)
            self.other_actors.append(vehicle)

        closer_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[0].get_location())
        further_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[1].get_location())

        self.closer_lane_start_waypoint = closer_lane_start_waypoint
        self.further_lane_start_waypoint = further_lane_start_waypoint

        closer_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_1_location, junction=True)
        closer_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_2_location, junction=True)
        closer_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_3_location, junction=True)
        closer_lane_vehicle_4_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_4_location, junction=True)
        closer_lane_vehicle_5_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_5_location, junction=True)
        closer_lane_vehicle_6_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_6_location, junction=True)

        further_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_1_location, junction=True)
        further_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_2_location, junction=True)
        further_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_3_location, junction=True)
        further_lane_vehicle_4_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_4_location, junction=True)
        further_lane_vehicle_5_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_5_location, junction=True)

        self.closer_lane_vehicle_1_waypoint = closer_lane_vehicle_1_waypoint
        self.closer_lane_vehicle_2_waypoint = closer_lane_vehicle_2_waypoint
        self.closer_lane_vehicle_3_waypoint = closer_lane_vehicle_3_waypoint
        self.closer_lane_vehicle_4_waypoint = closer_lane_vehicle_4_waypoint
        self.closer_lane_vehicle_5_waypoint = closer_lane_vehicle_5_waypoint
        self.closer_lane_vehicle_6_waypoint = closer_lane_vehicle_5_waypoint

        self.further_lane_vehicle_1_waypoint = further_lane_vehicle_1_waypoint
        self.further_lane_vehicle_2_waypoint = further_lane_vehicle_2_waypoint
        self.further_lane_vehicle_3_waypoint = further_lane_vehicle_3_waypoint
        self.further_lane_vehicle_4_waypoint = further_lane_vehicle_4_waypoint
        self.further_lane_vehicle_5_waypoint = further_lane_vehicle_5_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._reference_waypoint.transform.location.x,
                           self._reference_waypoint.transform.location.y,
                           self._reference_waypoint.transform.location.z + 1),
            self._reference_waypoint.transform.rotation)

        closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)
        self._closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)

        closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)
        self._closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)

        closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)
        self._closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)

        closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)
        self._closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)

        closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)
        self._closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)

        closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)
        self._closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)

        further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)
        self._further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)

        further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)
        self._further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)

        further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)
        self._further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)

        further_lane_vehicle_4_transform = carla.Transform(
            carla.Location(further_lane_vehicle_4_waypoint.transform.location.x,
                           further_lane_vehicle_4_waypoint.transform.location.y,
                           further_lane_vehicle_4_waypoint.transform.location.z),
            further_lane_vehicle_4_waypoint.transform.rotation)
        self._further_lane_vehicle_4_transform = carla.Transform(
            carla.Location(further_lane_vehicle_4_waypoint.transform.location.x,
                           further_lane_vehicle_4_waypoint.transform.location.y,
                           further_lane_vehicle_4_waypoint.transform.location.z),
            further_lane_vehicle_4_waypoint.transform.rotation)

        further_lane_vehicle_5_transform = carla.Transform(
            carla.Location(further_lane_vehicle_5_waypoint.transform.location.x,
                           further_lane_vehicle_5_waypoint.transform.location.y,
                           further_lane_vehicle_5_waypoint.transform.location.z),
            further_lane_vehicle_5_waypoint.transform.rotation)
        self._further_lane_vehicle_5_transform = carla.Transform(
            carla.Location(further_lane_vehicle_5_waypoint.transform.location.x,
                           further_lane_vehicle_5_waypoint.transform.location.y,
                           further_lane_vehicle_5_waypoint.transform.location.z),
            further_lane_vehicle_5_waypoint.transform.rotation)

        closer_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_1_transform, color='255,0,0')
        closer_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_2_transform, color='255,0,0')
        closer_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_3_transform, color='255,0,0')
        closer_lane_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_4_transform, color='255,0,0')
        closer_lane_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_5_transform, color='255,0,0')
        closer_lane_vehicle_6 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_6_transform, color='255,0,0')

        further_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_1_transform, color='255,0,0')
        further_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_2_transform, color='255,0,0')
        further_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_3_transform, color='255,0,0')
        further_lane_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_4_transform, color='255,0,0')
        further_lane_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_5_transform, color='255,0,0')

        self._other_actor_transforms.append(self._closer_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_3_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_4_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_5_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_6_transform)

        self._other_actor_transforms.append(self._further_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_3_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_4_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_5_transform)

        self.other_actors.append(closer_lane_vehicle_1)
        self.other_actors.append(closer_lane_vehicle_2)
        self.other_actors.append(closer_lane_vehicle_3)
        self.other_actors.append(closer_lane_vehicle_4)
        self.other_actors.append(closer_lane_vehicle_5)
        self.other_actors.append(closer_lane_vehicle_6)

        self.other_actors.append(further_lane_vehicle_1)
        self.other_actors.append(further_lane_vehicle_2)
        self.other_actors.append(further_lane_vehicle_3)
        self.other_actors.append(further_lane_vehicle_4)
        self.other_actors.append(further_lane_vehicle_5)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        lane_width = self._reference_waypoint.lane_width

        target_distance = 1000
        closer_lane_target_location = carla.Location(92.4, 20, 0)
        further_lane_target_location = carla.Location(88.4, 180, 0)

        spawn_all = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SpawnAll")

        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[0], 
                                                             self._other_actor_transforms[0],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[1],
                                                             self._other_actor_transforms[1],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[3],
                                                             self._other_actor_transforms[3],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[4],
                                                             self._other_actor_transforms[4],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[5],
                                                             self._other_actor_transforms[5],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh3'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[6],
                                                             self._other_actor_transforms[6],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh4'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[7],
                                                             self._other_actor_transforms[7],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh5'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[8],
                                                             self._other_actor_transforms[8],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh6'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[9],
                                                             self._other_actor_transforms[9],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[10],
                                                             self._other_actor_transforms[10],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[11],
                                                             self._other_actor_transforms[11],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh3'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[12],
                                                             self._other_actor_transforms[12],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh4'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[13],
                                                             self._other_actor_transforms[13],
                                                             velocity=4,
                                                             name='TransformSetteFurtherVeh5'))

        # closer_lane_veh0_sequence
        closer_lane_veh0_sequence = py_trees.composites.Sequence()
        closer_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[0],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_start_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh0"))

        # further_lane_veh0_sequence
        further_lane_veh0_sequence = py_trees.composites.Sequence()
        further_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[1],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_start_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh0"))

        # side_walk_ped0_sequence
        ped0_location0 = self.other_actors[2].get_location()
        ped0_location1 = carla.Location(ped0_location0.x, ped0_location0.y - 16)
        ped0_location2 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 16)
        ped0_location3 = carla.Location(ped0_location0.x, ped0_location0.y - 2)
        ped0_location4 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 2)
        ped0_plan = None
        demo_id = random.choice([0, 1, 2])
        if demo_id == 0:
            ped0_plan = [ped0_location1]
        elif demo_id == 1:
            ped0_plan = [ped0_location1, ped0_location2]
        elif demo_id == 2:
            ped0_plan = [ped0_location3, ped0_location4]
        side_walk_ped0_sequence = py_trees.composites.Sequence()
        side_walk_ped0_sequence.add_child(BasicPedestrianBehavior(self.other_actors[2],
                                                                  target_speed=1,
                                                                  plan=ped0_plan,
                                                                  blackboard_queue_name=None,
                                                                  avoid_collision=False,
                                                                  name="BasicPedestrianBehavior"))
        # Send the pedestrian to far away and keep walking
        alien_transform = carla.Transform(carla.Location(ped0_location0.x,
                                                         -ped0_location0.y,
                                                         ped0_location0.z+1.0))
        side_walk_ped0_sequence.add_child(ActorTransformSetter(self.other_actors[2],
                                                               alien_transform,
                                                               False,
                                                               name="ActorTransformerPed0"))
        side_walk_ped0_sequence.add_child(KeepVelocity(self.other_actors[2],
                                                       target_velocity=1.0,
                                                       distance=1000,
                                                       name='KeepVelocityPed0'))

        # closer_lane_veh1_sequence
        closer_lane_veh1_sequence = py_trees.composites.Sequence()
        closer_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[3],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_1_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh1"))

        # closer_lane_veh2_sequence
        closer_lane_veh2_sequence = py_trees.composites.Sequence()
        closer_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[4],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_2_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh2"))

        # closer_lane_veh3_sequence
        closer_lane_veh3_sequence = py_trees.composites.Sequence()
        closer_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[5],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_3_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh3"))

        # closer_lane_veh4_sequence
        closer_lane_veh4_sequence = py_trees.composites.Sequence()
        closer_lane_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[6],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_4_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh4"))

        # closer_lane_veh5_sequence
        closer_lane_veh5_sequence = py_trees.composites.Sequence()
        closer_lane_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[7],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_5_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh5"))

        # closer_lane_veh6_sequence
        closer_lane_veh6_sequence = py_trees.composites.Sequence()
        closer_lane_veh6_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[8],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_6_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh6"))

        # further_lane_veh1_sequence
        further_lane_veh1_sequence = py_trees.composites.Sequence()
        further_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[9],
                                                                     target_distance,
                                                                     target_location=further_lane_target_location,
                                                                     start_waypoint=self.further_lane_vehicle_1_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="FurtherVeh1"))

        # further_lane_veh2_sequence
        further_lane_veh2_sequence = py_trees.composites.Sequence()
        further_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[10],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_2_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh2"))

        # further_lane_veh3_sequence
        further_lane_veh3_sequence = py_trees.composites.Sequence()
        further_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[11],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_3_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh3"))

        # further_lane_veh4_sequence
        further_lane_veh4_sequence = py_trees.composites.Sequence()
        further_lane_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[12],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_4_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh4"))

        # further_lane_veh5_sequence
        further_lane_veh5_sequence = py_trees.composites.Sequence()
        further_lane_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[13],
                                                                     target_distance,
                                                                     target_location=further_lane_target_location,
                                                                     start_waypoint=self.further_lane_vehicle_5_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="FurtherVeh5"))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 40.0, "DriveDistance")
        endcondition.add_child(endcondition_merged)

        # building the tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(spawn_all)
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0)) # -random.random()*3

        closer_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="CloserLane")
        closer_lane_parallel.add_child(closer_lane_veh0_sequence)
        closer_lane_parallel.add_child(closer_lane_veh1_sequence)
        closer_lane_parallel.add_child(closer_lane_veh2_sequence)
        closer_lane_parallel.add_child(closer_lane_veh3_sequence)
        closer_lane_parallel.add_child(closer_lane_veh4_sequence)
        closer_lane_parallel.add_child(closer_lane_veh5_sequence)
        closer_lane_parallel.add_child(closer_lane_veh6_sequence)

        further_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="FurtherLane")
        further_lane_parallel.add_child(further_lane_veh0_sequence)
        further_lane_parallel.add_child(further_lane_veh1_sequence)
        further_lane_parallel.add_child(further_lane_veh2_sequence)
        further_lane_parallel.add_child(further_lane_veh3_sequence)
        further_lane_parallel.add_child(further_lane_veh4_sequence)
        further_lane_parallel.add_child(further_lane_veh5_sequence)

        side_walk_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SideWalk")
        side_walk_parallel.add_child(side_walk_ped0_sequence)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")
        root.add_child(closer_lane_parallel)
        root.add_child(further_lane_parallel)
        root.add_child(side_walk_parallel)
        root.add_child(endcondition_merged)

        sequence.add_child(root)

        return sequence

    def _on_tick_callback(self, toto):
        # print('***************************************************************************************')
        # print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        # print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        # print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        # print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        # print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        # print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        # if self._debug_mode:
        #     print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous:
            delta_time = curr_time - self._time_previous
            # print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        # velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if self._velocity_queue.qsize() < self._velocity_filter_size:
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        # acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if self._acc_queue.qsize() < self._acc_filter_size:
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if acc_current > self._acc_max:
                    self._acc_max = acc_current
                if acc_current < self._acc_min:
                    self._acc_min = acc_current
                # self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if acc_current >= 0.0:
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter += 1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter += 1


        # jerk
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
                # self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter + jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter += 1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter + jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter += 1

        # angular velocity
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
            # angular acc
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
            # angular jerk
            if delta_time > 0.0:
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size:
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if abs(angular_jerk_current) > self._angular_jerk_max:
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        # store previous numbers
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
        self.remove_all_actors()

class CasperJunctionTurning6_HighSpeed(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. Scenario 6
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 terminate_on_failure=True, timeout=50, report_enable=False):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._wmap = CarlaDataProvider.get_map()

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0
        # Need a report?
        # self._report_enable = report_enable

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial setting of traffic participants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_vehicle_transform = None

        self._cooperative_drivers = True
        self._dense_traffic = True

        self._cyclist_target_velocity = 10
        self._cyclist_transform = None
        self._other_actor_transforms = []

        self.closer_lane_start_waypoint = None
        self.further_lane_start_waypoint = None

        self._closer_lane_vehicle_1_location = 10
        self._closer_lane_vehicle_2_location = -5 + random.random() * 10 + 35
        self._closer_lane_vehicle_3_location = -5 + random.random() * 10 + 60
        self._closer_lane_vehicle_4_location = -5 + random.random() * 10 + 85
        self._closer_lane_vehicle_5_location = -5 + random.random() * 10 + 110
        self._closer_lane_vehicle_6_location = -5 + random.random() * 10 + 135

        self._closer_lane_vehicle_1_transform = None
        self._closer_lane_vehicle_2_transform = None
        self._closer_lane_vehicle_3_transform = None
        self._closer_lane_vehicle_4_transform = None
        self._closer_lane_vehicle_5_transform = None
        self._closer_lane_vehicle_6_transform = None

        self.closer_lane_vehicle_1_waypoint = None
        self.closer_lane_vehicle_2_waypoint = None
        self.closer_lane_vehicle_3_waypoint = None
        self.closer_lane_vehicle_4_waypoint = None
        self.closer_lane_vehicle_5_waypoint = None

        self._further_lane_vehicle_1_location = -5 + random.random() * 10 + 20
        self._further_lane_vehicle_2_location = -5 + random.random() * 10 + 45
        self._further_lane_vehicle_3_location = -5 + random.random() * 10 + 70
        self._further_lane_vehicle_4_location = -5 + random.random() * 10 + 95
        self._further_lane_vehicle_5_location = -5 + random.random() * 10 + 125

        self._further_lane_vehicle_1_transform = None
        self._further_lane_vehicle_2_transform = None
        self._further_lane_vehicle_3_transform = None
        self._further_lane_vehicle_4_transform = None
        self._further_lane_vehicle_5_transform = None

        self.further_lane_vehicle_1_waypoint = None
        self.further_lane_vehicle_2_waypoint = None
        self.further_lane_vehicle_3_waypoint = None
        self.further_lane_vehicle_4_waypoint = None
        self.further_lane_vehicle_5_waypoint = None

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning6_HighSpeed, self).__init__("CasperJunctionTurning6_HighSpeed",
                                                            ego_vehicles,
                                                            config,
                                                            world,
                                                            debug_mode,
                                                            terminate_on_failure=terminate_on_failure,
                                                            criteria_enable=criteria_enable)

        # Need a report?
        self._report_enable = report_enable

        if self._report_enable:

            #Evaluation report
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

        # add actors from xml file
        for idx, actor in enumerate(config.other_actors):
            if 'vehicle' in actor.model:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, color='255,0,0')
            else:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self._other_actor_transforms.append(actor.transform)
            self.other_actors.append(vehicle)

        closer_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[0].get_location())
        further_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[1].get_location())

        self.closer_lane_start_waypoint = closer_lane_start_waypoint
        self.further_lane_start_waypoint = further_lane_start_waypoint

        closer_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_1_location, junction=True)
        closer_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_2_location, junction=True)
        closer_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_3_location, junction=True)
        closer_lane_vehicle_4_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_4_location, junction=True)
        closer_lane_vehicle_5_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_5_location, junction=True)
        closer_lane_vehicle_6_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_6_location, junction=True)

        further_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_1_location, junction=True)
        further_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_2_location, junction=True)
        further_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_3_location, junction=True)
        further_lane_vehicle_4_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_4_location, junction=True)
        further_lane_vehicle_5_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_5_location, junction=True)

        self.closer_lane_vehicle_1_waypoint = closer_lane_vehicle_1_waypoint
        self.closer_lane_vehicle_2_waypoint = closer_lane_vehicle_2_waypoint
        self.closer_lane_vehicle_3_waypoint = closer_lane_vehicle_3_waypoint
        self.closer_lane_vehicle_4_waypoint = closer_lane_vehicle_4_waypoint
        self.closer_lane_vehicle_5_waypoint = closer_lane_vehicle_5_waypoint
        self.closer_lane_vehicle_6_waypoint = closer_lane_vehicle_5_waypoint

        self.further_lane_vehicle_1_waypoint = further_lane_vehicle_1_waypoint
        self.further_lane_vehicle_2_waypoint = further_lane_vehicle_2_waypoint
        self.further_lane_vehicle_3_waypoint = further_lane_vehicle_3_waypoint
        self.further_lane_vehicle_4_waypoint = further_lane_vehicle_4_waypoint
        self.further_lane_vehicle_5_waypoint = further_lane_vehicle_5_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._reference_waypoint.transform.location.x,
                           self._reference_waypoint.transform.location.y,
                           self._reference_waypoint.transform.location.z + 1),
            self._reference_waypoint.transform.rotation)

        closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)
        self._closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)

        closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)
        self._closer_lane_vehicle_2_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_2_waypoint.transform.location.x,
                           closer_lane_vehicle_2_waypoint.transform.location.y,
                           closer_lane_vehicle_2_waypoint.transform.location.z),
            closer_lane_vehicle_2_waypoint.transform.rotation)

        closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)
        self._closer_lane_vehicle_3_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_3_waypoint.transform.location.x,
                           closer_lane_vehicle_3_waypoint.transform.location.y,
                           closer_lane_vehicle_3_waypoint.transform.location.z),
            closer_lane_vehicle_3_waypoint.transform.rotation)

        closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)
        self._closer_lane_vehicle_4_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_4_waypoint.transform.location.x,
                           closer_lane_vehicle_4_waypoint.transform.location.y,
                           closer_lane_vehicle_4_waypoint.transform.location.z),
            closer_lane_vehicle_4_waypoint.transform.rotation)

        closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)
        self._closer_lane_vehicle_5_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_5_waypoint.transform.location.x,
                           closer_lane_vehicle_5_waypoint.transform.location.y,
                           closer_lane_vehicle_5_waypoint.transform.location.z),
            closer_lane_vehicle_5_waypoint.transform.rotation)

        closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)
        self._closer_lane_vehicle_6_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_6_waypoint.transform.location.x,
                           closer_lane_vehicle_6_waypoint.transform.location.y,
                           closer_lane_vehicle_6_waypoint.transform.location.z),
            closer_lane_vehicle_6_waypoint.transform.rotation)

        further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)
        self._further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)

        further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)
        self._further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)

        further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)
        self._further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)

        further_lane_vehicle_4_transform = carla.Transform(
            carla.Location(further_lane_vehicle_4_waypoint.transform.location.x,
                           further_lane_vehicle_4_waypoint.transform.location.y,
                           further_lane_vehicle_4_waypoint.transform.location.z),
            further_lane_vehicle_4_waypoint.transform.rotation)
        self._further_lane_vehicle_4_transform = carla.Transform(
            carla.Location(further_lane_vehicle_4_waypoint.transform.location.x,
                           further_lane_vehicle_4_waypoint.transform.location.y,
                           further_lane_vehicle_4_waypoint.transform.location.z),
            further_lane_vehicle_4_waypoint.transform.rotation)

        further_lane_vehicle_5_transform = carla.Transform(
            carla.Location(further_lane_vehicle_5_waypoint.transform.location.x,
                           further_lane_vehicle_5_waypoint.transform.location.y,
                           further_lane_vehicle_5_waypoint.transform.location.z),
            further_lane_vehicle_5_waypoint.transform.rotation)
        self._further_lane_vehicle_5_transform = carla.Transform(
            carla.Location(further_lane_vehicle_5_waypoint.transform.location.x,
                           further_lane_vehicle_5_waypoint.transform.location.y,
                           further_lane_vehicle_5_waypoint.transform.location.z),
            further_lane_vehicle_5_waypoint.transform.rotation)

        closer_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_1_transform, color='255,0,0')
        closer_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_2_transform, color='255,0,0')
        closer_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_3_transform, color='255,0,0')
        closer_lane_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_4_transform, color='255,0,0')
        closer_lane_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_5_transform, color='255,0,0')
        closer_lane_vehicle_6 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_6_transform, color='255,0,0')

        further_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_1_transform, color='255,0,0')
        further_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_2_transform, color='255,0,0')
        further_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_3_transform, color='255,0,0')
        further_lane_vehicle_4 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_4_transform, color='255,0,0')
        further_lane_vehicle_5 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_5_transform, color='255,0,0')

        self._other_actor_transforms.append(self._closer_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_3_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_4_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_5_transform)
        self._other_actor_transforms.append(self._closer_lane_vehicle_6_transform)

        self._other_actor_transforms.append(self._further_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_3_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_4_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_5_transform)

        self.other_actors.append(closer_lane_vehicle_1)
        self.other_actors.append(closer_lane_vehicle_2)
        self.other_actors.append(closer_lane_vehicle_3)
        self.other_actors.append(closer_lane_vehicle_4)
        self.other_actors.append(closer_lane_vehicle_5)
        self.other_actors.append(closer_lane_vehicle_6)

        self.other_actors.append(further_lane_vehicle_1)
        self.other_actors.append(further_lane_vehicle_2)
        self.other_actors.append(further_lane_vehicle_3)
        self.other_actors.append(further_lane_vehicle_4)
        self.other_actors.append(further_lane_vehicle_5)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        lane_width = self._reference_waypoint.lane_width

        target_distance = 1000
        closer_lane_target_location = carla.Location(92.4, 20, 0)
        further_lane_target_location = carla.Location(88.4, 180, 0)

        target_speed = 10

        spawn_all = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SpawnAll")

        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[0], 
                                                             self._other_actor_transforms[0],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[1],
                                                             self._other_actor_transforms[1],
                                                             velocity=target_speed,
                                                             name='TransformSetterFurtherVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[3],
                                                             self._other_actor_transforms[3],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[4],
                                                             self._other_actor_transforms[4],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[5],
                                                             self._other_actor_transforms[5],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh3'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[6],
                                                             self._other_actor_transforms[6],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh4'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[7],
                                                             self._other_actor_transforms[7],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh5'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[8],
                                                             self._other_actor_transforms[8],
                                                             velocity=target_speed,
                                                             name='TransformSetterCloserVeh6'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[9],
                                                             self._other_actor_transforms[9],
                                                             velocity=target_speed,
                                                             name='TransformSetterFurtherVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[10],
                                                             self._other_actor_transforms[10],
                                                             velocity=target_speed,
                                                             name='TransformSetterFurtherVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[11],
                                                             self._other_actor_transforms[11],
                                                             velocity=target_speed,
                                                             name='TransformSetterFurtherVeh3'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[12],
                                                             self._other_actor_transforms[12],
                                                             velocity=target_speed,
                                                             name='TransformSetterFurtherVeh4'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[13],
                                                             self._other_actor_transforms[13],
                                                             velocity=target_speed,
                                                             name='TransformSetteFurtherVeh5'))

        # closer_lane_veh0_sequence
        closer_lane_veh0_sequence = py_trees.composites.Sequence()
        closer_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[0],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_start_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh0"))

        # further_lane_veh0_sequence
        further_lane_veh0_sequence = py_trees.composites.Sequence()
        further_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[1],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_start_waypoint,
                                                                      reference_speed=target_speed,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh0"))

        # side_walk_ped0_sequence
        ped0_location0 = self.other_actors[2].get_location()
        ped0_location1 = carla.Location(ped0_location0.x, ped0_location0.y - 16)
        ped0_location2 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 16)
        ped0_location3 = carla.Location(ped0_location0.x, ped0_location0.y - 2)
        ped0_location4 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 2)
        ped0_plan = None
        demo_id = random.choice([0, 1, 2])
        if demo_id == 0:
            ped0_plan = [ped0_location1]
        elif demo_id == 1:
            ped0_plan = [ped0_location1, ped0_location2]
        elif demo_id == 2:
            ped0_plan = [ped0_location3, ped0_location4]
        side_walk_ped0_sequence = py_trees.composites.Sequence()
        side_walk_ped0_sequence.add_child(BasicPedestrianBehavior(self.other_actors[2],
                                                                  target_speed=1,
                                                                  plan=ped0_plan,
                                                                  blackboard_queue_name=None,
                                                                  avoid_collision=False,
                                                                  name="BasicPedestrianBehavior"))
        # Send the pedestrian to far away and keep walking
        alien_transform = carla.Transform(carla.Location(ped0_location0.x,
                                                         -ped0_location0.y,
                                                         ped0_location0.z+1.0))
        side_walk_ped0_sequence.add_child(ActorTransformSetter(self.other_actors[2],
                                                               alien_transform,
                                                               False,
                                                               name="ActorTransformerPed0"))
        side_walk_ped0_sequence.add_child(KeepVelocity(self.other_actors[2],
                                                       target_velocity=1.0,
                                                       distance=1000,
                                                       name='KeepVelocityPed0'))

        # closer_lane_veh1_sequence
        closer_lane_veh1_sequence = py_trees.composites.Sequence()
        closer_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[3],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_1_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh1"))

        # closer_lane_veh2_sequence
        closer_lane_veh2_sequence = py_trees.composites.Sequence()
        closer_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[4],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_2_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh2"))

        # closer_lane_veh3_sequence
        closer_lane_veh3_sequence = py_trees.composites.Sequence()
        closer_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[5],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_3_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh3"))

        # closer_lane_veh4_sequence
        closer_lane_veh4_sequence = py_trees.composites.Sequence()
        closer_lane_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[6],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_4_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh4"))

        # closer_lane_veh5_sequence
        closer_lane_veh5_sequence = py_trees.composites.Sequence()
        closer_lane_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[7],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_5_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh5"))

        # closer_lane_veh6_sequence
        closer_lane_veh6_sequence = py_trees.composites.Sequence()
        closer_lane_veh6_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[8],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_6_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh6"))

        # further_lane_veh1_sequence
        further_lane_veh1_sequence = py_trees.composites.Sequence()
        further_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[9],
                                                                     target_distance,
                                                                     target_location=further_lane_target_location,
                                                                     start_waypoint=self.further_lane_vehicle_1_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="FurtherVeh1"))

        # further_lane_veh2_sequence
        further_lane_veh2_sequence = py_trees.composites.Sequence()
        further_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[10],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_2_waypoint,
                                                                      reference_speed=target_speed,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh2"))

        # further_lane_veh3_sequence
        further_lane_veh3_sequence = py_trees.composites.Sequence()
        further_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[11],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_3_waypoint,
                                                                      reference_speed=target_speed,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh3"))

        # further_lane_veh4_sequence
        further_lane_veh4_sequence = py_trees.composites.Sequence()
        further_lane_veh4_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[12],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_4_waypoint,
                                                                      reference_speed=target_speed,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh4"))

        # further_lane_veh5_sequence
        further_lane_veh5_sequence = py_trees.composites.Sequence()
        further_lane_veh5_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[13],
                                                                     target_distance,
                                                                     target_location=further_lane_target_location,
                                                                     start_waypoint=self.further_lane_vehicle_5_waypoint,
                                                                     reference_speed=target_speed,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="FurtherVeh5"))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 40.0, "DriveDistance")
        endcondition.add_child(endcondition_merged)

        # building the tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(spawn_all)
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0)) # -random.random()*3

        closer_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="CloserLane")
        closer_lane_parallel.add_child(closer_lane_veh0_sequence)
        closer_lane_parallel.add_child(closer_lane_veh1_sequence)
        closer_lane_parallel.add_child(closer_lane_veh2_sequence)
        closer_lane_parallel.add_child(closer_lane_veh3_sequence)
        closer_lane_parallel.add_child(closer_lane_veh4_sequence)
        closer_lane_parallel.add_child(closer_lane_veh5_sequence)
        closer_lane_parallel.add_child(closer_lane_veh6_sequence)

        further_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="FurtherLane")
        further_lane_parallel.add_child(further_lane_veh0_sequence)
        further_lane_parallel.add_child(further_lane_veh1_sequence)
        further_lane_parallel.add_child(further_lane_veh2_sequence)
        further_lane_parallel.add_child(further_lane_veh3_sequence)
        further_lane_parallel.add_child(further_lane_veh4_sequence)
        further_lane_parallel.add_child(further_lane_veh5_sequence)

        side_walk_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SideWalk")
        side_walk_parallel.add_child(side_walk_ped0_sequence)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")
        root.add_child(closer_lane_parallel)
        root.add_child(further_lane_parallel)
        root.add_child(side_walk_parallel)
        root.add_child(endcondition_merged)

        sequence.add_child(root)

        return sequence

    def _on_tick_callback(self, toto):
        # print('***************************************************************************************')
        # print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        # print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        # print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        # print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        # print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        # print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        # if self._debug_mode:
        #     print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous:
            delta_time = curr_time - self._time_previous
            # print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        # velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if self._velocity_queue.qsize() < self._velocity_filter_size:
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        # acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if self._acc_queue.qsize() < self._acc_filter_size:
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if acc_current > self._acc_max:
                    self._acc_max = acc_current
                if acc_current < self._acc_min:
                    self._acc_min = acc_current
                # self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if acc_current >= 0.0:
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter += 1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter += 1


        # jerk
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
                # self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter + jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter += 1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter + jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter += 1

        # angular velocity
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
            # angular acc
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
            # angular jerk
            if delta_time > 0.0:
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size:
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if abs(angular_jerk_current) > self._angular_jerk_max:
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        # store previous numbers
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
        self.remove_all_actors()


class CasperJunctionTurning6Demo(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a left turn. Scenario 4
    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 terminate_on_failure=True, timeout=50, report_enable=False, demo_id=-1):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._wmap = CarlaDataProvider.get_map()

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6
        # Number of attempts made so far
        self._spawn_attempted = 0
        # Need a report?
        # self._report_enable = report_enable

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial setting of traffic participants
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._ego_vehicle_transform = None

        self._cooperative_drivers = True
        self._dense_traffic = True

        self._cyclist_target_velocity = 10
        self._cyclist_transform = None
        self._other_actor_transforms = []

        self.closer_lane_start_waypoint = None
        self.further_lane_start_waypoint = None
        self.close_id0 = 0
        self.far_id0 = 100

        if demo_id==-1:
            self.demo_id = random.randint(0, 2)
        else:
            self.demo_id = demo_id       

        self._closer_lane_vehicle_1_location = 100

        self._closer_lane_vehicle_1_transform = None

        self.closer_lane_vehicle_1_waypoint = None

        self._further_lane_vehicle_1_location = -5 + random.random() * 10 + 15
        self._further_lane_vehicle_2_location = -5 + random.random() * 10 + 35
        self._further_lane_vehicle_3_location = -5 + random.random() * 10 + 55

        self._further_lane_vehicle_1_transform = None
        self._further_lane_vehicle_2_transform = None
        self._further_lane_vehicle_3_transform = None

        self.further_lane_vehicle_1_waypoint = None
        self.further_lane_vehicle_2_waypoint = None
        self.further_lane_vehicle_3_waypoint = None

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()

        super(CasperJunctionTurning6Demo, self).__init__("CasperJunctionTurning6Demo",
                                                ego_vehicles,
                                                config,
                                                world,
                                                debug_mode,
                                                terminate_on_failure=terminate_on_failure,
                                                criteria_enable=criteria_enable)

        # Need a report?
        self._report_enable = report_enable

        if self._report_enable:

            #Evaluation report
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

        # add actors from xml file
        for idx, actor in enumerate(config.other_actors):
            if 'vehicle' in actor.model:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, color='255,0,0')
            else:
                vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self._other_actor_transforms.append(actor.transform)
            self.other_actors.append(vehicle)

        closer_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[0].get_location())
        further_lane_start_waypoint = self._wmap.get_waypoint(self.other_actors[1].get_location())

        self.closer_lane_start_waypoint = closer_lane_start_waypoint
        self.further_lane_start_waypoint = further_lane_start_waypoint

        closer_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(closer_lane_start_waypoint, self._closer_lane_vehicle_1_location, junction=True)

        further_lane_vehicle_1_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_1_location, junction=True)
        further_lane_vehicle_2_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_2_location, junction=True)
        further_lane_vehicle_3_waypoint, _ = get_waypoint_in_distance(further_lane_start_waypoint, self._further_lane_vehicle_3_location, junction=True)

        self.closer_lane_vehicle_1_waypoint = closer_lane_vehicle_1_waypoint

        self.further_lane_vehicle_1_waypoint = further_lane_vehicle_1_waypoint
        self.further_lane_vehicle_2_waypoint = further_lane_vehicle_2_waypoint
        self.further_lane_vehicle_3_waypoint = further_lane_vehicle_3_waypoint

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._reference_waypoint.transform.location.x,
                           self._reference_waypoint.transform.location.y,
                           self._reference_waypoint.transform.location.z + 1),
            self._reference_waypoint.transform.rotation)

        closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)
        self._closer_lane_vehicle_1_transform = carla.Transform(
            carla.Location(closer_lane_vehicle_1_waypoint.transform.location.x,
                           closer_lane_vehicle_1_waypoint.transform.location.y,
                           closer_lane_vehicle_1_waypoint.transform.location.z),
            closer_lane_vehicle_1_waypoint.transform.rotation)

        further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)
        self._further_lane_vehicle_1_transform = carla.Transform(
            carla.Location(further_lane_vehicle_1_waypoint.transform.location.x,
                           further_lane_vehicle_1_waypoint.transform.location.y,
                           further_lane_vehicle_1_waypoint.transform.location.z),
            further_lane_vehicle_1_waypoint.transform.rotation)

        further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)
        self._further_lane_vehicle_2_transform = carla.Transform(
            carla.Location(further_lane_vehicle_2_waypoint.transform.location.x,
                           further_lane_vehicle_2_waypoint.transform.location.y,
                           further_lane_vehicle_2_waypoint.transform.location.z),
            further_lane_vehicle_2_waypoint.transform.rotation)

        further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)
        self._further_lane_vehicle_3_transform = carla.Transform(
            carla.Location(further_lane_vehicle_3_waypoint.transform.location.x,
                           further_lane_vehicle_3_waypoint.transform.location.y,
                           further_lane_vehicle_3_waypoint.transform.location.z),
            further_lane_vehicle_3_waypoint.transform.rotation)

        closer_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', closer_lane_vehicle_1_transform, color='255,0,0')

        further_lane_vehicle_1 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_1_transform, color='255,0,0')
        further_lane_vehicle_2 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_2_transform, color='255,0,0')
        further_lane_vehicle_3 = CarlaDataProvider.request_new_actor('vehicle.audi.etron', further_lane_vehicle_3_transform, color='255,0,0')

        self._other_actor_transforms.append(self._closer_lane_vehicle_1_transform)

        self._other_actor_transforms.append(self._further_lane_vehicle_1_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_2_transform)
        self._other_actor_transforms.append(self._further_lane_vehicle_3_transform)

        self.other_actors.append(closer_lane_vehicle_1)

        self.other_actors.append(further_lane_vehicle_1)
        self.other_actors.append(further_lane_vehicle_2)
        self.other_actors.append(further_lane_vehicle_3)

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter the in the trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        ego vehicle has to avoid the crash after a left turn, but
        continue driving after the road is clear.If this does not happen
        within 90 seconds, a timeout stops the scenario.
        """
        lane_width = self._reference_waypoint.lane_width

        target_distance = 1000
        closer_lane_target_location = carla.Location(92.4, 20, 0)
        further_lane_target_location = carla.Location(88.4, 180, 0)

        spawn_all = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SpawnAll")

        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[0], 
                                                             self._other_actor_transforms[0],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[1],
                                                             self._other_actor_transforms[1],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh0'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[3],
                                                             self._other_actor_transforms[3],
                                                             velocity=4,
                                                             name='TransformSetterCloserVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[4],
                                                             self._other_actor_transforms[4],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh1'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[5],
                                                             self._other_actor_transforms[5],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh2'))
        spawn_all.add_child(ActorTransformSetterWithVelocity(self.other_actors[6],
                                                             self._other_actor_transforms[6],
                                                             velocity=4,
                                                             name='TransformSetterFurtherVeh3'))

        # closer_lane_veh0_sequence
        closer_lane_veh0_sequence = py_trees.composites.Sequence()
        closer_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[0],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_start_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh0",
                                                                     demo_id=self.demo_id))

        # further_lane_veh0_sequence
        further_lane_veh0_sequence = py_trees.composites.Sequence()
        further_lane_veh0_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[1],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_start_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh0",
                                                                      demo_id=self.demo_id))

        # side_walk_ped0_sequence
        ped0_location0 = self.other_actors[2].get_location()
        ped0_location1 = carla.Location(ped0_location0.x, ped0_location0.y - 16)
        ped0_location2 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 16)
        ped0_location3 = carla.Location(ped0_location0.x, ped0_location0.y - 2)
        ped0_location4 = carla.Location(ped0_location0.x - 12, ped0_location0.y - 2)
        ped0_plan = None
        if self.demo_id == 0:
            ped0_plan = [ped0_location1]
        elif self.demo_id == 1:
            ped0_plan = [ped0_location1, ped0_location2]
        elif self.demo_id == 2:
            ped0_plan = [ped0_location3, ped0_location4]
        side_walk_ped0_sequence = py_trees.composites.Sequence()
        side_walk_ped0_sequence.add_child(BasicPedestrianBehavior(self.other_actors[2],
                                                                  target_speed=1,
                                                                  plan=ped0_plan,
                                                                  blackboard_queue_name=None,
                                                                  avoid_collision=False,
                                                                  name="BasicPedestrianBehavior"))
        # Send the pedestrian to far away and keep walking
        alien_transform = carla.Transform(carla.Location(ped0_location0.x,
                                                         -ped0_location0.y,
                                                         ped0_location0.z+1.0))
        side_walk_ped0_sequence.add_child(ActorTransformSetter(self.other_actors[2],
                                                               alien_transform,
                                                               False,
                                                               name="ActorTransformerPed0"))
        side_walk_ped0_sequence.add_child(KeepVelocity(self.other_actors[2],
                                                       target_velocity=1.0,
                                                       distance=1000,
                                                       name='KeepVelocityPed0'))

        # closer_lane_veh1_sequence
        closer_lane_veh1_sequence = py_trees.composites.Sequence()
        closer_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[3],
                                                                     target_distance,
                                                                     target_location=closer_lane_target_location,
                                                                     start_waypoint=self.closer_lane_vehicle_1_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="CloserVeh1",
                                                                     demo_id=self.demo_id))

        # further_lane_veh1_sequence
        further_lane_veh1_sequence = py_trees.composites.Sequence()
        further_lane_veh1_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[4],
                                                                     target_distance,
                                                                     target_location=further_lane_target_location,
                                                                     start_waypoint=self.further_lane_vehicle_1_waypoint,
                                                                     reference_speed=5,
                                                                     cooperative_driver=self._cooperative_drivers,
                                                                     dense_traffic=self._dense_traffic,
                                                                     name="FurtherVeh1",
                                                                     demo_id=self.demo_id))

        # further_lane_veh2_sequence
        further_lane_veh2_sequence = py_trees.composites.Sequence()
        further_lane_veh2_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[5],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_2_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh2",
                                                                     demo_id=self.demo_id))

        # further_lane_veh3_sequence
        further_lane_veh3_sequence = py_trees.composites.Sequence()
        further_lane_veh3_sequence.add_child(IDMDriveDistanceBehavior(self.other_actors[6],
                                                                      target_distance,
                                                                      target_location=further_lane_target_location,
                                                                      start_waypoint=self.further_lane_vehicle_3_waypoint,
                                                                      reference_speed=5,
                                                                      cooperative_driver=self._cooperative_drivers,
                                                                      dense_traffic=self._dense_traffic,
                                                                      name="FurtherVeh3",
                                                                     demo_id=self.demo_id))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        endcondition_merged = DriveDistance(self.ego_vehicles[0], 40.0, "DriveDistance")
        endcondition.add_child(endcondition_merged)

        # building the tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        sequence.add_child(spawn_all)
        sequence.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0)) # -random.random()*3

        closer_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="CloserLane")
        closer_lane_parallel.add_child(closer_lane_veh0_sequence)
        closer_lane_parallel.add_child(closer_lane_veh1_sequence)

        further_lane_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="FurtherLane")
        further_lane_parallel.add_child(further_lane_veh0_sequence)
        further_lane_parallel.add_child(further_lane_veh1_sequence)
        further_lane_parallel.add_child(further_lane_veh2_sequence)
        further_lane_parallel.add_child(further_lane_veh3_sequence)

        side_walk_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="SideWalk")
        side_walk_parallel.add_child(side_walk_ped0_sequence)

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="IntersectionLeftTurn")
        root.add_child(closer_lane_parallel)
        root.add_child(further_lane_parallel)
        root.add_child(side_walk_parallel)
        root.add_child(endcondition_merged)

        sequence.add_child(root)

        return sequence

    def _on_tick_callback(self, toto):
        # print('***************************************************************************************')
        # print('vehicle velocity x', self.ego_vehicles[0].get_velocity().x )
        # print('vehicle velocity y', self.ego_vehicles[0].get_velocity().y )
        # print('vehicle velocity z', self.ego_vehicles[0].get_velocity().z )
        # print('vehicle acceleration x', self.ego_vehicles[0].get_acceleration().x )
        # print('vehicle acceleration y', self.ego_vehicles[0].get_acceleration().y )
        # print('vehicle acceleration z', self.ego_vehicles[0].get_acceleration().z )
        curr_time = GameTime.get_time()
        self._elapsed_time = curr_time - self._start_time


        # if self._debug_mode:
        #     print('Elapsed_time: ', self._elapsed_time)

        delta_time = 0.0
        if self._time_previous:
            delta_time = curr_time - self._time_previous
            # print("-------------- Debug: dt = {}----------------".format(delta_time))

        acc_current = self._acc_previous
        jerk_current = 0.0

        # velocity
        if self.ego_vehicles[0]:
            velocity_current = math.sqrt(pow(self.ego_vehicles[0].get_velocity().x, 2) + pow(self.ego_vehicles[0].get_velocity().y, 2))
        else:
            velocity_current = 0
        self._velocity_queue.put(velocity_current)
        if self._velocity_queue.qsize() < self._velocity_filter_size:
            self._velocity_sum += velocity_current
        else:
            self._velocity_sum = self._velocity_sum + velocity_current - self._velocity_queue.get()

        velocity_current = self._velocity_sum / self._velocity_queue.qsize()

        # acceleration
        if self._it_counter > self._velocity_filter_size:
            if delta_time > 0.0:
                acc_current = (velocity_current - self._velocity_previous)/delta_time
                self._acc_queue.put(acc_current)
                if self._acc_queue.qsize() < self._acc_filter_size:
                    self._acc_sum += acc_current
                else:
                    self._acc_sum = self._acc_sum + acc_current - self._acc_queue.get()
                acc_current = self._acc_sum / self._acc_queue.qsize()

                if acc_current > self._acc_max:
                    self._acc_max = acc_current
                if acc_current < self._acc_min:
                    self._acc_min = acc_current
                # self._acc_ave = (self._acc_ave * self._it_counter + acc_current) / (self._it_counter + 1)
                if acc_current >= 0.0:
                    self._throttle_ave = (self._throttle_ave * self._throttle_counter +  acc_current) / (self._throttle_counter + 1)
                    self._throttle_counter += 1
                else:
                    self._brake_ave = (self._brake_ave * self._brake_counter +  acc_current) / (self._brake_counter + 1)
                    self._brake_counter += 1


        # jerk
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
                # self._jerk_ave = (self._jerk_ave * self._it_counter + jerk_current) / (self._it_counter + 1)
                if jerk_current >= 0:
                    self._throttle_jerk_ave = (self._throttle_jerk_ave * self._throttle_jerk_counter + jerk_current) / (self._throttle_jerk_counter + 1)
                    self._throttle_jerk_counter += 1
                else:
                    self._brake_jerk_ave = (self._brake_jerk_ave * self._brake_jerk_counter + jerk_current) / (self._brake_jerk_counter + 1)
                    self._brake_jerk_counter += 1

        # angular velocity
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
            # angular acc
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
            # angular jerk
            if delta_time > 0.0:
                angular_jerk_current = (angular_acc_current - self._angular_acc_previous)/delta_time
                self._angular_jerk_queue.put(angular_jerk_current)
                if self._angular_jerk_queue.qsize() < self._angular_jerk_filter_size:
                    self._angular_jerk_sum += angular_jerk_current
                else:
                    self._angular_jerk_sum = self._angular_jerk_sum + angular_jerk_current - self._angular_jerk_queue.get()
                angular_jerk_current = self._angular_jerk_sum / self._angular_jerk_queue.qsize()

                if abs(angular_jerk_current) > self._angular_jerk_max:
                    self._angular_jerk_max = abs(angular_jerk_current)

                self._angular_jerk_ave = (self._angular_jerk_ave * self._it_counter + abs(angular_jerk_current)) / (self._it_counter + 1)

        # print('delta_time', delta_time )
        # print('velocity_current', velocity_current )
        # print('velocity_previous', self._velocity_previous )
        # print('acc_current', acc_current )
        # print('acc_previous', self._acc_previous )
        # print('jerk_current', jerk_current )

        # store previous numbers
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

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.
        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """

        return AlwaysSuccessTrigger()

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
