#!/usr/bin/env python

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
                                                                      IDMHighwayMergeAgentBehavior)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToLocationAlongRoute,
                                                                               InTriggerDistanceToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           generate_target_waypoint_in_route,
                                           get_location_in_distance,
                                           get_waypoint_in_distance)


def speed_ms_to_kmh(speed):
    return speed * 3.6


def distance2D(loc1, loc2):
    dx = loc1.x - loc2.x
    dy = loc1.y - loc2.y
    return math.sqrt(dx*dx + dy * dy)


class CasperHighwayMergeContinuous(BasicScenario):

    """
    This is a single ego vehicle scenario
    """
    category = "CasperHighwayMergeContinuous"

    # timeout = 20            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, debug_mode=False,
                 terminate_on_failure=False, criteria_enable=True, timeout=80,
                 dense_traffic=True, cooperative_drivers=False,
                 num_of_players=10, num_repetitions=1):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._dense_traffic = dense_traffic
        self._cooperative_drivers = cooperative_drivers
        self._speed_limit = 25


        print ("dense_traffic: ", dense_traffic)
        print ("cooperative_traffic: ", cooperative_drivers)

        ego_start_location = carla.Location(config.trigger_points[0].location.x,
                                            config.trigger_points[0].location.y,
                                            config.trigger_points[0].location.z)
        self._ego_start_waypoint = self._map.get_waypoint(ego_start_location)

        other_vehicles_location = carla.Location(16.4,
                                            200,
                                            config.trigger_points[0].location.z)
        self._other_vehicles_init_wp = self._map.get_waypoint(other_vehicles_location)

        self._veh_tf_actor_pool = carla.Transform(
            carla.Location(self._other_vehicles_init_wp.transform.location.x,
                           self._other_vehicles_init_wp.transform.location.y,
                           self._other_vehicles_init_wp.transform.location.z - 500), self._other_vehicles_init_wp.transform.rotation)

        self._veh_tf_behavior = carla.Transform(
            carla.Location(self._other_vehicles_init_wp.transform.location.x,
                           self._other_vehicles_init_wp.transform.location.y,
                           self._other_vehicles_init_wp.transform.location.z + 1), self._other_vehicles_init_wp.transform.rotation)

        # Target location for other car based on the current map
        self._target_location_others = carla.Location(15.1, -117.7, 1)

        self._on_tick_ref = 0

        # terminate_on_failure
        self.terminate_on_failure = terminate_on_failure

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(CasperHighwayMergeContinuous, self).__init__("CasperHighwayMergeContinuous",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    terminate_on_failure=terminate_on_failure,
                                                    criteria_enable=criteria_enable,
                                                    num_repetitions=num_repetitions,
                                                    num_of_players=num_of_players)

    def _initialize_ego_vehicle(self):
        """
        Custom initialization
        """

        # ego vehicle transform
        self._ego_vehicle_transform = carla.Transform(
            carla.Location(self._ego_start_waypoint.transform.location.x,
                           self._ego_start_waypoint.transform.location.y,
                           self._ego_start_waypoint.transform.location.z + 1),
            self._ego_start_waypoint.transform.rotation)

        # keep frozon before the behavior tree starts
        self.ego_vehicles[0].set_simulate_physics(False)

    def _update_actors(self, full_tree, index_all, first_time = False):

        """
        This scenario keeps track of a queue of up to "num_of_players" cars. It stochastically
        checks based on the spawn rate, if we should spawn a new car. If positive, we will do
        a quick check so that the car does not spawn on top of the previous car.

        This function creates the py_tree necessary to spawn and keep track of all the vehicles
        so that the can be controlled based on the atomic behavior class it is defined on.

        This function will be called continuously to and update the tree for each call if necessary.
        """

        new_spawned_car = False
        veh_actor_new = []

        # SPAWNING
        prob_spawn = 0.1
        min_traffic_dist = 10
        if (self._dense_traffic):
            default_s0 = 0.875
            default_headway = 0.875
        else:
            default_s0 = 1.75
            default_headway = 1.75

        spawn_car_val = random.uniform(0.0, 1.0)
        valid_spawn = False
        speed_veh = random.uniform(0.7 * self._speed_limit, 0.9 * self._speed_limit)
        if spawn_car_val > 1 - prob_spawn:
            if self.other_actors:
                # random initial speeds
                min_headway_dist = default_s0 + speed_veh * default_headway
                min_dist_spawn = max(min_headway_dist, min_traffic_dist)

                # Check distance with car in front
                loc_front = self.other_actors[-1].get_location()
                dist_front = distance2D(loc_front, self._other_vehicles_init_wp.transform.location)

                valid_spawn = dist_front > min_dist_spawn
            else:
                valid_spawn = True

        if valid_spawn or first_time:
            veh_actor_new = CarlaDataProvider.request_new_actor('vehicle.nissan.micra', self._veh_tf_actor_pool, color='255,0,0')
            physics_control = veh_actor_new.get_physics_control()
            physics_control.mass = physics_control.mass / 2
            veh_actor_new.apply_physics_control(physics_control)

            if veh_actor_new:
                self.other_actors.append(veh_actor_new)
                new_spawned_car = True
                # print("Creating new actor")
                # Register on the CARLA buffer
                CarlaDataProvider.register_actor(veh_actor_new)
            else:
                print("Could not create a new actor")


        # BEHAVIOR

        # Build behavior for other vehicles. For each vehicle we will have a sequence that is creation and then driving

        if new_spawned_car:
            name_veh_seq = "Sequence Behavior " + str(veh_actor_new.id)
            name_veh_drive = "Merging Lane " + str(veh_actor_new.id)
            sequence_veh = py_trees.composites.Sequence(name_veh_seq)

            merging_lane_veh = py_trees.composites.Parallel(name_veh_drive,
                        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            merging_lane_veh.add_child(IDMHighwayMergeAgentBehavior(veh_actor_new, self._target_location_others, self._other_vehicles_init_wp, cooperative_driver=self._cooperative_drivers, dense_traffic = self._dense_traffic, speed_limit = self._speed_limit))

            # Sequence: First creation and then driving behavior
            sequence_veh.add_child(ActorTransformSetterWithVelocity(veh_actor_new, self._veh_tf_behavior, speed_veh))
            sequence_veh.add_child(merging_lane_veh)

            # Add new child for each spawned car
            full_tree.children[index_all].add_child(sequence_veh)

            # Print the tree
            # print("Tree printing after adding agent")
            # py_trees.display.print_ascii_tree(full_tree, show_status=True)

        agent_del = []
        seq_remove = []
        # Clearing agents that have reached their destination
        for veh_seq in full_tree.children[index_all].children:
            status_veh = veh_seq.status

            if status_veh == py_trees.common.Status.FAILURE:
                print("It should never get to failure status for other cars")

            if status_veh == py_trees.common.Status.SUCCESS:
                # ActorTransformSetting is the first node
                actor_veh = veh_seq.children[0]._actor
                agent_del.append(actor_veh)

                # Remove tree child from node
                seq_remove.append(veh_seq)

        if len(agent_del) != len(seq_remove):
            print("ERROR. We should be removing same amount of agents and tree nodes")
            exit(0)

        if len(agent_del) > 1:
            print("ERROR. We should be removing only 1 agent at a time")
            py_trees.display.print_ascii_tree(full_tree, show_status=True)
            for agents in agents_del:
                loc_del = agent.get_location()
                print("position delete: [%f, %f]" % (loc_del.x, loc_del.y))
            exit(0)

        if agent_del:
            loc_del = agent_del[0].get_location()
            if CarlaDataProvider.actor_id_exists(agent_del[0].id):
                CarlaDataProvider.remove_actor_by_id(agent_del[0].id)

            if self.other_actors[0].id != agent_del[0].id:
                print("Why is the car that needs to be removed not the first on the queue")
                print("We should be deleting agent: %d but we first in queue is: %d" % (agent_del[0].id, self.other_actors[0].id))
                print("position %f: [%f, %f]" % (agent_del[0].id, loc_del.x, loc_del.y))
                loc_queue = self.other_actors[0].get_location()
                print("position %d: [%f, %f]" % (self.other_actors[0].id, loc_queue.x, loc_queue.y))
                py_trees.display.print_ascii_tree(full_tree, show_status=True)
            else:
                self.other_actors.popleft()

        # Remove sequences that are obsolete
        if seq_remove:
            full_tree.children[index_all].remove_child(seq_remove[0])

            # Print the tree
            # print("Tree printing after removing agent")
            # py_trees.display.print_ascii_tree(full_tree, show_status=True)

        return full_tree


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
                return InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0], ego_vehicle_route, start_location, 5)

            return InTriggerDistanceToLocation(self.ego_vehicles[0],
                                               start_location, 100.0)

        return None

    def _create_behavior_ego(self):
        """
        Behaviors for ego vehicle. End condition is based on driven distance.
        """
        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy. SUCCESS_ON_ALL)
        endcondition_merged = DriveDistance(self.ego_vehicles[0], 120.0, "DriveDistance" )
        endcondition.add_child(endcondition_merged)

        # Build ego behavior driving tree(creation + end_condition)
        behavior = py_trees.composites.Sequence("Sequence Ego Driving")
        behavior.add_child(ActorTransformSetterWithVelocity(self.ego_vehicles[0], self._ego_vehicle_transform, 0.25 * self._speed_limit))
        drive_ego = py_trees.composites.Parallel("Ego driving", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        drive_ego.add_child(endcondition_merged)
        behavior.add_child(drive_ego)

        return behavior

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
        if self._on_tick_ref != None:
            self._world.remove_on_tick(self._on_tick_ref)
        print('========================== ')

    def remove_on_tick(self):
        """
        Remove on_tick
        """
        if self._on_tick_ref != None:
            self._world.remove_on_tick(self._on_tick_ref)
