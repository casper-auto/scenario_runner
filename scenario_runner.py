#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA scenario_runner

This is the main script to be executed when running a scenario.
It loads the scenario configuration, loads the scenario and manager,
and finally triggers the scenario execution.
"""

from __future__ import print_function

import glob
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import signal
import sys
import time
import json
import pkg_resources
import csv
import carla

from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# Version of scenario_runner
VERSION = '0.9.11'


class ScenarioRunner(object):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    scenario_runner = ScenarioRunner(args)
    scenario_runner.run()
    del scenario_runner
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    finished = False

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self._args = args

        if args.timeout:
            self.client_timeout = float(args.timeout)

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)
        CarlaDataProvider.set_client(self.client)

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.11'):
            raise ImportError("CARLA version 0.9.11 or newer required. CARLA version found: {}".format(dist))

        #Report
        self._counter = 0
        self._report_filename = None
        self._success_counter = 0.0
        self._accident_counter = 0.0
        self._timeout_counter = 0.0
        self._time_to_merge_sum = 0.0
        #self._acc_ave_sum = 0.0
        self._throttle_ave_sum = 0.0
        self._brake_ave_sum = 0.0
        self._acc_min_sum = 0.0
        self._acc_max_sum = 0.0
        #self._jerk_ave_sum = 0.0
        self._throttle_jerk_ave_sum = 0.0
        self._brake_jerk_ave_sum = 0.0
        self._jerk_min_sum = 0.0
        self._jerk_max_sum = 0.0
        self._angular_acc_ave_sum = 0.0
        #self._angular_acc_min_sum = 0.0
        self._angular_acc_max_sum = 0.0
        self._angular_jerk_ave_sum = 0.0
        #self._angular_jerk_min_sum = 0.0
        self._angular_jerk_max_sum = 0.0

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self._args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(self._args.debug, self._args.sync, self._args.timeout)

        # Create signal handler for SIGINT
        self._shutdown_requested = False
        if sys.platform != 'win32':
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()

    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """

        # Path of all scenario at "srunner/scenarios" folder + the path of the additional scenario argument
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))
        scenarios_list.append(self._args.additionalScenario)

        for scenario_file in scenarios_list:

            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                if scenario in member:
                    return member[1]

            # Remove unused Python paths
            sys.path.pop(0)

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        if self.finished:
            return

        self.finished = True

        # Simulation still running and in synchronous mode?
        if self.world is not None and self._args.sync:
            try:
                # Reset to asynchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                self.client.get_trafficmanager(int(self._args.trafficManagerPort)).set_synchronous_mode(False)
            except RuntimeError:
                sys.exit(-1)

        self.manager.cleanup()

        CarlaDataProvider.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self._args.waitForEgo and self.ego_vehicles[i] is not None and self.ego_vehicles[i].is_alive:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles):
        """
        Spawn or update the ego vehicles
        """

        if not self._args.waitForEgo:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             actor_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)
                CarlaDataProvider.register_actor(self.ego_vehicles[i])

        # sync state
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _analyze_scenario(self, config):
        """
        Provide feedback about success/failure of a scenario
        """

        # Create the filename
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        json_filename = None
        config_name = config.name
        if self._args.outputDir != '':
            config_name = os.path.join(self._args.outputDir, config_name)

        if self._args.junit:
            junit_filename = config_name + current_time + ".xml"
        if self._args.json:
            json_filename = config_name + current_time + ".json"
        filename = None
        if self._args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self._args.output, filename, junit_filename, json_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self._args.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _record_criteria(self, criteria, name):
        """
        Filter the JSON serializable attributes of the criterias and
        dumps them into a file. This will be used by the metrics manager,
        in case the user wants specific information about the criterias.
        """
        file_name = name[:-4] + ".json"

        # Filter the attributes that aren't JSON serializable
        with open('temp.json', 'w', encoding='utf-8') as fp:

            criteria_dict = {}
            for criterion in criteria:

                criterion_dict = criterion.__dict__
                criteria_dict[criterion.name] = {}

                for key in criterion_dict:
                    if key != "name":
                        try:
                            key_dict = {key: criterion_dict[key]}
                            json.dump(key_dict, fp, sort_keys=False, indent=4)
                            criteria_dict[criterion.name].update(key_dict)
                        except TypeError:
                            pass

        os.remove('temp.json')

        # Save the criteria dictionary into a .json file
        with open(file_name, 'w', encoding='utf-8') as fp:
            json.dump(criteria_dict, fp, sort_keys=False, indent=4)

    def _load_and_wait_for_world(self, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaDataProvider
        """

        if self._args.reloadWorld:
            self.world = self.client.load_world(town)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if self._args.waitForEgo:
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()

        if self._args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)
        CarlaDataProvider.set_world(self.world)

        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town and CarlaDataProvider.get_map().name != "OpenDriveMap":
            print("The CARLA server uses the wrong map: {}".format(CarlaDataProvider.get_map().name))
            print("This scenario requires to use map: {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, config):
        """
        Load and run the scenario given by config
        """
        self._counter += 1
        print("\nIteration #: ", self._counter)
        result = False
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            try:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                config.agent = self.agent_instance
            except Exception as e:          # pylint: disable=broad-except
                traceback.print_exc()
                print("Could not setup required agent due to {}".format(e))
                self._cleanup()
                return False

        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))
        tm = self.client.get_trafficmanager(int(self._args.trafficManagerPort))
        tm.set_random_device_seed(int(self._args.trafficManagerSeed))
        if self._args.sync:
            tm.set_synchronous_mode(True)

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            if self._args.openscenario:
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=self._args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(config.type)

                dense_traffic_flag = False
                cooperative_drivers_flag = False
                if self._args.dense == "True" or self._args.dense == "true":
                    dense_traffic_flag = True
                if self._args.cooperative == "True" or self._args.cooperative == "true":
                    cooperative_drivers_flag = True

                if "Casper" in config.name:
                    # self.ego_vehicles[0].set_simulate_physics(False)
                    scenario = scenario_class(self.world,
                                              self.ego_vehicles,
                                              config,
                                              self._args.randomize,
                                              self._args.debug,
                                              report_enable=self._args.report_enable)
                else:
                    scenario = scenario_class(self.world,
                                              self.ego_vehicles,
                                              config,
                                              self._args.randomize,
                                              self._args.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        try:
            if self._args.record:
                recorder_name = "{}/{}/{}.log".format(
                    os.getenv('SCENARIO_RUNNER_ROOT', "./"), self._args.record, config.name)
                self.client.start_recorder(recorder_name, True)

            # Load scenario and run it
            self.manager.load_scenario(scenario, self.agent_instance)
            self.manager.run_scenario()

            # Provide outputs if required
            self._analyze_scenario(config)

            # Update report
            if self._args.report_enable:
                self.update_report(self._args, config, scenario)

            # Remove all actors, stop the recorder and save all criterias (if needed)
            scenario.remove_all_actors()
            if self._args.record:
                self.client.stop_recorder()
                self._record_criteria(self.manager.scenario.get_criteria(), recorder_name)

            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result

    def _run_scenarios(self):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False

        # Load the scenario configurations provided in the config file
        scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(
            self._args.scenario,
            self._args.configFile)
        if not scenario_configurations:
            print("Configuration for scenario {} cannot be found!".format(self._args.scenario))
            return result

        # Execute each configuration
        for config in scenario_configurations:
            for _ in range(self._args.repetitions):
                result = self._load_and_run_scenario(config)

            self._cleanup()

        if self._args.report_enable:
            self.add_summary_to_report()

        return result

    def _run_route(self):
        """
        Run the route scenario
        """
        result = False

        if self._args.route:
            routes = self._args.route[0]
            scenario_file = self._args.route[1]
            single_route = None
            if len(self._args.route) > 2:
                single_route = self._args.route[2]

        # retrieve routes
        route_configurations = RouteParser.parse_routes_file(routes, scenario_file, single_route)

        for config in route_configurations:
            for _ in range(self._args.repetitions):
                result = self._load_and_run_scenario(config)

                self._cleanup()
        return result

    def _run_openscenario(self):
        """
        Run a scenario based on OpenSCENARIO
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(self._args.openscenario):
            print("File does not exist")
            self._cleanup()
            return False

        openscenario_params = {}
        if self._args.openscenarioparams is not None:
            for entry in self._args.openscenarioparams.split(','):
                [key, val] = [m.strip() for m in entry.split(':')]
                openscenario_params[key] = val
        config = OpenScenarioConfiguration(self._args.openscenario, self.client, openscenario_params)

        result = self._load_and_run_scenario(config)
        self._cleanup()
        return result

    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        if self._args.report_enable:
            self.create_report(self._args)
        result = True
        if self._args.openscenario:
            result = self._run_openscenario()
        elif self._args.route:
            result = self._run_route()
        else:
            result = self._run_scenarios()

        print("No more scenarios .... Exiting")
        return result

    def create_report(self, args):
        '''
        report
        '''
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        behavior = ""
        traffic_density = ""
        if args.dense == "True" or args.dense == "true":
            traffic_density = "_dense-traffic"
        else:
            traffic_density = "_sparse-traffic"

        if args.cooperative == "True" or args.cooperative == "true":
            behavior = "_cooperative-drivers"
        else:
            behavior = "_aggressive-drivers"

        # report_filename = current_time + ".csv"
        self._report_filename = current_time + traffic_density + behavior + ".csv"
        with open(self._report_filename, mode='w') as csv_file:
            #fieldnames = ['emp_name', 'dept', 'birth_month']
            fieldnames = ['N', 'Time', 'Successful', 'Accident', 'Timeout', 'Brake_Ave', 'Throttle_Ave',
                          'Acc_Min', 'Acc_Max', 'Brake_Jerk_Ave', 'Throttle_Jerk_Ave', 'Jerk_Min', 'Jerk_Max',
                          'Angular_Acc_Ave', 'Angular_Acc_Max', 'Angular_Jerk_Ave', 'Angular_Jerk_Max']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def update_report(self, args, config, scenario):
        '''
        report
        '''
        analyze_flag = self.manager.analyze_scenario(args.output, None, None, None)

        #Success, timeout, accident
        success_flag = 0
        accident_flag = 0
        time_out_flag = 0

        if analyze_flag == 0:
            success_flag = 1
            self._success_counter += 1
        elif analyze_flag == 1:
            time_out_flag = 1
            self._timeout_counter += 1
        elif analyze_flag == 2:
            accident_flag = 1
            self._accident_counter += 1

        #Time to merge
        time_to_merge = scenario._elapsed_time

        #statistic
        if success_flag == 1:
            self._time_to_merge_sum += time_to_merge
            #self._acc_ave_sum += scenario._acc_ave
            self._throttle_ave_sum += scenario._throttle_ave
            self._brake_ave_sum += scenario._brake_ave
            self._acc_min_sum += scenario._acc_min
            self._acc_max_sum += scenario._acc_max
            #self._jerk_ave_sum += scenario._jerk_ave
            self._brake_jerk_ave_sum += scenario._brake_jerk_ave
            self._throttle_jerk_ave_sum += scenario._throttle_jerk_ave
            self._jerk_min_sum += scenario._jerk_min
            self._jerk_max_sum += scenario._jerk_max
            self._angular_acc_ave_sum += scenario._angular_acc_ave
            #self._angular_acc_min_sum += scenario._angular_acc_min
            self._angular_acc_max_sum += scenario._angular_acc_max
            self._angular_jerk_ave_sum += scenario._angular_jerk_ave
            #self._angular_jerk_min_sum += scenario._angular_jerk_min
            self._angular_jerk_max_sum += scenario._angular_jerk_max

        report = [self._counter,
                  round(time_to_merge, 2),
                  success_flag,
                  accident_flag,
                  time_out_flag,
                  round(scenario._brake_ave, 2),
                  round(scenario._throttle_ave, 2),
                  round(scenario._acc_min, 2),
                  round(scenario._acc_max, 2),
                  round(scenario._brake_jerk_ave, 2),
                  round(scenario._throttle_jerk_ave, 2),
                  round(scenario._jerk_min, 2),
                  round(scenario._jerk_max, 2),
                  round(scenario._angular_acc_ave, 2),
                  round(scenario._angular_acc_max, 2),
                  round(scenario._angular_jerk_ave, 2),
                  round(scenario._angular_jerk_max, 2)]
        with open(self._report_filename, mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(report)

    def add_summary_to_report(self):
        '''
        report
        '''
        blank = []
        header = ['N', 'Successful Time', 'Successful %', 'Accident %', 'Timeout %',
                    'Brake_Ave', 'Throttle_Ave', 'Acc_Min', 'Acc_Max',
                    'Brake_Jerk_Ave', 'Throttle_Jerk_Ave', 'Jerk_Min', 'Jerk_Max',
                    'Angular_Acc_Ave', 'Angular_Acc_Max',
                    'Angular_Jerk_Ave', 'Angular_Jerk_Max']
        summary = []
        if self._success_counter > 0 and self._counter > 0:
            summary = [self._counter,
                       round(self._time_to_merge_sum/self._success_counter, 2),
                       round(self._success_counter/self._counter * 100, 2),
                       round(self._accident_counter/self._counter * 100, 2),
                       round(self._timeout_counter/self._counter * 100, 2),
                       round(self._brake_ave_sum/self._success_counter, 2),
                       round(self._throttle_ave_sum/self._success_counter, 2),
                       round(self._acc_min_sum/self._success_counter, 2),
                       round(self._acc_max_sum/self._success_counter, 2),
                       round(self._brake_jerk_ave_sum/self._success_counter, 2),
                       round(self._throttle_jerk_ave_sum/self._success_counter, 2),
                       round(self._jerk_min_sum/self._success_counter, 2),
                       round(self._jerk_max_sum/self._success_counter, 2),
                       round(self._angular_acc_ave_sum/self._success_counter, 2),
                       round(self._angular_acc_max_sum/self._success_counter, 2),
                       round(self._angular_jerk_ave_sum/self._success_counter, 2),
                       round(self._angular_jerk_max_sum/self._success_counter, 2)]
        elif self._success_counter == 0 and self._counter > 0:
            summary = [self._counter, '---', round(self._success_counter/self._counter * 100, 2),
                       round(self._accident_counter/self._counter * 100, 2), round(self._timeout_counter/self._counter * 100, 2),
                       '---', '---', '---', '---',
                       '---', '---', '---', '---',
                       '---', '---',
                       '---', '---']
        else:
            summary = [0, '---', 0.00,
                       0.00, 0.00,
                       '---', '---', '---', '---',
                       '---', '---', '---', '---',
                       '---', '---',
                       '---', '---']

        with open(self._report_filename, mode='a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(blank)
            writer.writerow(header)
            writer.writerow(summary)

def main():
    """
    main function
    """
    description = ("CARLA Scenario Runner: Setup, Run and Evaluate scenarios using CARLA\n"
                   "Current version: " + VERSION)

    # pylint: disable=line-too-long
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--host', default='127.0.0.1',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--timeout', default="10.0",
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--sync', action='store_true',
                        help='Forces the simulation to run synchronously')
    parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')

    parser.add_argument(
        '--scenario', help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
    parser.add_argument('--openscenarioparams', help='Overwrited for OpenSCENARIO ParameterDeclaration')
    parser.add_argument(
        '--route', help='Run a route as a scenario (input: (route_file,scenario_file,[route id]))', nargs='+', type=str)

    parser.add_argument(
        '--agent', help="Agent used to execute the scenario. Currently only compatible with route-based scenarios.")
    parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument('--output', action="store_true", help='Provide results on stdout')
    parser.add_argument('--file', action="store_true", help='Write results into a txt file')
    parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
    parser.add_argument('--json', action="store_true", help='Write results into a JSON file')
    parser.add_argument('--outputDir', default='', help='Directory for output files (default: this directory)')

    parser.add_argument('--configFile', default='', help='Provide an additional scenario configuration file (*.xml)')
    parser.add_argument('--additionalScenario', default='', help='Provide additional scenario implementations (*.py)')

    parser.add_argument('--debug', action="store_true", help='Run with debug output')
    parser.add_argument('--reloadWorld', action="store_true",
                        help='Reload the CARLA world before starting a scenario (default=True)')
    parser.add_argument('--record', type=str, default='',
                        help='Path were the files will be saved, relative to SCENARIO_RUNNER_ROOT.\nActivates the CARLA recording feature and saves to file all the criteria information.')
    parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
    parser.add_argument('--repetitions', default=1, type=int, help='Number of scenario executions')
    parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')

    parser.add_argument('--dense', default=1, help='dense traffic')
    parser.add_argument('--cooperative', default=0, help='cooperative drivers vs noncooperative')
    parser.add_argument('--report_enable', action="store_true", default=False, help='produce report')
    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if not arguments.scenario and not arguments.openscenario and not arguments.route:
        print("Please specify either a scenario or use the route mode\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.route and (arguments.openscenario or arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.openscenarioparams and not arguments.openscenario:
        print("WARN: Ignoring --openscenarioparams when --openscenario is not specified")

    if arguments.route:
        arguments.reloadWorld = True

    if arguments.agent:
        arguments.sync = True

    scenario_runner = None
    result = True
    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()
    except Exception:   # pylint: disable=broad-except
        traceback.print_exc()

    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
    return not result


if __name__ == "__main__":
    sys.exit(main())
