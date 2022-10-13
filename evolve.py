"""
Main evolution driver for Geneva (GENetic EVAsion). This file performs the genetic algorithm,
and relies on the evaluator (evaluator.py) to provide fitness evaluations of each individual.
"""
import pandas as pd
import argparse
import copy
import logging
import operator
import os
import random
import subprocess as sp
import sys
import math

import actions.strategy
import actions.tree
import actions.trigger
import evaluator
import layers.packet

# Grab the terminal size for printing
try:
    _, COLUMNS = sp.check_output(['stty', 'size']).decode().split()
# If pytest has capturing enabled or this is run without a tty, catch the exception
except sp.CalledProcessError:
    _, COLUMNS = 0, 0


def setup_logger(log_level):
    """
    Sets up the logger. This will log at the specified level to "ga.log" and at debug level to "ga_debug.log".
    Logs are stored in the trials/ directory under a run-specific folder.
    Example: trials/2020-01-01_01:00:00/logs/ga.log

    Args:
        log_level (str): Log level to use in setting up the logger ("debug")
    """
    level = log_level.upper()
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "Unknown log level %s" % level
    actions.utils.CONSOLE_LOG_LEVEL = level.lower()

    # Setup needed folders
    ga_log_dir = actions.utils.setup_dirs(actions.utils.RUN_DIRECTORY)

    ga_log = os.path.join(ga_log_dir, "ga.log")
    ga_debug_log = os.path.join(ga_log_dir, "ga_debug.log")

    # Configure logging globally
    #formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    #logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s:%(message)s')
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s')

    # Set up the root logger
    logger = logging.getLogger("ga_%s" % actions.utils.RUN_DIRECTORY)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    setattr(logger, "ga_log_dir", ga_log_dir)

    # If this logger's handlers have already been set up, don't add them again
    if logger.handlers:
        return logger

    # Set log level of console
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Add a DEBUG file handler to send all the debug output to a file
    debug_file_handler = logging.FileHandler(ga_debug_log)
    debug_file_handler.setFormatter(formatter)
    debug_file_handler.setLevel(logging.DEBUG)
    logger.addHandler(debug_file_handler)

    # Add a file handler to send all the output to a file
    file_handler = logging.FileHandler(ga_log)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def collect_plugin_args(cmd, plugin, plugin_type, message=None):
    """
    Collects and prints arguments for a given plugin.

    Args:
        cmd (list): sys.argv or a list of args to parse
        plugin (str): Name of plugin to import ("http")
        plugin_type (str): Component of plugin to import ("client")
        message (str): message to override for printing
    """
    if not message:
        message = plugin_type
    try:
        _, cls = actions.utils.import_plugin(plugin, plugin_type)
        print("\n\n")
        print("=" * int(COLUMNS))
        print("Options for --test-type %s %s" % (plugin, message))
        cls.get_args(cmd)
    # Catch SystemExit here, as this is what argparse raises when --help is passed
    except (SystemExit, Exception):
        pass


def get_args(cmd):
    """
    Sets up argparse and collects arguments.

    Args:
        cmd (list): sys.argv or a list of args to parse

    Returns:
        namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Genetic algorithm for evolving censorship evasion.\n\nevolve.py uses a pass-through argument system to pass the command line arguments through different files in the system, including the evaluator (evaluator.py) and a given plugin (plugins/). --help will collect all these arguments.', add_help=False, prog="evolve.py")

    parser.add_argument('--test-type', action='store', choices=actions.utils.get_plugins(), default="http", help="plugin to launch")

    # Add help message separately so we can collect the help messages of all of the other parsers
    parser.add_argument('-h', '--help', action='store_true', default=False, help='print this help message and exit')

    # Control aspects of individuals
    ind_group = parser.add_argument_group('control aspects of individual strategies')
    ind_group.add_argument('--in-trees', action='store', type=int, default=0, help='starting # of input-direction action trees per strategy. Disables inbound forest if set to 0')
    ind_group.add_argument('--out-trees', action='store', type=int, default=1, help='starting # of output-direction action trees per strategy')
    ind_group.add_argument('--in-actions', action='store', type=int, default=2, help='starting # of input-direction actions per action tree')
    ind_group.add_argument('--out-actions', action='store', type=int, default=2, help='starting # of output-direction actions per action tree')
    ind_group.add_argument('--fix-trigger', action='store', help='fix all triggers for this evolution to a given trigger')

    # Options to control the population pool
    pop_group = parser.add_argument_group('control aspects of the population pool')
    pop_group.add_argument('--load-from', action='store', help="Load population from a generation file")
    pop_group.add_argument('--seed', action='store', help='seed strategy to initialize the population to.')

    # Options to evaluate and exit, skip evaluation, and to specify the type of test
    evaluation_group = parser.add_argument_group('control aspects of strategy evaluation')
    evaluation_group.add_argument('--eval-only', action='store', default=None, help='only evaluate fitness for a given strategy or file of strategies')
    evaluation_group.add_argument('--no-eval', action='store_true', help="Disable evaluator for debugging")
    evaluation_group.add_argument('--runs', action='store', type=int, default=1, help='number of times each strategy should be run for one evaluation (default 1, fitness is averaged).')
    evaluation_group.add_argument('--snort-alert-checkpoint',action='store', type=int, default=0, help='mark the end of the last run log, to quickly locate this time log')
    evaluation_group.add_argument('--suricata-alert-checkpoint',action='store', type=int, default=0, help='mark the end of the last run log, to quickly locate this time log')
    evaluation_group.add_argument('--snort-console-checkpoint',action='store', type=int, default=0, help='mark the end of the last run log, to quickly locate this time log')
    evaluation_group.add_argument('--suricata-console-checkpoint',action='store', type=int, default=0, help='mark the end of the last run log, to quickly locate this time log')

    # Hyperparameters for genetic algorithm
    genetic_group = parser.add_argument_group('control aspects of the genetic algorithm')
    genetic_group.add_argument('--elite-clones', action='store', type=int, default=3, help='number copies of the highest performing individual that should be propagated to the next generation.')
    genetic_group.add_argument('--mutation-pb', action='store', type=float, default=0.99, help='mutation probability')
    genetic_group.add_argument('--crossover-pb', action='store', type=float, default=0.4, help='crossover probability')
    genetic_group.add_argument('--allowed-retries', action='store', type=int, default=20, help='maximum number of times GA will generate any given individual')
    genetic_group.add_argument('--generations', type=int, action='store', default=50, help="number of generations to run for.")
    genetic_group.add_argument('--population', type=int, action='store', default=250, help="size of population.")
    genetic_group.add_argument('--no-reject-empty', action='store_true', default=False, help="disable mutation rejection of empty strategies")
    genetic_group.add_argument('--no-canary', action='store_true', help="disable canary phase")

    # Limit access to certain protocols, fields, actions, or types of individuals
    filter_group = parser.add_argument_group('limit access to certain protocols, fields, actions, or types of individuals')
    filter_group.add_argument('--protos', action="store", default="TCP", help="allow the GA to scope only to these protocols")
    filter_group.add_argument('--fields', action='store', default="", help='restrict the GA to only seeing given fields')
    filter_group.add_argument('--disable-fields', action='store', default="", help='restrict the GA to never using given fields')
    filter_group.add_argument('--no-gas', action="store_true", help="disables trigger gas")
    filter_group.add_argument('--disable-action', action='store', default="sleep,trace", help='disables specific actions')

    # Logging
    logging_group = parser.add_argument_group('control logging')
    logging_group.add_argument('--log', action='store', default="info", choices=("debug", "info", "warning", "critical", "error"), help="Sets the log level")
    logging_group.add_argument('--no-print-hall', action='store_true', help="does not print hall of fame at the end")
    logging_group.add_argument('--graph-trees', action='store_true', default=False, help='graph trees in addition to outputting to screen')

    # Misc
    usage_group = parser.add_argument_group('misc usage')
    usage_group.add_argument('--no-lock-file', default=(os.name == "posix"), action='store_true', help="does not use /lock_file.txt")
    usage_group.add_argument('--force-cleanup', action='store_true', default=False, help='cleans up all docker containers and networks after evolution')

    if not cmd:
        parser.error("No arguments specified")

    args, _ = parser.parse_known_args(cmd)

    epilog = "See the README.md for usage."
    # Override the help message to collect the pass through args
    if args.help:
        parser.print_help()
        print(epilog)
        print("=" * int(COLUMNS))
        print("\nevolve.py uses a pass-through argument system to evaluator.py and other parts of Geneva. These arguments are below.\n\n")
        evaluator.get_arg_parser(cmd).print_help()
        if args.test_type:
            collect_plugin_args(cmd, args.test_type, "plugin", message="parent plugin")
            collect_plugin_args(cmd, args.test_type, "client")
            collect_plugin_args(cmd, args.test_type, "server")
        raise SystemExit
    return args


def fitness_function(logger, population, ga_evaluator):
    """
    Calls the evaluator to evaluate a given population of strategies.
    Sets the .fitness attribute of each individual.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        population (list): List of individuals to evaluate  
        ga_evaluator (:obj:`evaluator.Evaluator`): An evaluator object to evaluate with

    Returns:
        list: Population post-evaluation

    """
    if ga_evaluator:
        #分开计算
        offspring_length=len(population)
        return_population=[]
        divide_num=500
        if offspring_length <= divide_num:
            return_population,whether_success=ga_evaluator.evaluate(population,False)
        else:
            count_num=-1
            while offspring_length > divide_num:
                count_num+=1
                return_population_part,whether_success=ga_evaluator.evaluate(population[divide_num*count_num:divide_num*(count_num+1)], False)
                logger.info("need to divide, current slide:%d",count_num)
                return_population=return_population+return_population_part
                offspring_length -=divide_num
            return_population_part,whether_success=ga_evaluator.evaluate(population[divide_num*(count_num+1):], False)
            return_population=return_population+return_population_part
        while whether_success == False: # bad thing happened, need to do it again
            logger.info("do ga_evaluator.evaluate again")
            # kill fd
            for i in range(5,1000):
                try:
                    os.close(i)
                except:
                    pass
            return_population,whether_success=ga_evaluator.evaluate(population,False)

        #return ga_evaluator.evaluate(population)
        return return_population

    for ind in population:
        ind.fitness = 0
        logger.info("[%s] Fitness %d: %s", -1, ind.fitness, str(ind))

    return population


def sel_random(individuals, k):
    """
    Implementation credit to DEAP: https://github.com/DEAP/deap
    Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    Args:
        individuals (list): A list of individuals to select from.
        k (int): The number of individuals to select.

    Returns:
        list: A list of selected individuals.
    """
    return [random.choice(individuals) for _ in range(k)]


def selection_tournament(individuals, k, tournsize, fit_attr="fitness"):
    """
    Implementation credit to DEAP: https://github.com/DEAP/deap
    Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    Args:
        individuals (list): A list of individuals to select from.
        k (int): The number of individuals to select.
        tournsize (int): The number of individuals participating in each tournament.
        fit_attr: The attribute of individuals to use as selection criterion (defaults to "fitness")

    Returns:
        list: A list of selected individuals.
    """
    chosen = []
    for _ in range(k):
        aspirants = sel_random(individuals, tournsize)
        chosen.append(copy.deepcopy(max(aspirants, key=operator.attrgetter(fit_attr))))
    return chosen

def selection_next_generation(logger,individuals, parents):
    """
    Select individual from the last generation based on states change
    Core thought: if one packet in the packets sequence can cause DPI0 state change and don't cause DPI1 state change. That's a good sign
    

    Args:
        individuals (list): A list of individuals to select from.
        parents (list): A list of individuals' parents
        k (int): The number of individuals to select.
        tournsize (int): The number of individuals participating in each tournament.
        fit_attr: The attribute of individuals to use as selection criterion (defaults to "fitness")

    Returns:
        list: A list of selected individuals.
    """
    chosen = []
    canary_round=False
    for i in range(len(parents)):
        individuals[i].father_inconsistent_packet_num=parents[i].inconsistent_packet_num
        father_strategy=parents[i]
        if father_strategy.environment_id=='canary':
            canary_round=True
            break
        child_strategy=individuals[i]
        decide_append_child=False
        decide_append_father=False
        decide_great_add_child=False
        if child_strategy.terminate_by_server == True:
            # [3.14] for testing we let go this function
            # logger.info('Terminate by server, maybe it is a bad strategy,we do not add it')
            # continue
            pass
        if father_strategy.inconsistent_packet_num==0 and child_strategy.inconsistent_packet_num!=0:
            #chosen.append(copy.deepcopy(child_strategy))
            decide_append_child=True
            logger.info('F:%d,C:%d. Add child %s.',0,child_strategy.inconsistent_packet_num,str(child_strategy))
        # father ok child bad
        if father_strategy.inconsistent_packet_num !=0 and child_strategy.inconsistent_packet_num==0:
            #chosen.append(copy.deepcopy(father_strategy))
            decide_append_father=True
            logger.info('F:%d,C:%d. Add father %s.',father_strategy.inconsistent_packet_num,0,str(father_strategy))
        # father and child both ok
        if father_strategy.inconsistent_packet_num !=0 and child_strategy.inconsistent_packet_num!=0:
            #chosen.append(copy.deepcopy(child_strategy))
            #chosen.append(copy.deepcopy(father_strategy))
            decide_append_child=True
            decide_append_child=True
            logger.info('F:%d,C:%d. Add father %s. child %s',father_strategy.inconsistent_packet_num,child_strategy.inconsistent_packet_num,str(father_strategy),str(child_strategy))
        # father and child both bad
        if father_strategy.inconsistent_packet_num ==0 and child_strategy.inconsistent_packet_num==0:
            #chosen.append(copy.deepcopy(father_strategy))
            decide_append_father=True
            logger.info('F:%d,C:%d. Add father %s.',0,0,str(father_strategy))

        # extra appending 
        if child_strategy.inconsistent_packet_num!=0:   
            # means it is compareable
            for i in child_strategy.change_happened_packet_num:
                print('result_snort_client:',child_strategy.result_snort)
                #print('result_snort_server:',child_strategy.result_snort)
                print('result_snort_suricata:',child_strategy.result_suricata)
                print('child_strategy.change_happened_packet_num:',child_strategy.change_happened_packet_num)
                print('child_strategy.inconsistent_packet_num:',child_strategy.inconsistent_packet_num)
                print('len of child_strategy.result_snort:',len(child_strategy.result_snort))
                print('len of child_strategy.result_suricata:',len(child_strategy.result_suricata))
                try:
                    end_state_snort_client=child_strategy.result_snort[i].client_end_stage
                    end_state_snort_server=child_strategy.result_snort[i].server_end_stage
                    end_state_suricata=child_strategy.result_suricata[i].end_stage
                # make compare
              
                #if (end_state_snort_client=='ESTABLISHED' or end_state_snort_server =='ESTABLISHED') and end_state_suricata !='established':
                #    flag=True
                #    logger.info('INTERESTING HIT state_snort_client:%s state_snort_server:%s VS state_suricata:%s',end_state_snort_client,end_state_snort_server,end_state_suricata)
                #    break
                    if (end_state_snort_client!='ESTABLISHED' and end_state_snort_server !='ESTABLISHED') and end_state_suricata =='established':
                        decide_great_add_child=True
                        logger.info('INTERESTING HIT state_snort_client:%s state_snort_server:%s VS state_suricata:%s',end_state_snort_client,end_state_snort_server,end_state_suricata)
                        break
                except:
                    print('Crash!')
                    print('i=',str(i))
                    print('change_happened_packet_num:',child_strategy.change_happened_packet_num)
                    print(child_strategy.result_snort[i],'snort_len:',len(child_strategy.result_snort[i]))
                    print(child_strategy.result_suricata[i],'suricata_len',len(child_strategy.result_suricata[i]))
                    raise



        if decide_append_child==True:
            chosen.append(copy.deepcopy(child_strategy))
            
        if decide_append_father==True:
            chosen.append(copy.deepcopy(father_strategy))

        if decide_great_add_child==True:
            logger.info('Five time added %s',str(child_strategy))
            chosen.append(copy.deepcopy(child_strategy))
            chosen.append(copy.deepcopy(child_strategy))
            chosen.append(copy.deepcopy(child_strategy))
            chosen.append(copy.deepcopy(child_strategy))
            chosen.append(copy.deepcopy(child_strategy))
    if canary_round == True:
        return individuals            
            


    return chosen

def selection_next_generation_vertical_overall(logger,individuals, parents, gen):
    """
    Select individual from the last generation based on states change
    Core thought: if one packet in the packets sequence can cause DPI0 state change and don't cause DPI1 state change. That's a good sign
    ***vertical-overall-version***

    Args:
        individuals (list): A list of individuals to select from.
        parents (list): A list of individuals' parents
        gen (int): 当前的轮次
        k (int): The number of individuals to select.
        tournsize (int): The number of individuals participating in each tournament.
        fit_attr: The attribute of individuals to use as selection criterion (defaults to "fitness")

    Returns:
        list: A list of selected individuals.
    """

    chosen = []
    father_chosen_hash = []
    canary_round=False
    for i in range(len(parents)):
        individuals[i].father_inconsistent_packet_num=parents[i].inconsistent_packet_num
        father_strategy=parents[i]
        if father_strategy.environment_id=='canary':
            canary_round=True
            break
        child_strategy=individuals[i]  
        child_strategy.first_added_time = gen
        
        child_strategy.father_suricata_state_change_overall=father_strategy.suricata_state_change_overall
        child_strategy.father_snort_state_change_overall_client=father_strategy.snort_state_change_overall_client
        child_strategy.father_snort_state_change_overall_server=father_strategy.snort_state_change_overall_server
        
        decide_append_child=False
        decide_append_father=False
        decide_great_add_child=False
        snort_overall_change = False
        suricata_overall_change = False
        child_add_more = False
        father_add_more = False
        # father_strartgy vs child_strategy change/not change?
        if father_strategy.snort_state_change_overall_client !=child_strategy.snort_state_change_overall_client or father_strategy.snort_state_change_overall_server !=child_strategy.snort_state_change_overall_server:
            snort_overall_change=True
        else:
            snort_overall_change=False
        if father_strategy.suricata_state_change_overall !=child_strategy.suricata_state_change_overall:
            suricata_overall_change=True
        else:
            suricata_overall_change=False
        
        
        if child_strategy.suricata_state_change_overall == [] or child_strategy.snort_state_change_overall_client == [] or child_strategy.snort_state_change_overall_server == []:
            suricata_overall_change=False
            snort_overall_change=False
        minor_add_flag=False 
        
        if suricata_overall_change or snort_overall_change: 
            parent_string=str(father_strategy.snort_state_change_overall_client)+str(father_strategy.snort_state_change_overall_server)+str(father_strategy.suricata_state_change_overall)
            child_string=str(child_strategy.snort_state_change_overall_client)+str(child_strategy.snort_state_change_overall_server)+str(child_strategy.suricata_state_change_overall)
            hash_pc=hash(parent_string+child_string)
            hash_cp=hash(child_string+parent_string)
            str_hash_father=str(father_strategy)
            str_hash_child=str(child_strategy)
            
            state_change_happened_before=False
            if (hash_pc in father_strategy.before_hash) or (hash_cp in child_strategy.before_hash) or (hash_pc in child_strategy.before_hash) or (hash_cp in father_strategy.before_hash):
                logger.info('hash state happen before! doing hash name compare') 
                state_change_happend_before=True

                if str_hash_father in child_strategy.before_hash_name or str_hash_child in father_strategy.before_hash_name:
                    logger.info('hash name compare fail! find same strategy! We do not add')
                    snort_overall_change = False
                    suricata_overall_change = False
                else:
                    minor_add_flag = True
                    logger.info('same state change but different strategy name! minor_add')
                    father_strategy.before_hash.append(hash_pc)
                    father_strategy.before_hash.append(hash_cp)
                    child_strategy.before_hash.append(hash_pc)
                    child_strategy.before_hash.append(hash_cp)
                    father_strategy.before_hash_name.append(str_hash_child)
                    father_strategy.before_hash_name.append(str_hash_father)
                    child_strategy.before_hash_name.append(str_hash_child)
                    child_strategy.before_hash_name.append(str_hash_father)



                    father_strategy.before_hash=list(set(father_strategy.before_hash))  # 去重
                    child_strategy.before_hash=list(set(child_strategy.before_hash))
                    father_strategy.before_hash_name=list(set(father_strategy.before_hash_name))
                    child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))


                
            else:
                logger.info('hash state pass! add hash state file')
                # hash_pc 
                father_strategy.before_hash.append(hash_pc)
                father_strategy.before_hash.append(hash_cp)
                child_strategy.before_hash.append(hash_pc)
                child_strategy.before_hash.append(hash_cp)
                father_strategy.before_hash_name.append(str_hash_child)
                father_strategy.before_hash_name.append(str_hash_father)
                child_strategy.before_hash_name.append(str_hash_child)
                child_strategy.before_hash_name.append(str_hash_father)
        
                father_strategy.before_hash=list(set(father_strategy.before_hash))  
                child_strategy.before_hash=list(set(child_strategy.before_hash))
                father_strategy.before_hash_name=list(set(father_strategy.before_hash_name))
                child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))

        # snort yes suricata no | suricata yes snort no   very very interesting
        if snort_overall_change ^ suricata_overall_change:
            father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2)  
            child_strategy.mutate_weight+=2   
            child_strategy.saved_reason='++'
            logger.info('very very interesting')
           

            if minor_add_flag:
                logger.info('minor_add_flag find in very very interesting')
                child_strategy.mutate_weight-=2
            else:
                if father_strategy.no_good_count >=2:
                    father_strategy.no_good_count -= 2  
                else:
                    father_strategy.no_good_count = 0
                child_strategy.no_good_count = 0
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)

            child_add_more=True
        # snort yes suricata yes   little  interesting
        if snort_overall_change==True and suricata_overall_change==True:
            #chosen.append(copy.deepcopy(child_strategy))
            father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2) 
            logger.info('little interesting')
            child_strategy.saved_reason='+'
            if minor_add_flag:
                logger.info('minor_add_flag find in little interesting')
                if child_strategy.mutate_weight>=1: child_strategy.mutate_weight-=1
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)

        # snort no suricata no   bad 
        if snort_overall_change ==False and suricata_overall_change==False:
            father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/3) 
            father_strategy.no_good_count+=1
            logger.info('father_strategy no_good_count add, from %d to %d',father_strategy.no_good_count-1,father_strategy.no_good_count)
            child_strategy.mutate_weight=0
            logger.info('bad')
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)


        # add strategy
        
        if father_add_more:
            for i in range(0,father_strategy.mutate_weight):
                chosen.append(copy.deepcopy(father_strategy))
                logger.info('add father_strategy:')
                logger.info('no_good_count:%d ; mutate_weight:%d',father_strategy.no_good_count,father_strategy.mutate_weight)
        else:
            if father_strategy.no_good_count<=4:
                if hash(str(father_strategy)) not in father_chosen_hash:

                    chosen.append(copy.deepcopy(father_strategy))
                    logger.info('add father_strategy:')
                    father_chosen_hash.append(hash(str(father_strategy)))
                    logger.info('no_good_count:%d ; mutate_weight:%d',father_strategy.no_good_count,father_strategy.mutate_weight)
                else:
                    logger.info('father_strategy add once, we do not add again')
            else:
                logger.info('no_good_count excess, do not add father')
        
        for i in range(0,child_strategy.mutate_weight):
            chosen.append(copy.deepcopy(child_strategy))
            logger.info('add child_strategy:')
            logger.info('no_good_count:%d ; mutate_weight:%d',child_strategy.no_good_count,child_strategy.mutate_weight)
 

        
    # canary round(father_strategy doesn't run/ can not compare)
    if canary_round == True:
        return individuals

            

    logger.info('chosen length: %d',len(chosen))
    
    return chosen

def selection_next_generation_vertical_overall_afl_mode(logger,individuals, seed_ind, gen):
    """
    Select individual from the last generation based on states change
    Core thought: if one packet in the packets sequence can cause DPI0 state change and don't cause DPI1 state change. That's a good sign
    ***vertical-overall-version***


    Args:
        individuals (list): A list of individuals to select from.
        seed_ind (Strategy): Current parent individual to compare with
        gen (int):
        k (int): The number of individuals to select.
        tournsize (int): The number of individuals participating in each tournament.
        fit_attr: The attribute of individuals to use as selection criterion (defaults to "fitness")

    Returns:
        list: A list of selected individuals. 
    """

    chosen = []
    father_chosen_hash = []
    canary_round=False
    father_strategy=seed_ind
    handicap = 0
    for i in range(len(individuals)):
        logger.info('---------------------------------------------------------')
        if individuals[i].useless == True:
            continue
        individuals[i].father_inconsistent_packet_num=father_strategy.inconsistent_packet_num

        child_strategy=individuals[i]  
        child_strategy.first_added_time = gen
        #
        child_strategy.father_suricata_state_change_overall=father_strategy.suricata_state_change_overall
        child_strategy.father_snort_state_change_overall_client=father_strategy.snort_state_change_overall_client
        child_strategy.father_snort_state_change_overall_server=father_strategy.snort_state_change_overall_server
        child_strategy.depth = father_strategy.depth + 1
        decide_append_child=False
        decide_append_father=False
        decide_great_add_child=False
        snort_overall_change = False
        suricata_overall_change = False
        child_add_more = False
        father_add_more = False
        # father_strartgy vs child_strategy change/not change?
        if father_strategy.snort_state_change_overall_client !=child_strategy.snort_state_change_overall_client or father_strategy.snort_state_change_overall_server !=child_strategy.snort_state_change_overall_server:
            snort_overall_change=True
        else:
            snort_overall_change=False
        if father_strategy.suricata_state_change_overall !=child_strategy.suricata_state_change_overall:
            suricata_overall_change=True
        else:
            suricata_overall_change=False
        
        
        if child_strategy.suricata_state_change_overall == [] or child_strategy.snort_state_change_overall_client == [] or child_strategy.snort_state_change_overall_server == []:
            suricata_overall_change=False
            snort_overall_change=False
        minor_add_flag=False 
        
        if suricata_overall_change or snort_overall_change: 
            parent_string=str(father_strategy.snort_state_change_overall_client)+str(father_strategy.snort_state_change_overall_server)+str(father_strategy.suricata_state_change_overall)
            child_string=str(child_strategy.snort_state_change_overall_client)+str(child_strategy.snort_state_change_overall_server)+str(child_strategy.suricata_state_change_overall)
            hash_pc=hash(parent_string+child_string)
            hash_cp=hash(child_string+parent_string)
            str_hash_father=str(father_strategy)
            str_hash_child=str(child_strategy)
            
            state_change_happened_before=False
            if (hash_pc in father_strategy.before_hash) or (hash_cp in child_strategy.before_hash) or (hash_pc in child_strategy.before_hash) or (hash_cp in father_strategy.before_hash):
                logger.info('hash state happen before! doing hash name compare') 
                state_change_happend_before=True

                if str_hash_father in child_strategy.before_hash_name or str_hash_child in father_strategy.before_hash_name:
                    logger.info('hash name compare fail! find same strategy! We do not add')
                    snort_overall_change = False
                    suricata_overall_change = False
                else:
                    minor_add_flag = True
                    logger.info('same state change but different strategy name! minor_add')
                    father_strategy.before_hash.append(hash_pc)
                    father_strategy.before_hash.append(hash_cp)
                    child_strategy.before_hash.append(hash_pc)
                    child_strategy.before_hash.append(hash_cp)
                    father_strategy.before_hash_name.append(str_hash_child)
                    father_strategy.before_hash_name.append(str_hash_father)
                    child_strategy.before_hash_name.append(str_hash_child)
                    child_strategy.before_hash_name.append(str_hash_father)



                    father_strategy.before_hash=list(set(father_strategy.before_hash))  
                    child_strategy.before_hash=list(set(child_strategy.before_hash))
                    father_strategy.before_hash_name=list(set(father_strategy.before_hash_name))
                    child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))


                
            else:
                logger.info('hash state pass! add hash state file It is a new hash state!')
                # hash_pc 
                father_strategy.before_hash.append(hash_pc)
                father_strategy.before_hash.append(hash_cp)
                child_strategy.before_hash.append(hash_pc)
                child_strategy.before_hash.append(hash_cp)
                father_strategy.before_hash_name.append(str_hash_child)
                father_strategy.before_hash_name.append(str_hash_father)
                child_strategy.before_hash_name.append(str_hash_child)
                child_strategy.before_hash_name.append(str_hash_father)
        
                father_strategy.before_hash=list(set(father_strategy.before_hash))  
                child_strategy.before_hash=list(set(child_strategy.before_hash))
                father_strategy.before_hash_name=list(set(father_strategy.before_hash_name))
                child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))

        # snort yes suricata no | suricata yes snort no   very very interesting
        if snort_overall_change ^ suricata_overall_change:
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2)  
            child_strategy.mutate_weight+=2   
            child_strategy.saved_reason='++'
            logger.info('very very interesting')
            logger.info('father_ID:%s %s',father_strategy.environment_id,str(father_strategy))
            logger.info('child_ID:%s %s',child_strategy.environment_id,str(child_strategy))
            if suricata_overall_change:
                logger.info(str(father_strategy.suricata_state_change_overall))
                logger.info(str(child_strategy.suricata_state_change_overall))
            if snort_overall_change:
                logger.info(str(father_strategy.snort_state_change_overall_client))
                logger.info(str(child_strategy.snort_state_change_overall_client))
                logger.info(str(father_strategy.snort_state_change_overall_server))
                logger.info(str(child_strategy.snort_state_change_overall_server))
           

            if minor_add_flag:
                logger.info('minor_add_flag find in very very interesting')
                child_strategy.mutate_weight-=2
            else:
                if father_strategy.no_good_count >=2:
                    father_strategy.no_good_count -= 2  
                else:
                    father_strategy.no_good_count = 0
                child_strategy.no_good_count = 0
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)

            child_add_more=True
        # snort yes suricata yes   little  interesting
        if snort_overall_change==True and suricata_overall_change==True:
            #chosen.append(copy.deepcopy(child_strategy))
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2) 
            logger.info('little interesting')
            logger.info('father_ID:%s %s',father_strategy.environment_id,str(father_strategy))
            logger.info('child_ID:%s %s',child_strategy.environment_id,str(child_strategy))
            if suricata_overall_change:
                logger.info(str(father_strategy.suricata_state_change_overall))
                logger.info(str(child_strategy.suricata_state_change_overall))
            if snort_overall_change:
                logger.info(str(father_strategy.snort_state_change_overall_client))
                logger.info(str(child_strategy.snort_state_change_overall_client))
                logger.info(str(father_strategy.snort_state_change_overall_server))
                logger.info(str(child_strategy.snort_state_change_overall_server))
            child_strategy.saved_reason='+'
            if minor_add_flag:
                logger.info('minor_add_flag find in little interesting')
                if child_strategy.mutate_weight>=1: child_strategy.mutate_weight-=1
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)
            child_strategy.no_good_count = 0
        # snort no suricata no   bad 
        if snort_overall_change ==False and suricata_overall_change==False:
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/3) 
            father_strategy.no_good_count+=1
            logger.info('father_strategy no_good_count add, from %d to %d',father_strategy.no_good_count-1,father_strategy.no_good_count)
            child_strategy.mutate_weight=0
            logger.info('bad')
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)
            handicap+=1

        # add strategy
        if child_strategy.saved_reason=='+' or child_strategy.saved_reason=='++':
            child_strategy.handicap = copy.deepcopy(handicap)
            chosen.append(copy.deepcopy(child_strategy))  
            handicap-=1


        # if father_add_more:
        #     for i in range(0,father_strategy.mutate_weight):
        #         chosen.append(copy.deepcopy(father_strategy))
        #         logger.info('add father_strategy:')
        #         logger.info('no_good_count:%d ; mutate_weight:%d',father_strategy.no_good_count,father_strategy.mutate_weight)
        # else:
        #     if father_strategy.no_good_count<=4:
        #         if hash(str(father_strategy)) not in father_chosen_hash:

        #             chosen.append(copy.deepcopy(father_strategy))
        #             logger.info('add father_strategy:')
        #             father_chosen_hash.append(hash(str(father_strategy)))
        #             logger.info('no_good_count:%d ; mutate_weight:%d',father_strategy.no_good_count,father_strategy.mutate_weight)
        #         else:
        #             logger.info('father_strategy add once, we do not add again')
        #     else:
        #         logger.info('no_good_count excess, do not add father')
        
        # for i in range(0,child_strategy.mutate_weight):
        #     chosen.append(copy.deepcopy(child_strategy))
        #     logger.info('add child_strategy:')
        #     logger.info('no_good_count:%d ; mutate_weight:%d',child_strategy.no_good_count,child_strategy.mutate_weight)
 

        
    # canary round(father_strategy doesn't run/ can not compare)
    # if canary_round == True:
    #     return individuals

    for i in range(len(chosen)):
        chosen[i].before_hash=copy.deepcopy(father_strategy.before_hash)
        chosen[i].before_hash_name=copy.deepcopy(father_strategy.before_hash_name)
        chosen[i].perf_score = calculate_score(chosen[i])
            

    logger.info('chosen length: %d',len(chosen))
    
    #if len(chosen)> 20000:  
    #    chosen=chosen[:20000]
    #    logger.info('chosen exceed, shrink to 20000')
    return chosen

def calculate_score(individual):
    """
    Like afl, we calculate mutate weight based on individual's depth handicap and init mutate_weight

    Args:
        individual (Strategy): Single Strategy
    """
    if individual.mutate_weight <=8: perf_score = 1000     # 8 100 |2
    #elif individual.mutate_weight <=4: perf_score = 200
    elif individual.mutate_weight <=16: perf_score = 1500   # 16 200 | 8 400
    elif individual.mutate_weight <=24: perf_score = 2000  # 24 400 | 12 800
    else :                              perf_score = 3000   # 800 | 1200

    # Adjust score based on handicap . Handicap is proportional to how late in the game we learned about this path. Latecommers are allowed to run for a bit longer 
    # until they catch up 
    if (individual.handicap >= 4):
        perf_score *=2     # 4
    elif (individual.handicap):
        perf_score *=1     # 2

    # Adjust score based on input depth , under the assumption that fuzzing deeper test cases is more likely to reveal stuff that can't be discovered with traditional fuzzers

    if   individual.depth < 4 : pass
    elif individual.depth >=4 and individual.depth <=7 : perf_score*=1.3  # >=4 <=7 2
    elif individual.depth >=8 and individual.depth <=13 : perf_score*=2   # 3
    elif individual.depth >=14 and individual.depth <=25 : perf_score*=3  # 4
    else :                                                 perf_score*=4  # 5

    # reduce + mutate_time
    if individual.saved_reason == '+':
        perf_score = perf_score/2
    # Make sure that we don't go over limit
    if (perf_score > 40 * 100) : perf_score = 4000

    return perf_score/2

def selection_next_generation_vertical_overall_afl_mode_new(logger,individuals, seed_ind, gen):
    """
    Select individual from the last generation based on states change
    Core thought: if one packet in the packets sequence can cause DPI0 state change and don't cause DPI1 state change. That's a good sign
    ***vertical-overall-version***


    Args:
        individuals (list): A list of individuals to select from.
        seed_ind (Strategy): Current parent individual to compare with
        gen (int): 
        k (int): The number of individuals to select.
        tournsize (int): The number of individuals participating in each tournament.
        fit_attr: The attribute of individuals to use as selection criterion (defaults to "fitness")

    Returns:
        list: A list of selected individuals. 
    """

    chosen = []
    father_chosen_hash = []
    canary_round=False
    father_strategy=seed_ind
    handicap = 0
    for i in range(len(individuals)):
        print(individuals[i].pretty_print_partial())
        logger.info('---------------------------------------------------------')
        if individuals[i].useless == True:
            logger.info('This individual\'s useless symbol is True, we continue to next one.')
            continue
        if str(individuals[i])==r' \/ ':  
            logger.info('This individual is empty strategy, we continue to next one.')
            continue
        if len(individuals[i].suricata_state_change_overall)==1 and len(individuals[i].snort_state_change_overall_client)==1 and len(individuals[i].snort_state_change_overall_server)==1:
            logger.info('This individual only has one state. negative. we continue to next one.')
    
        action_tree_count=0
        for action_tree in individuals[i].out_actions:
            action_tree_count+=1
        if action_tree_count>3:
            logger.info('This individual contains more than three action tree, we continue to next one')
            continue
        #individuals[i].father_inconsistent_packet_num=father_strategy.inconsistent_packet_num

        child_strategy=individuals[i] 
        child_strategy.first_added_time = gen
      
        child_strategy.father_suricata_state_change_overall=father_strategy.suricata_state_change_overall
        child_strategy.father_snort_state_change_overall_client=father_strategy.snort_state_change_overall_client
        child_strategy.father_snort_state_change_overall_server=father_strategy.snort_state_change_overall_server
        child_strategy.depth = father_strategy.depth + 1
        decide_append_child=False
        decide_append_father=False
        decide_great_add_child=False
        snort_overall_change = False
        suricata_overall_change = False
        child_add_more = False
        father_add_more = False
        # father_strartgy vs child_strategy change/not change?
        if father_strategy.snort_state_change_overall_client !=child_strategy.snort_state_change_overall_client or father_strategy.snort_state_change_overall_server !=child_strategy.snort_state_change_overall_server:
            snort_overall_change=True
        else:
            snort_overall_change=False
        if father_strategy.suricata_state_change_overall !=child_strategy.suricata_state_change_overall:
            suricata_overall_change=True
        else:
            suricata_overall_change=False
        
        
        if child_strategy.suricata_state_change_overall == [] or child_strategy.snort_state_change_overall_client == [] or child_strategy.snort_state_change_overall_server == []:
            suricata_overall_change=False
            snort_overall_change=False
        minor_add_flag=False 
        
        if suricata_overall_change or snort_overall_change: 
            parent_string=str(father_strategy.snort_state_change_overall_client)+str(father_strategy.snort_state_change_overall_server)+str(father_strategy.suricata_state_change_overall)
            child_string=str(child_strategy.snort_state_change_overall_client)+str(child_strategy.snort_state_change_overall_server)+str(child_strategy.suricata_state_change_overall)
            hash_pc=hash(parent_string+child_string) 
            hash_cp=hash(child_string+parent_string)
            
            str_hash_father=father_strategy.pretty_print_partial()
            str_hash_child=child_strategy.pretty_print_partial()
            
            state_change_happened_before=False
            if (hash_pc in father_strategy.before_hash) or (hash_cp in child_strategy.before_hash) or (hash_pc in child_strategy.before_hash) or (hash_cp in father_strategy.before_hash):
                logger.info('hash state happen before! doing hash name compare') 
                state_change_happened_before=True

                if str_hash_father in child_strategy.before_hash_name or str_hash_child in father_strategy.before_hash_name:
                    logger.info('hash name compare fail! find same form strategy! We do not add')
                    snort_overall_change = False
                    suricata_overall_change = False
                else:
                    minor_add_flag = True
                    logger.info('same state change but different strategy name! minor_add')
                    father_strategy.before_hash.append(hash_pc)
                    father_strategy.before_hash.append(hash_cp)
                    child_strategy.before_hash.append(hash_pc)
                    child_strategy.before_hash.append(hash_cp)
                    father_strategy.before_hash_name.append(str_hash_child)
                    father_strategy.before_hash_name.append(str_hash_father)
                    child_strategy.before_hash_name.append(str_hash_child)
                    child_strategy.before_hash_name.append(str_hash_father)



                    father_strategy.before_hash=list(set(father_strategy.before_hash))  
                    child_strategy.before_hash=list(set(child_strategy.before_hash))
                    father_strategy.before_hash_name=list(set(father_strategy.before_hash_name)) 
                    child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))


                
            else: 
                logger.info('hash state pass! add hash state file')
                # hash_pc 
                father_strategy.before_hash.append(hash_pc)
                father_strategy.before_hash.append(hash_cp)
                child_strategy.before_hash.append(hash_pc)
                child_strategy.before_hash.append(hash_cp)
                father_strategy.before_hash_name.append(str_hash_child)
                father_strategy.before_hash_name.append(str_hash_father)
                child_strategy.before_hash_name.append(str_hash_child)
                child_strategy.before_hash_name.append(str_hash_father)
        
                father_strategy.before_hash=list(set(father_strategy.before_hash))  
                child_strategy.before_hash=list(set(child_strategy.before_hash))
                father_strategy.before_hash_name=list(set(father_strategy.before_hash_name)) 
                child_strategy.before_hash_name=list(set(child_strategy.before_hash_name))

        # snort yes suricata no | suricata yes snort no   very very interesting
        if snort_overall_change==True and suricata_overall_change==False:
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2)  
            child_strategy.mutate_weight+=2   
            child_strategy.saved_reason='++'
            logger.info('very very interesting')
            logger.info('father_ID:%s %s',father_strategy.environment_id,str(father_strategy))
            logger.info('child_ID:%s %s',child_strategy.environment_id,str(child_strategy))
            logger.info('father_ID state_hash num: %d,name hash num: %d',len(father_strategy.before_hash),len(father_strategy.before_hash_name))
            logger.info('child_ID state_hash num: %d,name hash num: %d',len(child_strategy.before_hash),len(child_strategy.before_hash_name))
            #if suricata_overall_change:
            logger.info(str(father_strategy.suricata_state_change_overall))
            logger.info(str(child_strategy.suricata_state_change_overall))
            #if snort_overall_change:
            logger.info(str(father_strategy.snort_state_change_overall_client))
            logger.info(str(child_strategy.snort_state_change_overall_client))
            logger.info(str(father_strategy.snort_state_change_overall_server))
            logger.info(str(child_strategy.snort_state_change_overall_server))
           

            if minor_add_flag:
                logger.info('minor_add_flag find in very very interesting')
                child_strategy.mutate_weight-=2
            else:
                if father_strategy.no_good_count >=2:
                    father_strategy.no_good_count -= 2  
                else:
                    father_strategy.no_good_count = 0
                child_strategy.no_good_count = 0
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)

            child_add_more=True
        # snort yes suricata yes   little  interesting
        if (snort_overall_change==True and suricata_overall_change==True) or (snort_overall_change==False and suricata_overall_change==True):
            #chosen.append(copy.deepcopy(child_strategy))
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/2) 
            logger.info('little interesting')
            child_strategy.saved_reason='+'
            logger.info('father_ID:%s %s',father_strategy.environment_id,str(father_strategy))
            logger.info('child_ID:%s %s',child_strategy.environment_id,str(child_strategy))
            logger.info('father_ID state_hash num: %d,name hash num: %d',len(father_strategy.before_hash),len(father_strategy.before_hash_name))
            logger.info('child_ID state_hash num: %d,name hash num: %d',len(child_strategy.before_hash),len(child_strategy.before_hash_name))
            logger.info(str(father_strategy.suricata_state_change_overall))
            logger.info(str(child_strategy.suricata_state_change_overall))
            logger.info(str(father_strategy.snort_state_change_overall_client))
            logger.info(str(child_strategy.snort_state_change_overall_client))
            logger.info(str(father_strategy.snort_state_change_overall_server))
            logger.info(str(child_strategy.snort_state_change_overall_server))
            if minor_add_flag:
                logger.info('minor_add_flag find in little interesting')
                if child_strategy.mutate_weight>=1: child_strategy.mutate_weight-=1
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)
            child_strategy.no_good_count = 0
        # snort no suricata no   bad 
        if snort_overall_change ==False and suricata_overall_change==False:
            #father_strategy.mutate_weight=math.ceil(father_strategy.mutate_weight/3) 
            father_strategy.no_good_count+=1
            logger.info('father_strategy no_good_count add, from %d to %d',father_strategy.no_good_count-1,father_strategy.no_good_count)
            child_strategy.mutate_weight=0
            logger.info('bad')
            logger.info('parent mutate_weight:%d| child mutate_weight:%d',father_strategy.mutate_weight,child_strategy.mutate_weight)
            handicap+=1

        # add strategy
        if child_strategy.saved_reason=='+' or child_strategy.saved_reason=='++':
            child_strategy.handicap = handicap
            chosen.append(copy.deepcopy(child_strategy))  
            handicap-=1

    
   
    for i in range(len(chosen)):
        chosen[i].before_hash=copy.deepcopy(father_strategy.before_hash)
        chosen[i].before_hash_name=copy.deepcopy(father_strategy.before_hash_name)
        chosen[i].perf_score = calculate_score(chosen[i])
            

    logger.info('chosen length: %d',len(chosen))
    

    return chosen

def get_unique_population_size(population):
    """
    Computes number of unique individuals in a population.

    Args:
        population (list): Population list
    """
    uniques = {}
    for ind in population:
        uniques[str(ind)] = True
    return len(list(uniques.keys()))


def add_to_hof(hof, population):
    """
    Iterates over the current population and updates the hall of fame.
    The hall of fame is a dictionary that tracks the fitness of every
    run of every strategy ever.

    Args:
        hof (dict): Current hall of fame
        population (list): Population list

    Returns:
        dict: Updated hall of fame
    """
    for ind in population:
        if str(ind) not in hof:
            hof[str(ind)] = []
        hof[str(ind)].append(ind.fitness)

    return hof


def generate_strategy(logger, num_in_trees, num_out_trees, num_in_actions, num_out_actions, seed, environment_id=None, disabled=None):
    """
    Generates a strategy individual.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        num_in_trees (int): Number of trees to initialize in the inbound forest
        num_out_trees (int): Number of trees to initialize in the outbound forest
        num_in_actions (int): Number of actions to initialize in the each inbound tree
        num_out_actions (int): Number of actions to initialize in the each outbound tree
        environment_id (str, optional): Environment ID to assign to the new individual
        disabled (str, optional): List of actions that should not be considered in building a new strategy

    Returns:
        :obj:`actions.strategy.Strategy`: A new strategy object
    """
    try:
        strat = actions.strategy.Strategy([], [], environment_id=environment_id)
        strat.initialize(logger, num_in_trees, num_out_trees, num_in_actions, num_out_actions, seed, disabled=disabled)
    except Exception:
        logger.exception("Failure to generate strategy")
        raise

    return strat


def mutation_crossover(logger, population, hall, options):
    """
    Apply crossover and mutation on the offspring.

    Hall is a copy of the hall of fame, used to accept or reject mutations.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        population (list): Population of individuals
        hall (dict): Current hall of fame
        options (dict): Options to override settings. Accepted keys are:
            "crossover_pb" (float): probability of crossover
            "mutation_pb" (float): probability of mutation
            "allowed_retries" (int): number of times a strategy is allowed to exist in the hall of fame.
            "no_reject_empty" (bool): whether or not empty strategies should be rejected

    Returns:
        list: New population after mutation
    """
    cxpb = options.get("crossover_pb", 0.5)
    mutpb = options.get("mutation_pb", 0.5)

    offspring = copy.deepcopy(population)
    


    for i in range(len(offspring)):
        recover_offspring_variable(offspring[i]) 
        if random.random() < mutpb:

            mutation_accepted = False
            while not mutation_accepted:
                test_subject = copy.deepcopy(offspring[i])
                test_subject.father_environment_id = offspring[i].environment_id 
                mutate_individual(logger, test_subject)
                test_subject_string=str(test_subject)
                #if (test_subject_string.find('TCP:flags:replace:R')!=-1 and \
                if ( \
                    (test_subject_string.find('TCP:options-md5header:replace')!=-1 \
                    or test_subject_string.find('TCP:options-timestamp:replace')!=-1)) and random.random() < 0.8:
                    continue
                # Pull out some metadata about this proposed mutation
                fitness_history = hall.get(str(test_subject), [])

                # If we've seen this strategy 10 times before and it has always failed,
                # or if we have seen it 20 times already, or if it is an empty strategy,
                # reject this mutation and get another
                if len(fitness_history) >= 10 and all(fitness < 0 for fitness in fitness_history) or \
                   len(fitness_history) >= options.get("allowed_retries", 20) or \
                   (len(test_subject) == 0 and not options.get("no_reject_empty")):
                    mutation_accepted = False
                else:
                    mutation_accepted = True
            
            offspring[i] = test_subject
            offspring[i].fitness =-1000
            



    return offspring

def mutation_crossover_new(logger, task_sequence, hall, options, population):
    """
    Apply crossover and mutation on the offspring.

    Hall is a copy of the hall of fame, used to accept or reject mutations.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        task_sequence (list): Task sequence of individuals
        hall (dict): Current hall of fame
        options (dict): Options to override settings. Accepted keys are:
            "crossover_pb" (float): probability of crossover
            "mutation_pb" (float): probability of mutation
            "allowed_retries" (int): number of times a strategy is allowed to exist in the hall of fame.
            "no_reject_empty" (bool): whether or not empty strategies should be rejected

    Returns:
        list: New population after mutation
    """
    cxpb = options.get("crossover_pb", 0.9) # before 6.3 0.9 prev 0.5
    mutpb = options.get("mutation_pb", 0.9) # before 6.3 0.9 prev 0.5

    offspring = copy.deepcopy(task_sequence)
    
    for i in range(1, len(offspring), 3):  
       if random.random() < cxpb:
           ind = offspring[i - 1]
           mate_target=copy.deepcopy(population[random.randint(0,len(population)-1)])
           actions.strategy.mate(ind, mate_target, logger,indpb=0.5) 
           offspring[i - 1].fitness = -1000

    for i in range(len(offspring)):
        recover_offspring_variable(offspring[i]) 
        if random.random() < mutpb:

            mutation_accepted = False
            while not mutation_accepted:
                test_subject = copy.deepcopy(offspring[i])
                test_subject.father_environment_id = offspring[i].environment_id  
                mutate_individual(logger, test_subject)
                #test_subject_string=str(test_subject)
                ##if (test_subject_string.find('TCP:flags:replace:R')!=-1 and \
                #if ( \
                #    (test_subject_string.find('TCP:options-md5header:replace')!=-1 \
                #    or test_subject_string.find('TCP:options-timestamp:replace')!=-1)) and random.random() < 0.8:
                #    continue
                # Pull out some metadata about this proposed mutation
                fitness_history = hall.get(str(test_subject), [])

                # If we've seen this strategy 10 times before and it has always failed,
                # or if we have seen it 20 times already, or if it is an empty strategy,
                # reject this mutation and get another
                if len(fitness_history) >= 10 and all(fitness < 0 for fitness in fitness_history) or \
                   len(fitness_history) >= options.get("allowed_retries", 20) or \
                   (len(test_subject) == 0 and not options.get("no_reject_empty")):
                    mutation_accepted = False
                else:
                    mutation_accepted = True
            
            offspring[i] = test_subject
            offspring[i].fitness =-1000
            



    return offspring

def recover_offspring_variable(offspring):
    """
    recover necessary variable of strategy
    """
    
    offspring.fitness = -1000
    # new added
    offspring.result_snort = None
    offspring.result_suricata = None
    offspring.can_process_packet_compare = False
    offspring.inconsistent_packet_num = 0
    offspring.change_happened_packet_num=[] 
    offspring.terminate_by_server = False
    offspring.send_port_number = None

    offspring.snort_state_change_overall_client=[]   
    offspring.snort_state_change_overall_server=[]
    offspring.suricata_state_change_overall=[]
    offspring.saved_reason=''
    offspring.first_added_time= 0 
    offspring.has_child = False
    offspring.was_fuzzed = False

    


def mutate_individual(logger, ind):
    """
    Simply calls the mutate function of the given individual.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        ind (:obj:`actions.strategy.Strategy`): A strategy individual to mutate

    Returns:
        :obj:`actions.strategy.Strategy`: Mutated individual
    """
    return ind.mutate(logger)


def run_collection_phase(logger, ga_evaluator):
    """Individual mutation works best when it has seen real packets to base
    action and trigger values off of, instead of blindly fuzzing packets.
    Usually, the 0th generation is useless because it hasn't seen any real
    packets yet, and it bases everything off fuzzed data. To combat this, a
    canary phase is done instead.

    In the canary phase, a single dummy individual is evaluated to capture
    packets. Once the packets are captured, they are associated with all of the
    initial population pool, so all of the individuals have some packets to base
    their data off of.

    Since this phase by necessity requires the evaluator, this is only run if
    --no-eval is not specified.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        ga_evaluator (:obj:`evaluator.Evaluator`): An evaluator object to evaluate with

    Returns:
        str: ID of the test 'canary' strategy evaluated to do initial collection
    """
    canary = generate_strategy(logger, 0, 0, 0, 0, None, disabled=[])
    canary_id = ga_evaluator.canary_phase(canary)
    if not canary_id:
        return []
    return canary_id


def write_generation(filename, population):
    """
    Writes the population pool for a specific generation.

    Args:
        filename (str): Name of file to write the generation out to
        population (list): List of individuals to write out
    """
    # Open File as writable
    with open(filename, "w") as fd:
        # Write each individual to file
        for index, individual in enumerate(population):
            if index == len(population) - 1:
                fd.write(str(individual) + ' ID:' + str(individual.environment_id) + ' Cause_inconsistent:' + str(individual.inconsistent_packet_num) + ' Father_ID:' + str(individual.father_environment_id) + ' Cause_inconsistent:' + str(individual.father_inconsistent_packet_num))
            else:
                fd.write(str(individual) + ' ID:' + str(individual.environment_id) + ' Cause_inconsistent:' + str(individual.inconsistent_packet_num) + ' Father_ID:' + str(individual.father_environment_id) + ' Cause_inconsistent:' + str(individual.father_inconsistent_packet_num) + "\n")

def write_state_change(filename,population):
    """
    Writes the state_change for a specific generation.

    Args:
        filename (str): Name of file to write the state_change out to
        population (list): List of individuals to write out
    """
    # Open File as writable
    with open(filename, "w") as fd:
        # Write each individual to file
        fd.write('length of population:' + str(len(population)) + '\n')
        for index, individual in enumerate(population):
            fd.write(str(individual) + ' ID:' + str(individual.environment_id) + ' Father_ID:' + str(individual.father_environment_id) + ' Cause_inconsistent:' + str(individual.father_inconsistent_packet_num)+ "\n")
            result=individual.result_snort
            #print(individual.result_snort)
            #print(individual.result_suricata)
            if result !=None:
                fd.write('Mutate_weight: ' + str(individual.mutate_weight) + '\n')
                fd.write('father_suricata_state vs current_suricata_state\n')
                fd.write(str(individual.father_suricata_state_change_overall)+'\n')
                fd.write(str(individual.suricata_state_change_overall)+'\n')
                fd.write('father_snort_state vs current_snort_state\n')
                fd.write('snort_state_client\n')
                fd.write(str(individual.father_snort_state_change_overall_client)+'\n')
                fd.write(str(individual.snort_state_change_overall_client)+'\n')
                fd.write('snort_state_server\n')
                fd.write(str(individual.father_snort_state_change_overall_server)+'\n')
                fd.write(str(individual.snort_state_change_overall_server)+'\n')              




            
def write_next_generation_info(filename,population,gen):
    """
    Writes the state_change for a specific generation.

    Args:
        filename (str): Name of file to write the state_change out to
        population (list): List of individuals to write out
        gen (int): current generation
    """
    # Open File as writable
    with open(filename, "w") as fd:
        # Write each individual to file
        fd.write('length of population:' + str(len(population)) + '\n')
        mutate_weight_analysis=[]
        child_count=0
        first_added_time_analysis=[]
        saved_reason_analysis=[]
        no_good_count_analysis=[]
        perf_score_analysis=[]
        depth_analysis=[]
        handicap_analysis=[]

        for index, individual in enumerate(population):
            fd.write(str(individual) + ' ID:' + str(individual.environment_id) + ' father_ID:' + str(individual.father_environment_id) )
            if individual.first_added_time == gen:
                child_count+=1
                fd.write(' NEW\n')
                fd.write('perf_score:' + str(individual.perf_score)+' depth:' +str(individual.depth) + ' mutate_weight:' +str(individual.mutate_weight)+' no_good_count:'+str(individual.no_good_count)+' saved_reason:'+ str(individual.saved_reason)+ ' \n' )
            else:
                fd.write(' \n')
                fd.write('perf_score:' + str(individual.perf_score)+' depth:' +str(individual.depth) + ' mutate_weight:' +str(individual.mutate_weight)+' no_good_count:'+str(individual.no_good_count)+ ' \n')
            mutate_weight_analysis.append(individual.mutate_weight)
            first_added_time_analysis.append(individual.first_added_time)
            saved_reason_analysis.append(individual.saved_reason)
            no_good_count_analysis.append(individual.no_good_count)
            perf_score_analysis.append(individual.perf_score)
            depth_analysis.append(individual.depth)
            handicap_analysis.append(individual.handicap)
            #print(individual.result_snort)
            #print(individual.result_suricata)
            display_state=False
            if index==0 or str(individual)!=str(population[index-1]) :
                display_state=True
            if display_state:
                fd.write('father_suricata_state vs current_suricata_state\n')
                fd.write(str(individual.father_suricata_state_change_overall)+'\n')
                fd.write(str(individual.suricata_state_change_overall)+'\n')
                fd.write('father_snort_state vs current_snort_state\n')
                fd.write(str(individual.father_snort_state_change_overall_client)+'\n')
                fd.write(str(individual.snort_state_change_overall_client)+'\n')
                fd.write('&\n')
                fd.write(str(individual.father_snort_state_change_overall_server)+'\n')
                fd.write(str(individual.snort_state_change_overall_server)+'\n')
        # summarize
        fd.write('new add child:'+str(child_count)+'\n')
        mutate_weight_result=pd.value_counts(mutate_weight_analysis)
        first_added_time_result=pd.value_counts(first_added_time_analysis)
        saved_reason_result=pd.value_counts(saved_reason_analysis)
        no_good_count_result=pd.value_counts(no_good_count_analysis)
        perf_score_result=pd.value_counts(perf_score_analysis)
        depth_result=pd.value_counts(depth_analysis)
        handicap_result=pd.value_counts(handicap_analysis)
        fd.write('mutate_weight:\n')
        fd.write(str(mutate_weight_result)+'\n')
        fd.write('saved_reason:\n')
        fd.write(str(saved_reason_result)+'\n')
        fd.write('first_added_time:\n')
        fd.write(str(first_added_time_result)+'\n')
        fd.write('no_good_count:\n')
        fd.write(str(no_good_count_result)+'\n')
        fd.write('perf_score:\n')
        fd.write(str(perf_score_result)+'\n')
        fd.write('depth:\n')
        fd.write(str(depth_result)+'\n')
        fd.write('handicap:\n')
        fd.write(str(handicap_result)+'\n')


def load_generation(logger, filename):
    """
    Loads strategies from a file

    Args:
        logger (:obj:`logger.Logger`): A logger to log with
        filename (str): Filename of file containing newline separated strategies
            to read generation from
    """
    population = []
    with open(filename) as file:

        # Read each individual from file
        for individual in file:
            strategy = actions.utils.parse(individual, logger)
            population.append(strategy)

    return population


def initialize_population(logger, options, canary_id, disabled=None):
    """
    Initializes the population from either random strategies or strategies
    located in a file.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        options (dict): Options to respect in generating initial population.
            Options that can be specified as keys:

            "load_from" (str, optional): File to load population from
            population_size (int): Size of population to initialize

            "in-trees" (int): Number of trees to initialize in inbound forest
            of each individual

            "out-trees" (int): Number of trees to initialize in outbound forest
            of each individual

            "in-actions" (int): Number of actions to initialize in each
            inbound tree of each individual

            "out-actions" (int): Number of actions to initialize in each
            outbound tree of each individual

            "seed" (str): Strategy to seed this pool with
        canary_id (str): ID of the canary strategy, used to associate each new
            strategy with the packets captured during the canary phase
        disabled (list, optional): List of actions that are disabled

    Returns:
        list: New population of individuals
    """

    if options.get("load_from"):
        # Load the population from a file
        return load_generation(logger, options["load_from"])

    # Generate random strategies
    population = []

    for _ in range(options["population_size"]): 
        p = generate_strategy(logger, options["in-trees"], options["out-trees"], options["in-actions"],
                              options["out-actions"], options["seed"], environment_id=canary_id,
                              disabled=disabled)
        population.append(p)

    return population


def genetic_solve(logger, options, ga_evaluator):
    """
    Run genetic algorithm with given options.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        options (dict): Options to respect.
        ga_evaluator (:obj:`evaluator.Evaluator`): Evaluator to evaluate
            strategies with

    Returns:
        dict: Hall of fame of individuals
    """
    # Directory to save off each generation so evolution can be resumed
    ga_generations_dir = os.path.join(actions.utils.RUN_DIRECTORY, "generations")
    # Directory to save off each generation state_change so evolution can be resumed
    ga_state_change_dir = os.path.join(actions.utils.RUN_DIRECTORY, "state_changes")
    ga_success_dir = os.path.join(actions.utils.RUN_DIRECTORY, "success")
    hall = {}
    canary_id = None
    if ga_evaluator and not options["no-canary"]:
        canary_id = run_collection_phase(logger, ga_evaluator)
    else:
        logger.info("Skipping initial collection phase.")

    population = initialize_population(logger, options, canary_id, disabled=options["disable_action"])

    try:
        offspring = []
        elite_clones = []
        if options["seed"]:
            elite_clones = [actions.utils.parse(options["seed"], logger)]
        # afl_mode
        population = fitness_function(logger, population, ga_evaluator) # run the task first
        gen = 0
        filename = os.path.join(ga_generations_dir, "generation" + str(gen) + ".txt")
        write_generation(filename, population)
        while(True):
            if(len(population)==0): break
            gen+=1
            best_fit, best_ind = -10000, None
            fuzz_time=0
            strategy_index=0
            prior_population=[]  # prior_queue 
            current_seed_source='population'
            while strategy_index < len(population):  
                pickpb=random.random()
                if len(prior_population)!=0 and pickpb < 0.66:  
                    logger.info("seed picked from prior_population.") 
                    current_seed_source='prior'
                    population.append(copy.deepcopy(prior_population[0]))
                    seed_ind = population[-1]
                    seed_ind.was_fuzzed = True
                    del prior_population[0]
        
                else:
                    current_seed_source='population'
                    logger.info("seed picked from population.") 
                    seed_ind = population[strategy_index]
                    strategy_index+=1
                
                if seed_ind.was_fuzzed == True and current_seed_source=='population':  
                    logger.info("current item was_fuzzed") 
                    continue
                fuzz_time+=1
                task_sequence = [copy.deepcopy(seed_ind) for x in range(math.ceil(seed_ind.perf_score/100))]   # seed's child
                #offspring = mutation_crossover(logger, task_sequence, hall, options)
                offspring = mutation_crossover_new(logger, task_sequence, hall, options,population)
                # supplement source info based on 'current_seed_source'
                for seed in offspring:
                    if current_seed_source=='prior':
                        seed.source_dict['prior']+=1
                    else:
                        seed.source_dict['population']+=1
                # judge whether offspring should be added 
                offspring = fitness_function(logger, offspring, ga_evaluator)
                offspring = selection_next_generation_vertical_overall_afl_mode_new(logger, offspring, seed_ind, gen)
                logger.info("add %d to structure",len(offspring))
                if len(offspring) !=0:
                    seed_ind.has_child = True # mark the seed_ind has child now
                add2prior=0
                add2pop=0
                for new_ind in offspring:
                    if new_ind.saved_reason == '++':
                        prior_population.append(new_ind)
                        add2prior+=1
                    else:
                        population.append(new_ind)
                        add2pop+=1
                logger.info("add %d to prior & add %d to population",add2prior,add2pop)
                logger.info("current population length: %d",len(population))
                logger.info("current population pointer: %d",strategy_index)
                logger.info("current prior_population length: %d",len(prior_population))

                

                if fuzz_time%50==0:  
                    best_fit, best_ind = -10000, None
                    total_fitness=0
                    # Iterate over the offspring to find the best individual for printing
                    for ind in population:
                        if ind.fitness is None and ga_evaluator:
                            logger.error("No fitness for individual found: %s.", str(ind))
                            continue
                        total_fitness += ind.fitness
                        if ind.fitness is not None and ind.fitness >= best_fit:
                            best_fit = ind.fitness
                            best_ind = ind

                    # Check if any individuals of this generation belong in the hall of fame
                    hall = add_to_hof(hall, population)

                    # Save current hall of fame
                    filename = os.path.join(ga_generations_dir, "hall" + str(gen) +'_'+  str(fuzz_time) + ".txt")
                    write_hall(filename, hall)
                    filename = os.path.join(ga_state_change_dir, "state_change" + str(gen)+ '_'+  str(fuzz_time) + ".txt")
                    write_state_change(filename,population)
                    filename = os.path.join(ga_state_change_dir, "next_generation_info" + str(gen)+ '_'+ str(fuzz_time) + ".txt")
                    write_next_generation_info(filename, population, gen)

                 
            
            total_fitness=0
            # Iterate over the offspring to find the best individual for printing
            for ind in population:
                if ind.fitness is None and ga_evaluator:
                    logger.error("No fitness for individual found: %s.", str(ind))
                    continue
                total_fitness += ind.fitness
                if ind.fitness is not None and ind.fitness >= best_fit:
                    best_fit = ind.fitness
                    best_ind = ind

            # Check if any individuals of this generation belong in the hall of fame
            hall = add_to_hof(hall, population)

            # Save current hall of fame
            filename = os.path.join(ga_generations_dir, "hall_" + str(gen) + "_summary.txt")
            write_hall(filename, hall)
            filename = os.path.join(ga_state_change_dir, "state_change_" + str(gen)+ "_summary.txt")
            write_state_change(filename,population)

            
            population = list(filter(lambda x: x.no_good_count <=4 or (x.no_good_count<=8 and x.has_child ==False), population))

            filename = os.path.join(ga_state_change_dir, "next_generation_info_" + str(gen)+ "_summary.txt")
            write_next_generation_info(filename, population, gen)


    # If the user interrupted, try to gracefully shutdown
    except KeyboardInterrupt:
        # Only need to stop the evaluator if one is defined
        if ga_evaluator:
            ga_evaluator.stop = True
        logger.info("")

    finally:
        if options["force_cleanup"]:
            # Try to clean up any hanging docker containers/networks from the run
            logger.warning("Cleaning up docker...")
            try:
                sp.check_call("docker stop $(docker ps -aq) > /dev/null 2>&1", shell=True)
            except sp.CalledProcessError:
                pass

    return hall


def collect_results(hall_of_fame):
    """
    Collect final results from offspring.

    Args:
        hall_of_fame (dict): Hall of fame of individuals

    Returns:
        str: Formatted printout of the hall of fame
    """
    # Sort first on number of runs, then by fitness.
    best_inds = sorted(hall_of_fame, key=lambda ind: (len(hall_of_fame[ind]), sum(hall_of_fame[ind])/len(hall_of_fame[ind])))
    output = "Results: \n"
    for ind in best_inds:
        sind = str(ind)
        output += "Avg. Fitness %s: %s (Evaluated %d times: %s)\n" % (sum(hall_of_fame[sind])/len(hall_of_fame[sind]), sind, len(hall_of_fame[sind]), hall_of_fame[sind])
    return output


def print_results(hall_of_fame, logger):
    """
    Prints hall of fame.

    Args:
        hall_of_fame (dict): Hall of fame to print
        logger (:obj:`logging.Logger`): A logger to log results with
    """
    logger.info("\n%s", collect_results(hall_of_fame))


def write_hall(filename, hall_of_fame):
    """
    Writes hall of fame out to a file.

    Args:
        filename (str): Filename to write results to
        hall_of_fame (dict): Hall of fame of individuals
    """
    with open(filename, "w") as fd:
        fd.write(collect_results(hall_of_fame))


def eval_only(logger, requested, ga_evaluator, runs=1):
    """
    Parses a string representation of a given strategy and runs it
    through the evaluator.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        requested (str): String representation of requested strategy or filename
            of strategies
        ga_evaluator (:obj:`evaluator.Evaluator`): An evaluator to evaluate with
        runs (int): Number of times each strategy should be evaluated

    Returns:
        float: Success rate of tested strategies
    """
    # The user can specify a file that contains strategies - check first if that is the case
    if os.path.isfile(requested):
        with open(requested, "r") as fd:
            requested_strategies = fd.readlines()
        if not requested_strategies:
            logger.error("No strategies found in %s", requested)
            return None
    else:
        requested_strategies = [requested]
    # We want to override the client's default strategy retry logic to ensure
    # we test to the number of runs requested
    ga_evaluator.runs = 1
    population = []

    for requested in requested_strategies:
        for i in range(runs):
            ind = actions.utils.parse(requested, logger)
            population.append(ind)
        logging.info("Computing fitness for: %s", str(ind))
        logging.info("\n%s", ind.pretty_print())

    fits = []
    success = 0
    # Once the population has been parsed and built, test it
    fitness_function(logger, population, ga_evaluator)
    for strat in population:
        fits.append(strat.fitness)
    i = 0
    logger.info(fits)
    for fitness in fits:
        if fitness > 0:
            success += 1
            logger.info("Trial %d: success! (fitness = %d)", i, fitness)
        else:
            logger.info("Trial %d: failure! (fitness = %d)", i, fitness)
        i += 1
    if fits:
        logger.info("Overall %d/%d = %d%%", success, i, int((float(success)/float(i)) * 100))
    logger.info("Exiting eval-only.")
    return float(success)/float(i)


def restrict_headers(logger, protos, filter_fields, disabled_fields):
    """
    Restricts which protocols/fields can be accessed by the algorithm.

    Args:
        logger (:obj:`logging.Logger`): A logger to log with
        protos (str): Comma separated string of protocols that are allowed
        filter_fields (str): Comma separated string of fields to allow
        disabled_fields (str): Comma separated string of fields to disable
    """
    # Retrieve flag and protocol filters, and validate them
    protos = protos.upper().split(",")
    if filter_fields:
        filter_fields = filter_fields.lower().split(",")
    if disabled_fields:
        disabled_fields = disabled_fields.split(",")

    layers.packet.Packet.restrict_fields(logger, protos, filter_fields, disabled_fields)


def driver(cmd):
    """
    Main workflow driver for the solver. Parses flags and input data, and initiates solving.

    Args:
        cmd (list): sys.argv or a list of arguments

    Returns:
        dict: Hall of fame of individuals
    """
    # Parse the given arguments
    args = get_args(cmd)

    logger = setup_logger(args.log)

    lock_file_path = "/lock_file.txt"
    if not args.no_lock_file and os.path.exists(lock_file_path):
        logger.info("Lock file \"%s\" already exists.", lock_file_path)
        return None

    try:
        if not args.no_lock_file:
            # Create lock file to prevent interference between multiple runs
            open(lock_file_path, "w+")

        # Log the command run
        logger.debug("Launching strategy evolution: %s", " ".join(cmd))
        logger.info("Logging results to %s", logger.ga_log_dir)

        if args.no_eval and args.eval_only:
            print("ERROR: Cannot --eval-only with --no-eval.")
            return None

        requested_strategy = args.eval_only

        # Define an evaluator for this session
        ga_evaluator = None
        if not args.no_eval:
            cmd += ["--output-directory", actions.utils.RUN_DIRECTORY]
            ga_evaluator = evaluator.Evaluator(cmd, logger)

        # Check if the user only wanted to evaluate a single given strategy
        # If so, evaluate it, and exit
        if requested_strategy or requested_strategy == "":
            # Disable evaluator empty strategy skipping
            ga_evaluator.skip_empty = False
            eval_only(logger, requested_strategy, ga_evaluator, runs=args.runs)
            return None

        restrict_headers(logger, args.protos, args.fields, args.disable_fields)

        actions.trigger.GAS_ENABLED = (not args.no_gas)
        if args.fix_trigger:
            actions.trigger.FIXED_TRIGGER = actions.trigger.Trigger.parse(args.fix_trigger)

        requested_seed = args.seed
        if requested_seed or requested_seed == "":
            try:
                requested_seed = actions.utils.parse(args.seed, logger)
            except (TypeError, AssertionError, actions.tree.ActionTreeParseError):
                logger.error("Failed to parse given strategy: %s", requested_seed)
                raise

        # Record all of the options supplied by the user to pass to the GA
        options = {}
        options["no_reject_empty"] = not args.no_reject_empty
        options["population_size"] = args.population
        options["out-trees"] = args.out_trees
        options["in-trees"] = args.in_trees
        options["in-actions"] = args.in_actions
        options["out-actions"] = args.out_actions
        options["force_cleanup"] = args.force_cleanup
        options["num_generations"] = args.generations
        options["seed"] = args.seed
        options["elite_clones"] = args.elite_clones
        options["allowed_retries"] = args.allowed_retries
        options["mutation_pb"] = args.mutation_pb
        options["crossover_pb"] = args.crossover_pb
        options["no-canary"] = args.no_canary
        options["load_from"] = args.load_from

        disable_action = []
        if args.disable_action:
            disable_action = args.disable_action.split(",")
        options["disable_action"] = disable_action

        logger.info("Initializing %d strategies with %d input-action trees and %d output-action trees of input size %d and output size %d for evolution over %d generations.",
                    args.population, args.in_trees, args.out_trees, args.in_actions, args.out_actions, args.generations)

        hall_of_fame = {}
        strategy_hall={}   # type: flags:R options:md5header
        try:
            # Kick off the main genetic driver
            hall_of_fame = genetic_solve(logger, options, ga_evaluator)
        except KeyboardInterrupt:
            logger.info("User shutdown requested.")
        if ga_evaluator:
            ga_evaluator.shutdown()

        if hall_of_fame and not args.no_print_hall:
            # Print the final results
            print_results(hall_of_fame, logger)

        # Teardown the evaluator if needed
        if ga_evaluator:
            ga_evaluator.shutdown()
    finally:
        # Remove lock file
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
    return hall_of_fame


if __name__ == "__main__":
    driver(sys.argv[1:])
