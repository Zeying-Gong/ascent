"""The following imports are necessary for updating the registry"""
try:
    import frontier_exploration.base_explorer
    import frontier_exploration.frontier_detection
    import frontier_exploration.frontier_sensor
    import frontier_exploration.measurements
    import frontier_exploration.objnav_explorer
    import frontier_exploration.policy
    import frontier_exploration.trainer
    import frontier_exploration.utils.inflection_sensor
    import frontier_exploration.utils.multistory_episode_finder
except ModuleNotFoundError as e:
    # If the error was due to the habitat package not being installed, then pass, but
    # print a warning. Do not pass if it was due to another package being missing.
    if "habitat" not in e.name:
        raise e
    else:
        print(
            "Warning: importing habitat failed. Cannot register habitat_baselines "
            "components."
        )
