from gym.envs.registration import register

register(
    id='hitman-map1',
    entry_point='hitman_gym.envs:HitmanMap1',
)

register(
    id='hitman-map2',
    entry_point='hitman_gym.envs:HitmanMap2',
)