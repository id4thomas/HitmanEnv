from gym.envs.registration import register

register(
    id='hitman-v0',
    entry_point='hitman_gym.envs:HitmanMap1',
)

register(
    id='hitman-v1',
    entry_point='hitman_gym.envs:HitmanMap2',
)

register(
    id='hitman-v2',
    entry_point='hitman_gym.envs:HitmanMap3',
)