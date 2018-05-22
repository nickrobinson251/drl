from gym.envs.registration import register

register(id='Valkyrie-v0',
         entry_point='envs.valkyrie:ValkyrieEnv')
