from gym.envs.registration import register

register(id='Valkyrie-v0',
         entry_point='valkyrie.envs.valkyrie_env:ValkyrieEnvBasic')
