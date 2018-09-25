from gym.envs.registration import register

register(id='Valkyrie-v0',
         entry_point='valkyrie.envs.valkyrie_env:ValkyrieEnvBasic')

register(id='InvertedPendulum-v0',
         entry_point='valkyrie.envs.pendulum_env:InvertedPendulumEnv')

register(id='Balancebot-v0',
         entry_point='valkyrie.envs.balancebot_env:BalancebotEnv')
