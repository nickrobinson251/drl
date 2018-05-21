from gym.envs.registration import register

register(id='balancebot-v0',
         entry_point='balancebot.envs:BalancebotEnv')
