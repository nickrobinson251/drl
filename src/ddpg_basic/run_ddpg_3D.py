from ddpg import *
from gym import wrappers
import gc
import numpy as np
import time
from logger import logger
from Interpolate import *
from valkyrie_gym_env import Valkyrie, FilterClass
gc.enable()

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

dir_path = 'record/3D/with_external_force_disturbance/2018_05_17_16.16.33'#'2017_05_29_18.23.49/with_force'
#dir_path = 'record/3D/without_external_force_disturbance/2018_03_30_02.19.45'#'2017_05_29_18.23.49/with_force'
MONITOR_DIR = dir_path
def main():
	config = Configuration()
	config.load_configuration(dir_path)
	config.print_configuration()

	ENV_NAME = config.conf['env-id']  # 'HumanoidBalanceFilter-v0'#'HumanoidBalance-v0'
	EPISODES = config.conf['epoch-num']
	TEST = config.conf['test-num']
	step_lim = config.conf['total-step-num']

	episode_count = config.conf['epoch-num']
	action_bounds = config.conf['action-bounds']

	PD_frequency = config.conf['LLC-frequency']
	Physics_frequency = config.conf['Physics-frequency']
	network_frequency = config.conf['HLC-frequency']
	sampling_skip = int(PD_frequency / network_frequency)

	reward_decay = 1.0
	reward_scale = 0.05  # Normalizing the scale of reward to 10#0.1#1.0/sampling_skip#scale down the reward
	max_time=10
	max_steps = int(max_time * network_frequency)

	BEST_REWARD = 0

	EPISODES = 1
	STEPS = 2500000

	force = 600
	impulse = 0.01
	force_chest = [0, 0]  # max(0,force_chest[1]-300*1.0 / EXPLORE)]
	force_pelvis = [0, 0]
	force_period = [5 * PD_frequency, (5 + 0.1) * PD_frequency]  # impulse / force * FPS

	env = Valkyrie(max_time=max_time, renders=True, initial_gap_time=1.0, PD_freq=PD_frequency, Physics_freq=Physics_frequency, Kp=config.conf['Kp'],
                   Kd=config.conf['Kd'], bullet_default_PD=config.conf['bullet-default-PD'],
				   logFileName = os.path.dirname(os.path.realpath(__file__))+'/'+dir_path,
				   controlled_joints_list=config.conf['controlled-joints'])
	#env._setupCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, 0, 1.0])
	#env._setupCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.7])
	env._setupCamera(cameraDistance=2.0, cameraYaw=90, cameraPitch=-15, cameraTargetPosition=[0, 0, 1.0])

	agent = DDPG(env,config)
	agent.load_weight(dir_path)
	agent.ob_normalize1.load_normalization(dir_path)#TODO test observation normalization
	agent.ob_normalize1.print_normalization()#TODO test observation normalization
	agent.ob_normalize2.load_normalization(dir_path)  # TODO test observation normalization
	agent.ob_normalize2.print_normalization()  # TODO test observation normalization

	step_count = 0

	total_reward = 0
	force_chest = [0, 0, 0]
	force_pelvis = [0, 0, 0]

	logging=logger(dir_path)

	t_max = 0#timer
	t_min = 100
	t_total=[]
	prev_action = []

	interpolate = dict()
	if config.conf['joint-interpolation'] == True:
		joint_interpolate = {}
		for joint in config.conf['actor-action-joints']:
			interpolate = JointTrajectoryInterpolate()
			# joint_interpolate[joint] = interpolate
			joint_interpolate.update({joint: interpolate})

	for i in range(TEST):
		#_ = env._reset()
		_ = env._reset(Kp=config.conf['Kp'], Kd=config.conf['Kd'])
		#env.resetJointStates(base_orn_nom=[0,0,0.707,0.707])

		step_count = 0
		total_reward = 0

		_ = agent.reset()
		env._startLoggingVideo()
		action = np.zeros((len(config.conf['actor-action-joints']),))  # 4 dimension output of actor network, hip, knee, waist, ankle
		action_interpolate = np.zeros((len(config.conf['actor-action-joints']),))  # 4 dimension output of actor network, hip, knee, waist, ankle
		control_action = np.zeros((len(config.conf['controlled-joints']),))
		state, reward, done, _ = env._step(control_action)

		for j in range(max_steps):
			#print(env.COM_pos_local)
			#print(env.COM_pos)
			#print(env.linkCOMPos['rightAnklePitch'])
			#step_count += 1  # counting total steps during training

			prev_action = np.array(action)

			#t0 = time.time()
			#update action
			state = env.getExtendedObservation()
			if agent.config.conf['normalize-observations']:
				state_norm = agent.ob_normalize1.normalize(np.asarray(state))
				state_norm = np.reshape(state_norm, (agent.state_dim))  # reshape intp(?,)
			else:
				state_norm = state
			action = agent.action(state_norm)
			action = np.clip(action, action_bounds[0], action_bounds[1])

			#t1 = time.time()

			#total = t1 - t0
			#t_total.append(total)
			#prev_action = action
			reward_add = 0
			#env.render()
			# env.drawCOM()
			# env.getSupportPolygon()
			# env.drawSupportPolygon()

			readings = env.getReading()

			if config.conf['joint-interpolation'] == True:
				# setup joint interpolation
				for n in range(len(config.conf['actor-action-joints'])):
					joint_name = config.conf['actor-action-joints'][n]
					joint_interpolate[joint_name].cubic_interpolation_setup(prev_action[n], 0, action[n], 0,
																			1.0 / float(network_frequency))

			action_org = action

			for k in range(sampling_skip):
				step_count += 1
				if ((step_count >= force_period[0]) and (step_count < force_period[1])):
					if step_count == force_period[0]:
						print(env.rejectableForce(0.1))
						force = env.rejectableForce(0.1)[1]
						force_chest = [0, 0, 0]
						force_pelvis = [force, 0, 0]
					#env.applyForceOnPelvis(force=force_pelvis, pos=[0,0,0])
				else:
					force_chest = [0, 0, 0]
					force_pelvis = [0, 0, 0]
					#env.applyForceOnPelvis(force=force_pelvis, pos=[0, 0, 0])
				if ((step_count >= force_period[0]-250) and (step_count < force_period[1]+250)):
					text = ''#''600N applied on pelvis for 0.1s'
				else:
					text = ''

				if config.conf['joint-interpolation'] == True:
					for n in range(len(config.conf['actor-action-joints'])):
						joint_name = config.conf['actor-action-joints'][n]
						action_interpolate[n] = joint_interpolate[joint_name].interpolate(1.0 / PD_frequency)

				if len(control_action) == 7 and len(action) == 4:
					control_action[0:4] = action_interpolate
					control_action[4:7] = action_interpolate[1:4]  # duplicate leg control signals
				elif len(control_action) == 11 and len(action_interpolate) == 4:
					control_action[0] = action_interpolate[0]  # torso pitch
					control_action[1] = 0.0  # hip roll
					control_action[2] = action_interpolate[1]  # hip pitch
					control_action[3] = action_interpolate[2]  # knee pitch
					control_action[4] = action_interpolate[3]  # ankle pitch
					control_action[5] = 0.0  # ankle roll
					control_action[6:11] = control_action[1:6]
				elif len(control_action) == 13 and len(action_interpolate) == 4:
					control_action[0] = action_interpolate[0]  # torso pitch
					control_action[1] = 0.0  # hip yaw
					control_action[2] = 0.0  # hip roll
					control_action[3] = action_interpolate[1]  # hip pitch
					control_action[4] = action_interpolate[2]  # knee pitch
					control_action[5] = action_interpolate[3]  # ankle pitch
					control_action[6] = 0.0  # ankle roll
					control_action[7:13] = control_action[1:7]
				elif len(control_action) == 11 and len(action_interpolate) == 11:
					control_action[:] = action_interpolate[:]
				elif len(control_action) == 13 and len(action_interpolate) == 11:
					control_action[0] = action_interpolate[0]  # torso pitch
					control_action[1] = 0.0  # hip yaw
					control_action[2] = action_interpolate[1]  # hip roll
					control_action[3] = action_interpolate[2]  # hip pitch
					control_action[4] = action_interpolate[3]  # knee pitch
					control_action[5] = action_interpolate[4]  # ankle pitch
					control_action[6] = action_interpolate[5]  # ankle roll
					control_action[7] = 0.0  # hip yaw
					control_action[8] = action_interpolate[6]  # hip roll
					control_action[9] = action_interpolate[7]  # hip pitch
					control_action[10] = action_interpolate[8]  # knee pitch
					control_action[11] = action_interpolate[9]  # ankle pitch
					control_action[12] = action_interpolate[10]  # ankle roll
				elif len(control_action) == 13 and len(action_interpolate) == 13:
					control_action[:] = action_interpolate[:]

				next_state, reward, done, _ = env._step(control_action, force_pelvis)
				reward_add = reward + reward_decay * reward_add

				logging.add_run('target_joint',np.array(action))
				logging.add_run('interpolated_target_joint',np.array(action_interpolate))

				ob = env.getObservation()
				ob_filtered = env.getFilteredObservation()
				for l in range(len(ob)):
					logging.add_run('observation'+str(l),ob[l])
					logging.add_run('filtered_observation' + str(l), ob_filtered[l])

				readings = env.getExtendedReading()
				for key, value in readings.items():
					logging.add_run(key,value)

			reward = reward_add*reward_scale  # / sampling_skip
			total_reward += reward
			if done:
				break

		env._stopLoggingVideo()
	ave_reward = total_reward / TEST
	logging.save_run()
	print(' Evaluation Average Reward:' + str(ave_reward))
	#t_min=min(t_total)
	#t_max=max(t_total)
	#t_avg=sum(t_total)/float(len(t_total))
	#(t_min,t_max,t_avg)
	#env.monitor.close()

if __name__ == '__main__':
	main()
