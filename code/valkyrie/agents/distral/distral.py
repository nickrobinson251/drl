def distill(expert_policy, distil_policy, alpha, beta, gamma):
    alphas = [0.5, 1]
    betas = [3e3, 1e3, 3e4]

    # Calculate loss function according to Equation 1

    # Store three type of losses[
    reward_losses = []
    distill_losses = []
    entropy_losses = []

    # Give format
    alpha = Variable(torch.Tensor([alpha]))
    beta = Variable(torch.Tensor([beta]))

    # Retrive distilled policy actions
    distill_actions = distilled.saved_actions

    # Retrieve policy actions and rewards
    policy_actions = policy.saved_actions
    rewards = policy.rewards

    # Obtain discounts
    R = 1.
    discounts = []
    for r in policy.rewards[::-1]:
        R *= gamma
        discounts.insert(0, R)

    discounts = torch.Tensor(discounts)
    # print(discounts)

    for log_prob_i, log_prob_0, d, r in zip(
            policy_actions, distill_actions, discounts, rewards):
        reward_losses.append(-d * Variable(torch.Tensor([r])))
        distill_losses.append(-((d*alpha)/beta) * log_prob_0)
        entropy_losses.append((d/beta) * log_prob_i)

    #print('Reward Loss: ',torch.stack(reward_losses).sum().data[0])
    #print('Entropy Loss: ',torch.stack(entropy_losses).sum().data[0])
    #print('Distill Loss: ',torch.stack(distill_losses).sum().data[0])

    # Perform optimization step
    opt_policy.zero_grad()
    opt_distilled.zero_grad()

    loss = torch.stack(reward_losses).sum(
        ) + torch.stack(entropy_losses).sum() + torch.stack(distill_losses).sum()

    loss.backward(retain_graph=True)

    # for param in policy.parameters():
    #    param.grad.data.clamp_(-1, 1)

    opt_policy.step()
    opt_distilled.step()

    # Clean memory
    del policy.rewards[:]
    del policy.saved_actions[:]
    del policy.saved_actions[:]


def reward_regularised(action, state, env, policy_0, policy_i, alpha, beta):
    """Reward function regualised by shapng and entropy terms.

    See equation (9) in Distral paper.
    """
    reward = env.reward(action, state)
    reg_shaping = alpha / beta * np.log(policy_0(action, state))
    reg_entropy = 1.0 / beta * np.log(policy_i(action, state))
    return reward + reg_shaping - reg_entropy


def distral_objective(
        policy_0,
        task_policies,
        envs,
        time_steps,
        alpha=0.5,
        beta=1e3,
        gamma=1):
    """

    See equation (1) in Distral paper.
    """
    reward_term = []
    shaping_term = []
    entropy_term = []
    for i, policy_i in enumerate(task_policies):
        state = envs[i].observation
        for t in range(time_steps):
            action = policy_i(state)
            state, reward, done, _ = envs[i].step(action)
            gamma *= gamma
            reward_term.append(gamma * reward)
            shaping_term.append(
                gamma * alpha/beta * np.log(policy_0(action, state)))
            entropy_term.append(gamma / beta * np.log(policy_i(action, state)))
            state, reward, done, _ = envs[i].step(action)
    objectiv = np.sum(reward_term) + np.sum(shaping_term) - np.sum(entropy_term)
    return objectiv
