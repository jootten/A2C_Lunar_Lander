from multiprocessing import Process, Queue
import gym

observations = []

def run_env(q):
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()
    observations = []
    for _ in range(66):
        env.render()
        observations.append(env.step(env.action_space.sample()))
    env.close()
    q.put(observations)


def main():
    q_list = []
    proc_list = []

    for i in range(8):
        q_list.append(Queue())
        proc_list.append(Process(target=run_env, args=(q_list[i],)))

    for proc in proc_list:
        proc.start()

    

    for proc in proc_list:
        proc.join()


    for q in q_list:
        while not q.empty():
            print(q.get())

    

    



if __name__ == '__main__':
    main()